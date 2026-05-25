"""
Bot state machine.

Two layers:

1. **Global phase** — coarse time-bucketed state of the trading day.
   Transitions are driven by NY wall-clock time, NOT by events.
   ::

      PRE_MARKET  → BUILDING_OR → HUNTING → AFTERNOON → WIND_DOWN → FORCED_FLAT

2. **Per-symbol state** — FLAT / IN_POSITION / COOLDOWN.  Cooldown ends
   after ``risk.symbol_cooldown_minutes`` from the last exit.

Entry evaluation is gated by:
  • current phase allowing entries (HUNTING, or AFTERNOON in 13:00–15:00),
  • per-symbol state being FLAT,
  • the symbol not being in COOLDOWN,
  • all of the LONG/SHORT entry rules (filled in by Step 6).

The skeleton's ``evaluate()`` currently returns ``Decision(action="skip",
rejected_gate="not_implemented")`` — Step 6 fills the gates in WITHOUT
changing the surrounding orchestration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Callable, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import StrategyConfig, get_strategy_config
from indicators import (
    ema,
    macd,
    rsi,
    rvol_bar,
    session_vwap,
)

NY_TZ = ZoneInfo("America/New_York")


# ── Enums ────────────────────────────────────────────────────────────────

class BotPhase(str, Enum):
    PRE_MARKET   = "pre_market"
    BUILDING_OR  = "building_or"
    HUNTING      = "hunting"           # 09:45 → morning_end
    AFTERNOON    = "afternoon"         # morning_end → 15:30
    WIND_DOWN    = "wind_down"         # 15:30 → forced_flat_time
    FORCED_FLAT  = "forced_flat"       # forced_flat_time onward
    CLOSED       = "closed"            # after 16:00 — no activity at all


class SymbolState(str, Enum):
    FLAT         = "flat"
    IN_POSITION  = "in_position"
    COOLDOWN     = "cooldown"


# ── Decision ────────────────────────────────────────────────────────────

@dataclass
class Decision:
    """The result of evaluating one symbol on one closed bar."""
    action: Literal["skip", "long", "short"]
    rejected_gate: str | None = None
    gates: dict[str, bool] = field(default_factory=dict)
    indicators: dict[str, object] = field(default_factory=dict)  # floats + bar ts

    @property
    def passed(self) -> bool:
        return self.action != "skip"


# ── Candlestick helpers (used by retest entries) ────────────────────────

def _is_bullish_reversal(s: dict[str, float]) -> bool:
    """True if the active bar is a hammer or bullish-engulfing pattern.

    Hammer: small body near the top of the range, long lower wick (≥ 2× body),
    no significant upper wick.  Engulfing: prior bar bearish, current bar
    bullish, current body covers prior body."""
    body = abs(s["close"] - s["open"])
    rng = s["high"] - s["low"]
    if rng <= 0:
        return False
    lower_wick = min(s["open"], s["close"]) - s["low"]
    upper_wick = s["high"] - max(s["open"], s["close"])

    hammer = (
        body <= rng / 3
        and lower_wick >= 2 * body
        and lower_wick >= rng / 2
        and upper_wick <= body
    )
    engulfing = (
        s["prev_close"] < s["prev_open"]      # prior bearish
        and s["close"] > s["open"]            # current bullish
        and s["close"] >= s["prev_open"]
        and s["open"] <= s["prev_close"]
    )
    return hammer or engulfing


def _is_bearish_reversal(s: dict[str, float]) -> bool:
    """Shooting star or bearish engulfing."""
    body = abs(s["close"] - s["open"])
    rng = s["high"] - s["low"]
    if rng <= 0:
        return False
    lower_wick = min(s["open"], s["close"]) - s["low"]
    upper_wick = s["high"] - max(s["open"], s["close"])

    shooting_star = (
        body <= rng / 3
        and upper_wick >= 2 * body
        and upper_wick >= rng / 2
        and lower_wick <= body
    )
    engulfing = (
        s["prev_close"] > s["prev_open"]      # prior bullish
        and s["close"] < s["open"]            # current bearish
        and s["close"] <= s["prev_open"]
        and s["open"] >= s["prev_close"]
    )
    return shooting_star or engulfing


# ── Per-symbol bookkeeping ──────────────────────────────────────────────

@dataclass
class SymbolContext:
    or_high: float | None = None
    or_low: float | None = None
    state: SymbolState = SymbolState.FLAT
    cooldown_until: datetime | None = None
    last_exit_at: datetime | None = None
    # Breakout-state — set the first time price closes above/below the OR
    # so subsequent entries must take the "retest" path, not "first breakout".
    has_closed_above_or_high: bool = False
    has_closed_below_or_low: bool = False


# ── State machine ──────────────────────────────────────────────────────

DailyLossCapOk = Callable[[], bool]
LocateOk = Callable[[str], bool]   # callable: symbol → (shortable AND easy_to_borrow)


class StateMachine:
    def __init__(
        self,
        cfg: StrategyConfig | None = None,
        *,
        daily_loss_cap_ok: DailyLossCapOk | None = None,
        locate_ok: LocateOk | None = None,
    ) -> None:
        self.cfg = cfg or get_strategy_config()
        self.phase: BotPhase = BotPhase.PRE_MARKET
        self._symbols: dict[str, SymbolContext] = {}
        self.daily_loss_cap_ok = daily_loss_cap_ok or (lambda: True)
        self.locate_ok = locate_ok or (lambda _sym: True)

        # Boundary times parsed once
        e = self.cfg.entry
        self._or_minutes = int(self.cfg.indicators["opening_range_minutes"])
        self._morning_start  = _parse_time(e["morning_start"])    # 09:45
        self._morning_end    = _parse_time(e["morning_end"])      # 11:30
        self._afternoon_start = _parse_time(e["afternoon_start"]) # 13:00
        self._afternoon_end   = _parse_time(e["afternoon_end"])   # 15:00
        self._forced_flat_at  = _parse_time(self.cfg.exits["forced_flat_time"])  # 15:55
        self._wind_down_at    = time(15, 30)                       # spec literal
        self._cooldown        = timedelta(minutes=self.cfg.risk["symbol_cooldown_minutes"])

    # ── Phase transitions ─────────────────────────────────────────

    def update_phase(self, now: datetime) -> BotPhase:
        """Recompute the global phase from current NY time. Idempotent."""
        if now.tzinfo is None:
            raise ValueError("now must be tz-aware")
        ny = now.astimezone(NY_TZ).time()

        if ny >= time(16, 0):
            self.phase = BotPhase.CLOSED
        elif ny >= self._forced_flat_at:
            self.phase = BotPhase.FORCED_FLAT
        elif ny >= self._wind_down_at:
            self.phase = BotPhase.WIND_DOWN
        elif ny >= self._morning_end:
            self.phase = BotPhase.AFTERNOON
        elif ny >= self._morning_start:
            self.phase = BotPhase.HUNTING
        elif ny >= time(9, 30):
            self.phase = BotPhase.BUILDING_OR
        else:
            self.phase = BotPhase.PRE_MARKET
        return self.phase

    # ── Entry-time gate ───────────────────────────────────────────

    def entry_rvol_threshold(self, now: datetime) -> float | None:
        """Return the RVOL_bar minimum required to enter at this time.
        ``None`` means new entries are disallowed at all."""
        ny = now.astimezone(NY_TZ).time()
        if self._morning_start <= ny < self._morning_end:
            return float(self.cfg.entry["rvol_bar_min"])
        if self._afternoon_start <= ny < self._afternoon_end:
            return float(self.cfg.entry["afternoon_rvol_min"])
        return None

    def entries_allowed(self, now: datetime) -> bool:
        return self.entry_rvol_threshold(now) is not None

    # ── Per-symbol API ───────────────────────────────────────────

    def _ctx(self, symbol: str) -> SymbolContext:
        c = self._symbols.get(symbol)
        if c is None:
            c = SymbolContext()
            self._symbols[symbol] = c
        return c

    def record_or(self, symbol: str, high: float, low: float) -> None:
        c = self._ctx(symbol)
        c.or_high, c.or_low = float(high), float(low)

    def get_or(self, symbol: str) -> tuple[float | None, float | None]:
        c = self._ctx(symbol)
        return c.or_high, c.or_low

    def get_symbol_state(self, symbol: str, *, now: datetime | None = None) -> SymbolState:
        c = self._ctx(symbol)
        # Auto-decay cooldown if expired
        if c.state == SymbolState.COOLDOWN and now is not None and c.cooldown_until is not None:
            if now >= c.cooldown_until:
                c.state = SymbolState.FLAT
                c.cooldown_until = None
        return c.state

    def mark_position_opened(self, symbol: str) -> None:
        c = self._ctx(symbol)
        c.state = SymbolState.IN_POSITION

    def mark_position_closed(self, symbol: str, *, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("at must be tz-aware")
        c = self._ctx(symbol)
        c.state = SymbolState.COOLDOWN
        c.last_exit_at = at
        c.cooldown_until = at + self._cooldown

    def is_in_cooldown(self, symbol: str, now: datetime) -> bool:
        return self.get_symbol_state(symbol, now=now) == SymbolState.COOLDOWN

    # ── Evaluation entry point ──────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        bars_1min: pd.DataFrame,
        *,
        now: datetime,
        atr_5min: float | None = None,
        side_hint: Literal["long", "short", "both"] = "both",
    ) -> Decision:
        """Evaluate ``symbol`` on the close of the most recent 1-min bar.

        ``bars_1min`` must contain enough history (≥ ~30 bars) to define
        EMA20, MACD(26,9), RSI(14), volume MA(20), and VWAP-5min-ago.
        ``atr_5min`` is the most recent ATR(14) on 5-min bars — passed in
        so we don't recompute it on every 1-min tick.
        """
        self.update_phase(now)
        gates: dict[str, bool] = {}

        # ── Orchestration gates ─────────────────────────────────
        if not self.entries_allowed(now):
            gates["entries_allowed_by_phase"] = False
            return Decision("skip", "entries_allowed_by_phase", gates)
        gates["entries_allowed_by_phase"] = True

        if self.get_symbol_state(symbol, now=now) != SymbolState.FLAT:
            gates["symbol_flat"] = False
            return Decision("skip", "symbol_flat", gates)
        gates["symbol_flat"] = True

        or_h, or_l = self.get_or(symbol)
        if or_h is None or or_l is None:
            gates["opening_range_known"] = False
            return Decision("skip", "opening_range_known", gates)
        gates["opening_range_known"] = True

        # Gate (10): daily loss cap (delegated to circuit_breakers)
        if not self.daily_loss_cap_ok():
            gates["daily_loss_cap_ok"] = False
            return Decision("skip", "daily_loss_cap_ok", gates)
        gates["daily_loss_cap_ok"] = True

        # ── Indicator snapshot for the bar ──────────────────────
        snap = self._snapshot(symbols=symbol, bars=bars_1min, now=now)
        if snap is None:
            gates["sufficient_history"] = False
            return Decision("skip", "sufficient_history", gates)
        gates["sufficient_history"] = True

        # Update breakout-state for THIS bar (the bar we're evaluating).
        # Important to do this AFTER reading prior state but BEFORE deciding,
        # because a first-breakout entry consumes the "first-time" privilege.
        ctx = self._ctx(symbol)
        was_above_or_high = ctx.has_closed_above_or_high
        was_below_or_low = ctx.has_closed_below_or_low
        if snap["close"] > or_h:
            ctx.has_closed_above_or_high = True
        if snap["close"] < or_l:
            ctx.has_closed_below_or_low = True

        # ── Compute long & short gate stacks ────────────────────
        long_gates = (
            self._check_long_gates(snap, or_h, or_l, was_above_or_high)
            if side_hint in ("long", "both") else None
        )
        short_gates = (
            self._check_short_gates(symbol, snap, or_h, or_l, was_below_or_low)
            if side_hint in ("short", "both") else None
        )

        # Long wins ties (spec doesn't define collision; in practice
        # they're mutually exclusive: price can't be both above AND below VWAP).
        if long_gates is not None and all(long_gates.values()):
            gates.update(long_gates)
            return Decision("long", None, gates, snap)

        if short_gates is not None and all(short_gates.values()):
            gates.update(short_gates)
            return Decision("short", None, gates, snap)

        # Neither side passed — merge BOTH gate dicts so the signal log
        # records the full long+short evaluation. `rvol_bar_ok` exists on
        # both sides with the same boolean, so the overwrite is benign.
        merged: dict[str, bool] = {}
        if long_gates is not None:
            merged.update(long_gates)
        if short_gates is not None:
            merged.update(short_gates)
        gates.update(merged)
        # rejected_gate: first failing gate scanning long-side first, then short.
        rej = next((k for k, v in (long_gates or {}).items() if not v), None)
        if rej is None:
            rej = next((k for k, v in (short_gates or {}).items() if not v), "unknown")
        return Decision("skip", rej, gates, snap)

    # ── Indicator snapshot for the active bar ──────────────────

    def _snapshot(
        self,
        *,
        symbols: str,
        bars: pd.DataFrame,
        now: datetime,
    ) -> dict[str, float] | None:
        """Compute the per-bar indicator snapshot. Returns None if we
        don't yet have enough history for one of the slow indicators."""
        ind = self.cfg.indicators
        slow_period = max(ind["ema_slow"], ind["macd"][1], ind["volume_ma_bars"])
        if len(bars) < slow_period + 5:   # +5 for VWAP_5min_ago lookback
            return None

        close = bars["close"]
        vwap_s = session_vwap(bars)
        if pd.isna(vwap_s.iloc[-1]) or pd.isna(vwap_s.iloc[-6]):
            return None

        ema_fast_s = ema(close, ind["ema_fast"])
        ema_slow_s = ema(close, ind["ema_slow"])
        rsi_s = rsi(close, ind["rsi_period"])
        mfast, mslow, msig = ind["macd"]
        macd_d = macd(close, mfast, mslow, msig)
        rvol_s = rvol_bar(bars["volume"], ind["volume_ma_bars"])

        # Validate slow indicators warm
        for s in (ema_fast_s.iloc[-1], ema_slow_s.iloc[-1], rsi_s.iloc[-1],
                  macd_d["hist"].iloc[-1], rvol_s.iloc[-1]):
            if pd.isna(s):
                return None

        last = bars.iloc[-1]
        prev = bars.iloc[-2] if len(bars) >= 2 else last
        return {
            "ts":         bars.index[-1],
            "open":       float(last["open"]),
            "high":       float(last["high"]),
            "low":        float(last["low"]),
            "close":      float(last["close"]),
            "volume":     float(last["volume"]),
            "prev_open":  float(prev["open"]),
            "prev_high":  float(prev["high"]),
            "prev_low":   float(prev["low"]),
            "prev_close": float(prev["close"]),
            "vwap":       float(vwap_s.iloc[-1]),
            "vwap_5m_ago": float(vwap_s.iloc[-6]),
            "ema_fast":   float(ema_fast_s.iloc[-1]),
            "ema_slow":   float(ema_slow_s.iloc[-1]),
            "rsi":        float(rsi_s.iloc[-1]),
            "macd_hist":  float(macd_d["hist"].iloc[-1]),
            "rvol_bar":   float(rvol_s.iloc[-1]),
            "rvol_required": float(self.entry_rvol_threshold(now) or math.inf),
        }

    # ── Gate stacks ─────────────────────────────────────────────

    def _check_long_gates(
        self,
        s: dict[str, float],
        or_high: float,
        or_low: float,
        was_above_or_high_pre_bar: bool,
    ) -> dict[str, bool]:
        # Spec gate 2: first breakout OR retest with bullish reversal
        first_breakout = (s["close"] > or_high) and not was_above_or_high_pre_bar
        bullish = _is_bullish_reversal(s)
        retest_or = (
            was_above_or_high_pre_bar
            and s["prev_low"] <= or_high <= s["prev_high"] + 1e-9
            and s["close"] > or_high
            and bullish
        )
        retest_vwap = (
            s["prev_low"] <= s["vwap"] <= s["prev_high"] + 1e-9
            and s["close"] > s["vwap"]
            and bullish
        )
        return {
            "trigger_long":         first_breakout or retest_or or retest_vwap,
            "price_above_vwap":     s["close"] > s["vwap"],
            "vwap_slope_positive":  s["vwap"] > s["vwap_5m_ago"],
            "ema_alignment_long":   s["ema_fast"] > s["ema_slow"],
            "rsi_in_band_long":     self.cfg.entry["rsi_long_band"][0] < s["rsi"] < self.cfg.entry["rsi_long_band"][1],
            "macd_hist_positive":   s["macd_hist"] > 0,
            "rvol_bar_ok":          s["rvol_bar"] >= s["rvol_required"],
        }

    def _check_short_gates(
        self,
        symbol: str,
        s: dict[str, float],
        or_high: float,
        or_low: float,
        was_below_or_low_pre_bar: bool,
    ) -> dict[str, bool]:
        first_breakdown = (s["close"] < or_low) and not was_below_or_low_pre_bar
        bearish = _is_bearish_reversal(s)
        retest_or = (
            was_below_or_low_pre_bar
            and s["prev_high"] >= or_low >= s["prev_low"] - 1e-9
            and s["close"] < or_low
            and bearish
        )
        retest_vwap = (
            s["prev_high"] >= s["vwap"] >= s["prev_low"] - 1e-9
            and s["close"] < s["vwap"]
            and bearish
        )
        return {
            "trigger_short":        first_breakdown or retest_or or retest_vwap,
            "price_below_vwap":     s["close"] < s["vwap"],
            "vwap_slope_negative":  s["vwap"] < s["vwap_5m_ago"],
            "ema_alignment_short":  s["ema_fast"] < s["ema_slow"],
            "rsi_in_band_short":    self.cfg.entry["rsi_short_band"][0] < s["rsi"] < self.cfg.entry["rsi_short_band"][1],
            "macd_hist_negative":   s["macd_hist"] < 0,
            "rvol_bar_ok":          s["rvol_bar"] >= s["rvol_required"],
            "locate_ok":            self.locate_ok(symbol),
        }



# ── Helpers ─────────────────────────────────────────────────────────────

def _parse_time(hhmm: str) -> time:
    """Accept '09:45' or '09:45:00'."""
    parts = hhmm.split(":")
    h = int(parts[0]); m = int(parts[1]); s = int(parts[2]) if len(parts) > 2 else 0
    return time(h, m, s)

"""
Three-stage exit manager.

Per-position state machine that, on each closed 1-min bar, emits at most
one ``ExitAction`` per tranche:

  T1  — price hits ±1R → sell ⅓, move stop to breakeven on remainder
  T2  — price hits ±2R **OR** tags VWAP±2σ **OR** tags prior-day H/L
        → sell ⅓, trail remainder under EMA9
  T3  — close breaks EMA9 against position **OR** close reclaims VWAP
        against position **OR** wall-clock ≥ 15:55 ET → close final ⅓

Hard exit overrides (bypass T1/T2/T3 logic):

  • account-level kill switch active
  • 15:55 ET force-flat
  • VWAP reclaim against position **with RVOL_bar ≥ 1.5** (premature reversal
    signal — spec calls this out specifically)

The EM does NOT submit orders.  It returns the action the caller (main
loop) should execute via ``orders.OrderManager``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable, Literal
from zoneinfo import ZoneInfo

from config import StrategyConfig, get_strategy_config

NY_TZ = ZoneInfo("America/New_York")

Side = Literal["long", "short"]
ActionTranche = Literal["T1", "T2", "T3", "HARD"]


# ── State ───────────────────────────────────────────────────────────────

@dataclass
class PositionState:
    symbol: str
    side: Side
    entry_price: float
    initial_qty: int
    qty_remaining: int
    R: float                       # per-share risk (entry − stop magnitude)
    stop_price: float              # current — moves to BE after T1, trails after T2
    t1_qty: int
    t2_qty: int
    t3_qty: int
    t1_done: bool = False
    t2_done: bool = False
    # Optional levels supplied by the scanner / caller:
    prior_day_high: float | None = None
    prior_day_low: float | None = None
    vwap_band_upper_2sigma: float | None = None  # refreshed each bar by EM caller
    vwap_band_lower_2sigma: float | None = None


@dataclass
class ExitAction:
    """A single action the caller should execute on this bar."""
    symbol: str
    tranche: ActionTranche
    side_to_close: Side
    qty: int
    new_stop: float | None
    reason: str
    is_terminal: bool   # True if this action closes the position entirely


# ── Manager ─────────────────────────────────────────────────────────────

class ExitManager:
    def __init__(
        self,
        cfg: StrategyConfig | None = None,
        *,
        kill_switch_active: Callable[[], bool] = lambda: False,
    ) -> None:
        self.cfg = cfg or get_strategy_config()
        self.positions: dict[str, PositionState] = {}
        self.kill_switch_active = kill_switch_active

        # Parse boundary time / thresholds once
        e = self.cfg.exits
        self._t1_R = float(e["t1_R"])
        self._t2_R = float(e["t2_R"])
        self._fractions = tuple(e["scale_fractions"])
        self._forced_flat_time = _parse_time(e["forced_flat_time"])  # 15:55
        self._vwap_reversal_rvol = float(self.cfg.entry["rvol_bar_min"])  # 1.5

    # ── Lifecycle ─────────────────────────────────────────────────

    def open_position(
        self,
        *,
        symbol: str,
        side: Side,
        entry_price: float,
        qty: int,
        stop_price: float,
        prior_day_high: float | None = None,
        prior_day_low: float | None = None,
    ) -> PositionState:
        R = abs(entry_price - stop_price)
        t1_q, t2_q, t3_q = self._split_tranches(qty)
        pos = PositionState(
            symbol=symbol.upper(),
            side=side,
            entry_price=entry_price,
            initial_qty=qty,
            qty_remaining=qty,
            R=R,
            stop_price=stop_price,
            t1_qty=t1_q,
            t2_qty=t2_q,
            t3_qty=t3_q,
            prior_day_high=prior_day_high,
            prior_day_low=prior_day_low,
        )
        self.positions[pos.symbol] = pos
        return pos

    def close_position(self, symbol: str) -> None:
        self.positions.pop(symbol.upper(), None)

    def get(self, symbol: str) -> PositionState | None:
        return self.positions.get(symbol.upper())

    # ── Per-bar evaluation ───────────────────────────────────────

    def evaluate_bar(
        self,
        symbol: str,
        *,
        bar: dict,                       # {open, high, low, close, volume}
        indicators: dict,                # snap from state_machine
        now: datetime,
    ) -> ExitAction | None:
        pos = self.positions.get(symbol.upper())
        if pos is None:
            return None

        # ── Hard overrides — bypass T1/T2/T3 ─────────────────────
        if self.kill_switch_active():
            return self._close_all(pos, reason="kill_switch")

        ny = now.astimezone(NY_TZ).time()
        if ny >= self._forced_flat_time:
            return self._close_all(pos, reason="forced_flat")

        if self._vwap_reverses_against_position(pos, bar, indicators):
            return self._close_all(pos, reason="vwap_reversal_high_rvol")

        # ── Normal T1/T2/T3 progression ─────────────────────────
        if not pos.t1_done and self._t1_hit(pos, bar):
            qty = min(pos.t1_qty, pos.qty_remaining)
            pos.t1_done = True
            pos.qty_remaining -= qty
            pos.stop_price = pos.entry_price  # move stop to breakeven
            terminal = pos.qty_remaining <= 0
            if terminal:
                self.close_position(pos.symbol)
            return ExitAction(
                symbol=pos.symbol, tranche="T1",
                side_to_close=pos.side, qty=qty,
                new_stop=pos.entry_price, reason="t1_target",
                is_terminal=terminal,
            )

        if pos.t1_done and not pos.t2_done and self._t2_hit(pos, bar, indicators):
            qty = min(pos.t2_qty, pos.qty_remaining)
            pos.t2_done = True
            pos.qty_remaining -= qty
            # Trail under EMA9 — exact value supplied by caller via indicators
            new_stop = self._ema9_trail_stop(pos, indicators)
            terminal = pos.qty_remaining <= 0
            if not terminal and new_stop is not None:
                pos.stop_price = new_stop
            if terminal:
                self.close_position(pos.symbol)
            return ExitAction(
                symbol=pos.symbol, tranche="T2",
                side_to_close=pos.side, qty=qty,
                new_stop=new_stop, reason="t2_target",
                is_terminal=terminal,
            )

        if pos.t2_done and self._t3_hit(pos, bar, indicators):
            return self._close_all(pos, reason="t3_trail", tranche="T3")

        return None

    # ── Tranche helpers ──────────────────────────────────────────

    def _split_tranches(self, qty: int) -> tuple[int, int, int]:
        if qty < 3:
            # Too small to split — collapse to a single tranche on T3.
            return 0, 0, qty
        f1, f2, _f3 = self._fractions
        t1 = max(1, math.floor(qty * f1))
        t2 = max(1, math.floor(qty * f2))
        t3 = qty - t1 - t2
        if t3 < 1:
            # Pathological — give the runner at least 1 share by shaving T2.
            t3 = 1
            t2 = max(1, t2 - 1)
            t1 = qty - t2 - t3
        return t1, t2, t3

    # ── Trigger detection ───────────────────────────────────────

    def _t1_hit(self, pos: PositionState, bar: dict) -> bool:
        target = pos.entry_price + (self._t1_R * pos.R if pos.side == "long"
                                    else -self._t1_R * pos.R)
        return bar["high"] >= target if pos.side == "long" else bar["low"] <= target

    def _t2_hit(self, pos: PositionState, bar: dict, ind: dict) -> bool:
        # (a) ±2R
        target_R = pos.entry_price + (self._t2_R * pos.R if pos.side == "long"
                                      else -self._t2_R * pos.R)
        hit_r = bar["high"] >= target_R if pos.side == "long" else bar["low"] <= target_R

        # (b) VWAP±2σ band
        band = (pos.vwap_band_upper_2sigma if pos.side == "long"
                else pos.vwap_band_lower_2sigma)
        hit_band = False
        if band is not None:
            hit_band = bar["high"] >= band if pos.side == "long" else bar["low"] <= band

        # (c) prior-day high / low
        pdh, pdl = pos.prior_day_high, pos.prior_day_low
        hit_prior = False
        if pos.side == "long" and pdh is not None:
            hit_prior = bar["high"] >= pdh
        elif pos.side == "short" and pdl is not None:
            hit_prior = bar["low"] <= pdl

        return hit_r or hit_band or hit_prior

    def _t3_hit(self, pos: PositionState, bar: dict, ind: dict) -> bool:
        ema9 = ind.get("ema_fast")
        vwap = ind.get("vwap")
        # Close breaks EMA9 against position
        if ema9 is not None:
            if pos.side == "long" and bar["close"] < ema9:
                return True
            if pos.side == "short" and bar["close"] > ema9:
                return True
        # Close reclaims VWAP against position
        if vwap is not None:
            if pos.side == "long" and bar["close"] < vwap:
                return True
            if pos.side == "short" and bar["close"] > vwap:
                return True
        return False

    def _vwap_reverses_against_position(
        self, pos: PositionState, bar: dict, ind: dict
    ) -> bool:
        """Spec hard-exit: 'Price closes back through VWAP against position
        on a bar with RVOL_bar ≥ 1.5'."""
        vwap = ind.get("vwap")
        rvol = ind.get("rvol_bar")
        if vwap is None or rvol is None:
            return False
        if rvol < self._vwap_reversal_rvol:
            return False
        if pos.side == "long":
            return bar["close"] < vwap
        return bar["close"] > vwap

    def _ema9_trail_stop(self, pos: PositionState, ind: dict) -> float | None:
        ema9 = ind.get("ema_fast")
        if ema9 is None:
            return None
        buf = float(self.cfg.stops["structural_buffer"])
        return ema9 - buf if pos.side == "long" else ema9 + buf

    def _close_all(
        self,
        pos: PositionState,
        *,
        reason: str,
        tranche: ActionTranche = "HARD",
    ) -> ExitAction:
        qty = pos.qty_remaining
        pos.qty_remaining = 0
        self.close_position(pos.symbol)
        return ExitAction(
            symbol=pos.symbol, tranche=tranche,
            side_to_close=pos.side, qty=qty,
            new_stop=None, reason=reason, is_terminal=True,
        )


def _parse_time(hhmm: str) -> time:
    parts = hhmm.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) > 2 else 0
    return time(h, m, s)

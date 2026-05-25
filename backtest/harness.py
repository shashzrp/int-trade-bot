"""
Vectorized backtest harness.

Replays historical 1-min bars **through the same evaluation code path
that drives the live bot** — ``state_machine.evaluate`` for entries and
``exit_manager.evaluate_bar`` for exits.  This is the single most
important constraint: divergent backtest logic always produces a
strategy that looks great offline and fails in production.

Fill model (spec §backtest):
  • Long entries use the ASK side of the spread (approximated as bar close
    + ``slippage_bps`` since 1-min bars lack quotes).
  • Short entries use the BID side (close − slippage_bps).
  • All stop/T1/T2/T3 fills assume execution at the level + slippage_bps
    in the direction that hurts the trader.
  • Alpaca's stock commission schedule is currently $0 → ignored.

CLI:
    python -m backtest.harness --symbols AAPL,MSFT --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import StrategyConfig, get_strategy_config
from exit_manager import ExitManager
from indicators import (
    atr,
    opening_range,
    resample_to_5min,
    session_vwap,
    session_vwap_bands,
    ema,
    rsi,
    macd,
    rvol_bar,
)
from orders import OrderSizing, StopCalculator
from state_machine import StateMachine, SymbolState

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")


# ── Result types ────────────────────────────────────────────────────────

@dataclass
class FillRecord:
    ts: datetime
    symbol: str
    side: str               # 'long' / 'short'
    tranche: str            # 'entry' / 'T1' / 'T2' / 'T3' / 'HARD' / 'EOD_FLAT'
    qty: int
    price: float
    pnl: float = 0.0        # 0 on entries; signed on exits
    r_per_share: float = 0.0  # |entry − stop|, set on entry fills only


@dataclass
class BacktestResult:
    fills: list[FillRecord] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    starting_equity: float = 0.0
    ending_equity: float = 0.0


# ── Engine ──────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(
        self,
        cfg: StrategyConfig,
        *,
        symbols: list[str],
        starting_equity: float = 100_000.0,
        slippage_bps: float = 1.5,
    ) -> None:
        self.cfg = cfg
        self.symbols = [s.upper() for s in symbols]
        self.starting_equity = float(starting_equity)
        self.slippage_bps = slippage_bps

        self._sizing = OrderSizing(cfg)
        self._stop_calc = StopCalculator(cfg)

    # ── Public ───────────────────────────────────────────────────

    def run(self, bars_by_symbol: dict[str, pd.DataFrame]) -> BacktestResult:
        # Slice into trading-day-sized chunks
        unique_days = sorted({d for df in bars_by_symbol.values() for d in df.index.date})
        equity = self.starting_equity
        all_fills: list[FillRecord] = []
        daily_equity: list[tuple[date, float]] = []

        for day in unique_days:
            day_fills, equity = self._run_day(day, bars_by_symbol, equity)
            all_fills.extend(day_fills)
            daily_equity.append((day, equity))

        eq_series = pd.Series(
            [eq for _, eq in daily_equity],
            index=pd.DatetimeIndex([d for d, _ in daily_equity]),
        )
        return BacktestResult(
            fills=all_fills,
            equity_curve=eq_series,
            starting_equity=self.starting_equity,
            ending_equity=equity,
        )

    # ── Per-day replay ────────────────────────────────────────────

    def _run_day(
        self,
        day: date,
        bars_by_symbol: dict[str, pd.DataFrame],
        equity_start: float,
    ) -> tuple[list[FillRecord], float]:
        # Fresh state machines per day: per-symbol breakout flags / OR reset.
        sm = StateMachine(self.cfg, daily_loss_cap_ok=lambda: True,
                          locate_ok=lambda _s: True)
        em = ExitManager(self.cfg, kill_switch_active=lambda: False)

        day_bars = {
            sym: df[df.index.date == day]
            for sym, df in bars_by_symbol.items()
            if not df.empty
        }
        day_bars = {s: d for s, d in day_bars.items() if not d.empty}

        # Chronological union of all bar timestamps across symbols
        all_ts = sorted({ts for df in day_bars.values() for ts in df.index})
        fills: list[FillRecord] = []
        equity = equity_start

        for ts in all_ts:
            for sym in self.symbols:
                df = day_bars.get(sym)
                if df is None or ts not in df.index:
                    continue
                bar = df.loc[ts]
                df_so_far = df.loc[:ts]

                # ── 1) OR construction during BUILDING_OR ────────
                or_pair = opening_range(
                    df_so_far, session_date=day,
                    minutes=int(self.cfg.indicators["opening_range_minutes"]),
                )
                if or_pair is not None:
                    sm.record_or(sym, *or_pair)

                # ── 2) Manage existing position ──────────────────
                pos = em.get(sym)
                if pos is not None:
                    # Refresh 2σ bands for T2 trigger
                    bands = session_vwap_bands(df_so_far, sigmas=(2.0,))
                    if not pd.isna(bands[2.0]["upper"].iloc[-1]):
                        pos.vwap_band_upper_2sigma = float(bands[2.0]["upper"].iloc[-1])
                    if not pd.isna(bands[2.0]["lower"].iloc[-1]):
                        pos.vwap_band_lower_2sigma = float(bands[2.0]["lower"].iloc[-1])

                    snap = self._snap(df_so_far)
                    action = em.evaluate_bar(sym, bar=bar.to_dict(),
                                              indicators=snap, now=ts)
                    if action is not None:
                        fill_price = self._exit_fill_price(pos, bar, action)
                        pnl = self._pnl(pos.side, action.qty, pos.entry_price, fill_price)
                        fills.append(FillRecord(
                            ts=ts, symbol=sym, side=pos.side,
                            tranche=action.tranche, qty=action.qty,
                            price=fill_price, pnl=pnl,
                        ))
                        equity += pnl
                        if action.is_terminal:
                            sm.mark_position_closed(sym, at=ts)
                        continue   # don't also evaluate entry on this bar

                # ── 3) Evaluate entry ────────────────────────────
                if sm.get_symbol_state(sym, now=ts) != SymbolState.FLAT:
                    continue

                atr5 = self._atr_5min(df_so_far)
                if atr5 is None or atr5 <= 0:
                    continue

                decision = sm.evaluate(sym, df_so_far, now=ts, atr_5min=atr5)
                if not decision.passed:
                    continue

                # Synthesize a fill
                side = decision.action
                close = float(bar["close"])
                entry_fill = self._entry_fill_price(side, close)
                stop = (
                    self._stop_calc.stop_for_long(entry=entry_fill, atr_5min=atr5,
                                                  or_low=sm.get_or(sym)[1])
                    if side == "long"
                    else self._stop_calc.stop_for_short(entry=entry_fill, atr_5min=atr5,
                                                        or_high=sm.get_or(sym)[0])
                )
                size = self._sizing.size(equity=equity, entry=entry_fill, stop=stop)
                if size.is_skip:
                    continue

                em.open_position(
                    symbol=sym, side=side, entry_price=entry_fill,
                    qty=size.shares, stop_price=stop,
                )
                sm.mark_position_opened(sym)
                fills.append(FillRecord(
                    ts=ts, symbol=sym, side=side, tranche="entry",
                    qty=size.shares, price=entry_fill, pnl=0.0,
                    r_per_share=abs(entry_fill - stop),
                ))

        # End-of-day mark: close any still-open positions at the last bar's close.
        for sym in list(em.positions.keys()):
            pos = em.get(sym)
            if pos is None:
                continue
            last_close = day_bars[sym]["close"].iloc[-1]
            fill_price = self._slippage_against(pos.side, float(last_close))
            pnl = self._pnl(pos.side, pos.qty_remaining, pos.entry_price, fill_price)
            fills.append(FillRecord(
                ts=day_bars[sym].index[-1], symbol=sym, side=pos.side,
                tranche="EOD_FLAT", qty=pos.qty_remaining,
                price=fill_price, pnl=pnl,
            ))
            equity += pnl
            em.close_position(sym)

        return fills, equity

    # ── Helpers ───────────────────────────────────────────────────

    def _atr_5min(self, df: pd.DataFrame) -> float | None:
        df5 = resample_to_5min(df)
        if df5.empty:
            return None
        s = atr(df5, period=int(self.cfg.indicators["atr_period"]))
        v = s.iloc[-1]
        return None if pd.isna(v) else float(v)

    @staticmethod
    def _snap(df: pd.DataFrame) -> dict:
        close = df["close"]
        return {
            "vwap":      _last_or_none(session_vwap(df)),
            "ema_fast":  _last_or_none(ema(close, 9)),
            "rsi":       _last_or_none(rsi(close, 14)),
            "macd_hist": _last_or_none(macd(close)["hist"]),
            "rvol_bar":  _last_or_none(rvol_bar(df["volume"], 20)),
        }

    def _entry_fill_price(self, side: str, close: float) -> float:
        bps = self.slippage_bps / 10_000.0
        return close * (1 + bps) if side == "long" else close * (1 - bps)

    def _slippage_against(self, side: str, price: float) -> float:
        """Apply slippage in the direction that hurts the trader."""
        bps = self.slippage_bps / 10_000.0
        return price * (1 - bps) if side == "long" else price * (1 + bps)

    def _exit_fill_price(self, pos, bar, action) -> float:
        # For T1/T2 we'd ideally fill at the target price; for T3/HARD at the
        # bar's close. Use close as a safe approximation with adverse slippage.
        return self._slippage_against(pos.side, float(bar["close"]))

    @staticmethod
    def _pnl(side: str, qty: int, entry: float, exit_price: float) -> float:
        return (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty


def _last_or_none(s: pd.Series) -> float | None:
    if len(s) == 0:
        return None
    v = s.iloc[-1]
    return None if pd.isna(v) else float(v)


# ── CLI ─────────────────────────────────────────────────────────────────

def _main() -> int:   # pragma: no cover — manual / CI run
    parser = argparse.ArgumentParser(description="Run the strategy backtest.")
    parser.add_argument("--symbols", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="backtest/_cache")
    parser.add_argument("--report-out", type=str, default="backtest/reports")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from clients import get_data_client
    from backtest.data_loader import load_1min_bars
    from backtest.report import build_report

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    bars = load_1min_bars(get_data_client(), symbols, start=start, end=end,
                          cache_dir=Path(args.cache_dir))
    if not bars:
        logger.error("no data — aborting")
        return 1

    cfg = get_strategy_config()
    eng = BacktestEngine(cfg, symbols=symbols)
    result = eng.run(bars)

    report = build_report(result, out_dir=Path(args.report_out))
    print(report.text_summary())

    return 0 if report.passes_viability_gates() else 2


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(_main())

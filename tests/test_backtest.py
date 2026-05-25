"""
End-to-end smoke test for the backtest harness.

We don't fetch real Alpaca data here — that's the manual `python -m
backtest.harness` call. The test builds synthetic 1-min bars across two
days, runs the harness, and verifies:

  • the harness completes without error
  • the report's text summary is produced
  • a fills.csv and equity_curve.png are written to the report dir
  • the metric struct is well-formed
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from backtest.harness import BacktestEngine, BacktestResult, FillRecord
from backtest.report import build_report, compute_metrics
from config import get_strategy_config


NY = ZoneInfo("America/New_York")


def _build_session_bars(day: date, n_bars: int = 90) -> pd.DataFrame:
    """Build a 90-minute session bar series for one day."""
    start = datetime.combine(day, datetime.min.time().replace(hour=9, minute=30), tzinfo=NY)
    idx = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(n_bars)])
    rng = np.random.default_rng(int(day.toordinal()))
    closes = 100 + np.cumsum(rng.normal(0, 0.05, n_bars))
    return pd.DataFrame(
        {"open": closes, "high": closes + 0.05, "low": closes - 0.05,
         "close": closes, "volume": np.r_[np.full(15, 1500), np.full(n_bars - 15, 3000)]},
        index=idx,
    )


def test_engine_runs_clean_on_synthetic_data():
    cfg = get_strategy_config()
    bars = {
        "AAPL": pd.concat([
            _build_session_bars(date(2024, 1, 16), 90),
            _build_session_bars(date(2024, 1, 17), 90),
        ]),
    }
    eng = BacktestEngine(cfg, symbols=["AAPL"], starting_equity=100_000.0)
    result = eng.run(bars)

    assert result.starting_equity == 100_000.0
    assert isinstance(result.equity_curve, pd.Series)
    assert len(result.equity_curve) == 2   # two trading days


def test_report_produces_files(tmp_path):
    cfg = get_strategy_config()
    bars = {"AAPL": _build_session_bars(date(2024, 1, 16), 90)}
    result = BacktestEngine(cfg, symbols=["AAPL"]).run(bars)

    metrics = build_report(result, out_dir=tmp_path)
    assert (tmp_path / "equity_curve.png").exists()
    assert (tmp_path / "fills.csv").exists()
    # Text summary contains all the required headings
    summary = metrics.text_summary()
    for key in ("Win rate", "Avg win/avg loss", "Profit factor",
                "Max drawdown", "Sharpe"):
        assert key in summary


def test_metrics_zeros_when_no_trades():
    cfg = get_strategy_config()
    # An empty bar set produces no trades
    result = BacktestEngine(cfg, symbols=["AAPL"]).run({"AAPL": pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz=NY),
    )})
    m = compute_metrics(result)
    assert m.n_trades == 0
    assert m.win_rate == 0
    assert m.passes_viability_gates() is False  # zero-trade strategy fails gates


def test_viability_gate_logic():
    """compute_metrics → passes_viability_gates: all 5 thresholds must clear."""
    from backtest.report import ReportMetrics
    great = ReportMetrics(
        n_trades=100, n_wins=50, n_losses=50,
        win_rate=0.50, avg_win=2.0, avg_loss=1.0,
        win_loss_ratio=2.0, profit_factor=1.5,
        sharpe=1.3, max_drawdown_pct=0.10,
        total_return_pct=20.0, trades_per_day=2.0,
    )
    assert great.passes_viability_gates() is True

    bad_winrate = ReportMetrics(**{**great.__dict__, "win_rate": 0.40})
    assert bad_winrate.passes_viability_gates() is False

    bad_pf = ReportMetrics(**{**great.__dict__, "profit_factor": 1.0})
    assert bad_pf.passes_viability_gates() is False

    bad_dd = ReportMetrics(**{**great.__dict__, "max_drawdown_pct": 0.20})
    assert bad_dd.passes_viability_gates() is False

    bad_sharpe = ReportMetrics(**{**great.__dict__, "sharpe": 0.9})
    assert bad_sharpe.passes_viability_gates() is False

    bad_wl = ReportMetrics(**{**great.__dict__, "win_loss_ratio": 1.5})
    assert bad_wl.passes_viability_gates() is False


# ── Tranche breakdown ───────────────────────────────────────────────────

def _ts(i: int) -> datetime:
    return datetime(2024, 1, 16, 9, 30, tzinfo=NY) + timedelta(minutes=i)


def _entry(i: int, qty: int = 99, price: float = 100.0, r: float = 1.0) -> FillRecord:
    return FillRecord(ts=_ts(i), symbol="AAPL", side="long", tranche="entry",
                      qty=qty, price=price, pnl=0.0, r_per_share=r)


def _exit(i: int, tranche: str, qty: int, pnl: float) -> FillRecord:
    # price is irrelevant for breakdown logic — only tranche/qty/pnl matter.
    return FillRecord(ts=_ts(i), symbol="AAPL", side="long", tranche=tranche,
                      qty=qty, price=100.0, pnl=pnl)


def _result(fills: list[FillRecord]) -> BacktestResult:
    return BacktestResult(fills=fills, equity_curve=pd.Series(dtype=float),
                          starting_equity=100_000.0, ending_equity=100_000.0)


def test_tranche_breakdown_classifies_all_four_buckets():
    """One trade in each bucket. Verify counts and avg-R per bucket."""
    # qty=99 split → t1=33, t2=33, t3=33; entry r_per_share=1.0  →  R per trade = qty*R = 99
    # Trade A: stopped before T1 — HARD with full qty at −1R per share = −99 PnL → −1.0R total
    trade_a = [_entry(0), _exit(1, "HARD", 99, -99.0)]

    # Trade B: T1 fires (+33 PnL at +1R per share), runner exits at HARD around BE (≈0 PnL)
    # Total = +33 → R-multiple = 33 / 99 = +0.333
    trade_b = [_entry(10), _exit(11, "T1", 33, 33.0), _exit(12, "HARD", 66, 0.0)]

    # Trade C: T1 (+33) + T2 (+66 at +2R per share) + runner HARD near BE (0)
    # Total = +99 → R-multiple = +1.0
    trade_c = [_entry(20), _exit(21, "T1", 33, 33.0),
               _exit(22, "T2", 33, 66.0), _exit(23, "EOD_FLAT", 33, 0.0)]

    # Trade D: T1 (+33) + T2 (+66) + T3 (+99 at +3R per share)
    # Total = +198 → R-multiple = +2.0
    trade_d = [_entry(30), _exit(31, "T1", 33, 33.0),
               _exit(32, "T2", 33, 66.0), _exit(33, "T3", 33, 99.0)]

    m = compute_metrics(_result(trade_a + trade_b + trade_c + trade_d))
    tb = m.tranche_breakdown

    assert (tb.n_stopped, tb.n_t1, tb.n_t2, tb.n_runner) == (1, 1, 1, 1)
    assert tb.n_total == 4
    assert tb.avg_R_stopped == pytest.approx(-1.0)
    assert tb.avg_R_t1 == pytest.approx(33 / 99)
    assert tb.avg_R_t2 == pytest.approx(1.0)
    assert tb.avg_R_runner == pytest.approx(2.0)


def test_tranche_breakdown_averages_within_bucket():
    """Two trades land in the runner bucket with different R-multiples."""
    # Trade 1: full runner +198 PnL → +2.0R
    t1 = [_entry(0), _exit(1, "T1", 33, 33.0),
          _exit(2, "T2", 33, 66.0), _exit(3, "T3", 33, 99.0)]
    # Trade 2: runner gives back some — total +99 PnL → +1.0R
    t2 = [_entry(10), _exit(11, "T1", 33, 33.0),
          _exit(12, "T2", 33, 66.0), _exit(13, "T3", 33, 0.0)]

    m = compute_metrics(_result(t1 + t2))
    assert m.tranche_breakdown.n_runner == 2
    assert m.tranche_breakdown.avg_R_runner == pytest.approx(1.5)


def test_tranche_breakdown_empty_when_no_fills():
    m = compute_metrics(_result([]))
    tb = m.tranche_breakdown
    assert (tb.n_stopped, tb.n_t1, tb.n_t2, tb.n_runner) == (0, 0, 0, 0)
    assert tb.avg_R_stopped == 0.0


def test_text_summary_includes_tranche_breakdown():
    fills = [_entry(0), _exit(1, "HARD", 99, -99.0)]
    m = compute_metrics(_result(fills))
    text = m.text_summary()
    for heading in ("Tranche breakdown",
                    "Stopped (no T1)", "Exited at T1",
                    "Exited at T2", "Runner (T3)",
                    "avg R ="):
        assert heading in text


def test_harness_entry_fills_carry_r_per_share():
    """End-to-end: the engine writes a non-zero r_per_share onto entry fills."""
    cfg = get_strategy_config()
    bars = {"AAPL": _build_session_bars(date(2024, 1, 16), 90)}
    result = BacktestEngine(cfg, symbols=["AAPL"]).run(bars)
    entries = [f for f in result.fills if f.tranche == "entry"]
    # If the synthetic data produces any entries, they must carry positive R.
    for e in entries:
        assert e.r_per_share > 0, f"entry fill missing r_per_share: {e}"

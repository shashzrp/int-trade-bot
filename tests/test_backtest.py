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

from backtest.harness import BacktestEngine
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

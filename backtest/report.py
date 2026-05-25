"""
Backtest report: stats + equity-curve PNG + viability-gate check.

Viability gates (spec §backtest target metrics):
  win rate ≥ 45%
  avg win / avg loss ≥ 1.8
  profit factor ≥ 1.4
  max drawdown ≤ 15%
  Sharpe (annualized) ≥ 1.2
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.harness import BacktestResult, FillRecord

logger = logging.getLogger(__name__)

# Annualization factor for daily-return Sharpe (US trading days)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class ReportMetrics:
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float          # fraction in [0, 1]
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    profit_factor: float
    sharpe: float
    max_drawdown_pct: float  # fraction in [0, 1]
    total_return_pct: float
    trades_per_day: float

    def text_summary(self) -> str:
        return (
            f"Backtest Report\n"
            f"  Trades:           {self.n_trades} ({self.trades_per_day:.2f} / day)\n"
            f"  Wins / Losses:    {self.n_wins} / {self.n_losses}\n"
            f"  Win rate:         {self.win_rate * 100:.2f}%\n"
            f"  Avg win/avg loss: {self.win_loss_ratio:.2f}\n"
            f"  Profit factor:    {self.profit_factor:.2f}\n"
            f"  Max drawdown:     {self.max_drawdown_pct * 100:.2f}%\n"
            f"  Sharpe (annual):  {self.sharpe:.2f}\n"
            f"  Total return:     {self.total_return_pct:.2f}%\n"
        )

    def passes_viability_gates(self) -> bool:
        return (
            self.win_rate >= 0.45
            and self.win_loss_ratio >= 1.8
            and self.profit_factor >= 1.4
            and self.max_drawdown_pct <= 0.15
            and self.sharpe >= 1.2
        )


def compute_metrics(result: BacktestResult) -> ReportMetrics:
    """Aggregate fills into round-trip P&L per (entry → close) pairing.

    Strategy: a trade is the SET of fills sharing the same symbol between an
    'entry' fill and the next terminal-close (T3 / EOD_FLAT).  Simpler proxy:
    sum the per-fill `pnl` field (which is 0 on entries) — gives the same
    realized total.  But for win/loss counts we group by (symbol, position id).
    """
    fills = result.fills
    if not fills:
        return ReportMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Group into per-position P&L
    trade_pnls: list[float] = []
    current_pnl = 0.0
    current_active = False
    for fr in fills:
        if fr.tranche == "entry":
            if current_active and abs(current_pnl) > 0:
                trade_pnls.append(current_pnl)
            current_pnl = 0.0
            current_active = True
        else:
            current_pnl += fr.pnl
    if current_active and abs(current_pnl) > 0:
        trade_pnls.append(current_pnl)

    arr = np.array(trade_pnls, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    n_trades = len(arr)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n_trades) if n_trades else 0.0
    avg_win = float(wins.mean()) if n_wins else 0.0
    avg_loss = float(-losses.mean()) if n_losses else 0.0  # positive
    wl_ratio = (avg_win / avg_loss) if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0
    pf = (wins.sum() / -losses.sum()) if losses.size and losses.sum() != 0 else float("inf") if wins.sum() > 0 else 0.0

    # Equity-curve based stats
    eq = result.equity_curve
    if eq.empty:
        sharpe = 0.0
        max_dd = 0.0
    else:
        daily_ret = eq.pct_change().dropna()
        sharpe = (
            (daily_ret.mean() / daily_ret.std()) * math.sqrt(TRADING_DAYS_PER_YEAR)
            if daily_ret.std() > 0 else 0.0
        )
        rolling_max = eq.cummax()
        dd = (eq - rolling_max) / rolling_max
        max_dd = float(-dd.min()) if not dd.empty else 0.0

    n_days = max(1, len(eq))
    trades_per_day = n_trades / n_days
    total_ret_pct = (
        (result.ending_equity - result.starting_equity) / result.starting_equity * 100
        if result.starting_equity > 0 else 0.0
    )

    return ReportMetrics(
        n_trades=n_trades, n_wins=n_wins, n_losses=n_losses,
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        win_loss_ratio=wl_ratio, profit_factor=pf,
        sharpe=float(sharpe), max_drawdown_pct=max_dd,
        total_return_pct=total_ret_pct, trades_per_day=trades_per_day,
    )


def build_report(result: BacktestResult, *, out_dir: Path) -> ReportMetrics:
    metrics = compute_metrics(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_equity_curve_png(result, out_dir / "equity_curve.png")
    _save_fills_csv(result, out_dir / "fills.csv")
    return metrics


def _save_equity_curve_png(result: BacktestResult, path: Path) -> None:
    # Local import so matplotlib is optional for non-backtest paths.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    if not result.equity_curve.empty:
        ax.plot(result.equity_curve.index, result.equity_curve.values, label="Equity")
        ax.axhline(result.starting_equity, color="grey", linestyle=":", label="Start")
    ax.set_title("Backtest Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_fills_csv(result: BacktestResult, path: Path) -> None:
    rows = [
        {"ts": fr.ts, "symbol": fr.symbol, "side": fr.side, "tranche": fr.tranche,
         "qty": fr.qty, "price": fr.price, "pnl": fr.pnl}
        for fr in result.fills
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

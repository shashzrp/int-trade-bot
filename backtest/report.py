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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.harness import BacktestResult, FillRecord

logger = logging.getLogger(__name__)

# Annualization factor for daily-return Sharpe (US trading days)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class TrancheBreakdown:
    """How trades resolved across the three-stage exit ladder.

    Buckets are mutually exclusive and are defined by the *furthest* tranche
    each trade reached:

      stopped  — trade closed (HARD / EOD_FLAT) without T1 ever firing
      t1       — T1 fired; trade closed before T2
      t2       — T1+T2 fired; trade closed before T3 (runner force-flatted)
      runner   — T3 fired (runner trail exit)

    ``avg_R_*`` is the mean of total-trade R-multiples for that bucket, where
    R-multiple = total_trade_pnl / (initial_qty × r_per_share).
    """
    n_stopped: int
    n_t1: int
    n_t2: int
    n_runner: int
    avg_R_stopped: float
    avg_R_t1: float
    avg_R_t2: float
    avg_R_runner: float

    @property
    def n_total(self) -> int:
        return self.n_stopped + self.n_t1 + self.n_t2 + self.n_runner

    def text_summary(self) -> str:
        total = max(1, self.n_total)
        def pct(n: int) -> str:
            return f"{n / total * 100:5.1f}%"
        return (
            f"  Tranche breakdown:\n"
            f"    Stopped (no T1):  {self.n_stopped:4d}  ({pct(self.n_stopped)})  "
            f"avg R = {self.avg_R_stopped:+.2f}\n"
            f"    Exited at T1:     {self.n_t1:4d}  ({pct(self.n_t1)})  "
            f"avg R = {self.avg_R_t1:+.2f}\n"
            f"    Exited at T2:     {self.n_t2:4d}  ({pct(self.n_t2)})  "
            f"avg R = {self.avg_R_t2:+.2f}\n"
            f"    Runner (T3):      {self.n_runner:4d}  ({pct(self.n_runner)})  "
            f"avg R = {self.avg_R_runner:+.2f}\n"
        )


def _empty_breakdown() -> TrancheBreakdown:
    return TrancheBreakdown(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)


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
    tranche_breakdown: TrancheBreakdown = field(default_factory=_empty_breakdown)

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
            f"{self.tranche_breakdown.text_summary()}"
        )

    def passes_viability_gates(self) -> bool:
        return (
            self.win_rate >= 0.45
            and self.win_loss_ratio >= 1.8
            and self.profit_factor >= 1.4
            and self.max_drawdown_pct <= 0.15
            and self.sharpe >= 1.2
        )


@dataclass
class _Trade:
    """One entry → terminal-close grouping, used internally for metrics."""
    entry_qty: int
    r_per_share: float
    total_pnl: float
    tranches_hit: set[str]   # subset of {'T1','T2','T3','HARD','EOD_FLAT'}


def _group_trades(fills: list[FillRecord]) -> list[_Trade]:
    """Walk fills in order; each 'entry' starts a new trade and exit fills
    accumulate onto it until the next entry."""
    trades: list[_Trade] = []
    cur: _Trade | None = None
    for fr in fills:
        if fr.tranche == "entry":
            if cur is not None:
                trades.append(cur)
            cur = _Trade(
                entry_qty=fr.qty,
                r_per_share=fr.r_per_share,
                total_pnl=0.0,
                tranches_hit=set(),
            )
        else:
            if cur is None:
                continue   # defensive: exit before any entry — skip
            cur.total_pnl += fr.pnl
            cur.tranches_hit.add(fr.tranche)
    if cur is not None:
        trades.append(cur)
    return trades


def _bucket_for(trade: _Trade) -> str:
    """Furthest tranche reached, regardless of how the trade terminally closed."""
    if "T3" in trade.tranches_hit:
        return "runner"
    if "T2" in trade.tranches_hit:
        return "t2"
    if "T1" in trade.tranches_hit:
        return "t1"
    return "stopped"


def _compute_tranche_breakdown(trades: list[_Trade]) -> TrancheBreakdown:
    buckets: dict[str, list[float]] = {"stopped": [], "t1": [], "t2": [], "runner": []}
    for t in trades:
        denom = t.entry_qty * t.r_per_share
        # Zero-risk trades shouldn't happen (entry≠stop) but guard anyway:
        r_mult = (t.total_pnl / denom) if denom > 0 else 0.0
        buckets[_bucket_for(t)].append(r_mult)

    def avg(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return TrancheBreakdown(
        n_stopped=len(buckets["stopped"]),
        n_t1=len(buckets["t1"]),
        n_t2=len(buckets["t2"]),
        n_runner=len(buckets["runner"]),
        avg_R_stopped=avg(buckets["stopped"]),
        avg_R_t1=avg(buckets["t1"]),
        avg_R_t2=avg(buckets["t2"]),
        avg_R_runner=avg(buckets["runner"]),
    )


def compute_metrics(result: BacktestResult) -> ReportMetrics:
    """Aggregate fills into round-trip P&L per (entry → close) pairing.

    A trade = one 'entry' fill plus every exit fill until the next entry.
    Win/loss counts use $-PnL; the tranche breakdown classifies each trade by
    the furthest stage of the exit ladder it reached.
    """
    fills = result.fills
    if not fills:
        return ReportMetrics(
            n_trades=0, n_wins=0, n_losses=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            win_loss_ratio=0.0, profit_factor=0.0,
            sharpe=0.0, max_drawdown_pct=0.0,
            total_return_pct=0.0, trades_per_day=0.0,
            tranche_breakdown=_empty_breakdown(),
        )

    trades = _group_trades(fills)
    # Preserve prior behaviour: win/loss stats ignore exact-zero-PnL trades.
    trade_pnls = [t.total_pnl for t in trades if abs(t.total_pnl) > 0]
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
        tranche_breakdown=_compute_tranche_breakdown(trades),
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
         "qty": fr.qty, "price": fr.price, "pnl": fr.pnl,
         "r_per_share": fr.r_per_share}
        for fr in result.fills
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

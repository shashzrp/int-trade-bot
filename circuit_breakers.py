"""
Account-level risk gates.

Aggregates the per-day / per-week loss caps and concurrency limits into a
single ``can_take_new_entry()`` master gate that the state machine consults
before placing any order.  Also exposes ``flatten_all()`` for the panic
path used by the daily-loss-cap test and the 15:55 wind-down.

All thresholds come from ``cfg.risk``:

  per_trade_pct            (1.0)   — sizing only; not used here
  daily_loss_cap_pct       (3.0)
  weekly_loss_cap_pct      (6.0)
  max_consecutive_losses   (3)
  max_concurrent_positions (3)
  max_trades_per_day       (5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient

from config import StrategyConfig, get_strategy_config
from persistence import TradeStore

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")


@dataclass
class GateResult:
    allowed: bool
    reason: str | None = None


class CircuitBreakers:
    def __init__(
        self,
        cfg: StrategyConfig | None,
        store: TradeStore,
        trading_client: TradingClient,
    ) -> None:
        self.cfg = (cfg or get_strategy_config()).risk
        self.store = store
        self.tc = trading_client
        # Captured at the start of each trading day so the daily-loss-cap
        # is measured against a fixed reference, not a moving baseline.
        self._starting_equity_today: Decimal | None = None
        self._starting_equity_week: Decimal | None = None
        # Set when the weekly cap trips — manual reset required.
        self._halted_until: date | None = None

    # ── Day / week initialization ─────────────────────────────────

    def set_starting_equity_today(self, equity: float | Decimal) -> None:
        self._starting_equity_today = Decimal(str(equity))
        logger.info("circuit_breakers: starting_equity_today=%s", self._starting_equity_today)

    def set_starting_equity_week(self, equity: float | Decimal) -> None:
        self._starting_equity_week = Decimal(str(equity))

    def halt_for_next_trading_day(self, today: date) -> None:
        """Mark the bot halted until the calendar day AFTER ``today``."""
        self._halted_until = today + timedelta(days=1)

    def manually_reset_weekly_halt(self) -> None:
        self._halted_until = None

    # ── Pure threshold checks ────────────────────────────────────

    def daily_pnl_pct(self, current_equity: float | Decimal) -> Decimal | None:
        if self._starting_equity_today is None or self._starting_equity_today == 0:
            return None
        eq = Decimal(str(current_equity))
        return (eq - self._starting_equity_today) / self._starting_equity_today * Decimal("100")

    def weekly_pnl_pct(self, current_equity: float | Decimal) -> Decimal | None:
        if self._starting_equity_week is None or self._starting_equity_week == 0:
            return None
        eq = Decimal(str(current_equity))
        return (eq - self._starting_equity_week) / self._starting_equity_week * Decimal("100")

    def daily_loss_cap_breached(self, current_equity: float | Decimal) -> bool:
        pnl_pct = self.daily_pnl_pct(current_equity)
        if pnl_pct is None:
            return False
        return pnl_pct <= Decimal(str(-self.cfg["daily_loss_cap_pct"]))

    def weekly_loss_cap_breached(self, current_equity: float | Decimal) -> bool:
        pnl_pct = self.weekly_pnl_pct(current_equity)
        if pnl_pct is None:
            return False
        return pnl_pct <= Decimal(str(-self.cfg["weekly_loss_cap_pct"]))

    def consecutive_losses_breached(self) -> bool:
        n = self.store.get_consecutive_losses()
        return n >= int(self.cfg["max_consecutive_losses"])

    def trades_today_at_max(self, today: date) -> bool:
        return self.store.get_trades_today_count(today) >= int(self.cfg["max_trades_per_day"])

    def concurrent_positions_at_max(self) -> bool:
        return len(self.store.get_open_positions()) >= int(self.cfg["max_concurrent_positions"])

    # ── Master gate ──────────────────────────────────────────────

    def can_take_new_entry(self, *, today: date, current_equity: float | Decimal) -> GateResult:
        """The single decision the state machine should consult before
        placing an order.  Returns ``(allowed, reason)``."""
        if self._halted_until is not None and today < self._halted_until:
            return GateResult(False, "weekly_halt_active")
        if self.daily_loss_cap_breached(current_equity):
            return GateResult(False, "daily_loss_cap")
        if self.weekly_loss_cap_breached(current_equity):
            return GateResult(False, "weekly_loss_cap")
        if self.consecutive_losses_breached():
            return GateResult(False, "consecutive_losses")
        if self.trades_today_at_max(today):
            return GateResult(False, "trades_per_day")
        if self.concurrent_positions_at_max():
            return GateResult(False, "max_concurrent_positions")
        return GateResult(True, None)

    # ── Panic flatten ─────────────────────────────────────────────

    def flatten_all(self, *, reason: str = "circuit_breaker") -> None:
        """Cancel all open orders, then close all positions at market.

        Two-step: orders first (so close-position market orders aren't
        rejected by hanging limit orders on the same symbol)."""
        logger.warning("flatten_all reason=%s", reason)
        try:
            self.tc.cancel_orders()
        except Exception as exc:    # alpaca-py raises SDK-specific exceptions
            logger.error("cancel_orders failed during flatten reason=%s err=%s", reason, exc)
        try:
            self.tc.close_all_positions(cancel_orders=True)
        except Exception as exc:
            logger.error("close_all_positions failed during flatten reason=%s err=%s", reason, exc)

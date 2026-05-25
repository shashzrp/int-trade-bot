"""
Order placement: position sizing, stop computation, locate verification,
and bracket submission with chase logic.

Four layers, each independently testable:

  OrderSizing      — pure math (per-trade risk %, notional cap, skip-too-loose).
  StopCalculator   — tighter of ATR(1.5×) and structural (OR ± $0.05).
  LocateCache      — per-day cache of (shortable, easy_to_borrow).  Required
                     for SHORT entries per the spec; saves API quota.
  OrderManager     — orchestrates submission, chase / cancel-and-replace,
                     bracket OCO with stop+TP at T2.  Every order gets a
                     unique `client_order_id` for idempotency.

The spec's "place T1 limit at +1R as a SEPARATE order after fill" lives in
``exit_manager.py`` (Step 8) — keeping concerns clean.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Literal
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest

from config import StrategyConfig, get_strategy_config
from persistence import TradeStore, make_client_order_id

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")


# ── Exceptions ──────────────────────────────────────────────────────────

class PdtError(RuntimeError):
    """Pattern-Day-Trader rule violation at startup."""


# ── Sizing ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SizingResult:
    shares: int
    risk_dollars: float        # what we actually risk
    notional: float            # qty × entry
    binding_cap: str           # 'risk' / 'notional' / 'too_loose' / 'fractional_share'

    @property
    def is_skip(self) -> bool:
        return self.shares <= 0


class OrderSizing:
    """All numbers in dollars / shares.  Stateless except for the cfg."""

    def __init__(self, cfg: StrategyConfig | None = None) -> None:
        self.cfg = (cfg or get_strategy_config())

    def size(self, *, equity: float, entry: float, stop: float) -> SizingResult:
        if entry <= 0 or stop <= 0:
            return SizingResult(0, 0.0, 0.0, "invalid_input")

        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return SizingResult(0, 0.0, 0.0, "invalid_input")

        # Gate: skip if the stop is too loose (would risk > N% of price per share).
        max_per_share_pct = float(self.cfg.stops["max_risk_per_share_pct"])
        if risk_per_share > entry * max_per_share_pct / 100.0:
            return SizingResult(0, 0.0, 0.0, "too_loose")

        # 1% of equity at risk.
        risk_dollars_budget = equity * float(self.cfg.risk["per_trade_pct"]) / 100.0
        shares_from_risk = math.floor(risk_dollars_budget / risk_per_share)

        # 20% notional cap.
        notional_cap = equity * float(self.cfg.risk["max_notional_pct"]) / 100.0
        shares_from_notional = math.floor(notional_cap / entry)

        if shares_from_risk <= 0 or shares_from_notional <= 0:
            return SizingResult(0, 0.0, 0.0, "fractional_share")

        if shares_from_notional < shares_from_risk:
            shares = shares_from_notional
            binding = "notional"
        else:
            shares = shares_from_risk
            binding = "risk"

        return SizingResult(
            shares=int(shares),
            risk_dollars=shares * risk_per_share,
            notional=shares * entry,
            binding_cap=binding,
        )


# ── Stop calculator ─────────────────────────────────────────────────────

class StopCalculator:
    def __init__(self, cfg: StrategyConfig | None = None) -> None:
        self.cfg = cfg or get_strategy_config()

    def stop_for_long(self, *, entry: float, atr_5min: float, or_low: float) -> float:
        atr_stop = entry - float(self.cfg.stops["atr_multiplier"]) * atr_5min
        struct_stop = or_low - float(self.cfg.stops["structural_buffer"])
        # Tighter = closer to entry = higher for longs.
        return max(atr_stop, struct_stop)

    def stop_for_short(self, *, entry: float, atr_5min: float, or_high: float) -> float:
        atr_stop = entry + float(self.cfg.stops["atr_multiplier"]) * atr_5min
        struct_stop = or_high + float(self.cfg.stops["structural_buffer"])
        # Tighter = closer to entry = lower for shorts.
        return min(atr_stop, struct_stop)


# ── Locate cache ────────────────────────────────────────────────────────

@dataclass
class LocateInfo:
    asof: date
    shortable: bool
    easy_to_borrow: bool


class LocateCache:
    """Caches per-day locate results so we don't hammer `get_asset` once
    per signal evaluation on a SHORT candidate.

    Required for SHORT entries — the spec explicitly mandates BOTH
    `shortable` AND `easy_to_borrow` to be True via Alpaca's asset endpoint.
    """

    def __init__(self, trading_client: TradingClient) -> None:
        self.tc = trading_client
        self._cache: dict[str, LocateInfo] = {}

    def lookup(self, symbol: str, *, today: date | None = None) -> LocateInfo:
        sym = symbol.upper()
        today = today or datetime.now(tz=NY_TZ).date()
        info = self._cache.get(sym)
        if info is not None and info.asof == today:
            return info
        asset = self.tc.get_asset(sym)
        info = LocateInfo(
            asof=today,
            shortable=bool(getattr(asset, "shortable", False)),
            easy_to_borrow=bool(getattr(asset, "easy_to_borrow", False)),
        )
        self._cache[sym] = info
        return info

    def is_locate_ok(self, symbol: str, *, today: date | None = None) -> bool:
        info = self.lookup(symbol, today=today)
        return info.shortable and info.easy_to_borrow


# ── Order plan ──────────────────────────────────────────────────────────

@dataclass
class EntryPlan:
    """Fully-specified entry, sized + priced + with bracket levels."""
    symbol: str
    side: Literal["long", "short"]
    qty: int
    entry_limit: float       # mid ± offset
    stop_loss: float
    take_profit: float       # T2 price (entry ± 2R)
    R: float                 # per-share risk
    initial_risk_dollars: float


@dataclass
class SubmissionResult:
    accepted: bool
    client_order_id: str | None
    order_id: str | None
    final_status: str
    attempts: int
    reason: str | None = None


# ── Order manager ───────────────────────────────────────────────────────

class OrderManager:
    """Bracket submission with cancel-and-replace chase."""

    def __init__(
        self,
        trading_client: TradingClient,
        store: TradeStore,
        cfg: StrategyConfig | None = None,
        *,
        locate_cache: LocateCache | None = None,
        sizing: OrderSizing | None = None,
        stops: StopCalculator | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc),
    ) -> None:
        self.tc = trading_client
        self.store = store
        self.cfg = cfg or get_strategy_config()
        self.locate = locate_cache or LocateCache(trading_client)
        self.sizing = sizing or OrderSizing(self.cfg)
        self.stops = stops or StopCalculator(self.cfg)
        self._sleep = sleep_fn
        self._clock = clock_fn

    # ── PDT gate ──────────────────────────────────────────────────

    def assert_pdt_ok(self) -> None:
        """Refuse to start under the FINRA pattern-day-trader rule:
        accounts under $25k equity require ``pattern_day_trader=True``
        flagged by the broker before they can day-trade. We bail loud."""
        acct = self.tc.get_account()
        equity = float(acct.equity)
        is_pdt = bool(getattr(acct, "pattern_day_trader", False))
        if equity < 25_000 and not is_pdt:
            raise PdtError(
                f"Account equity ${equity:,.2f} is under the $25,000 FINRA "
                "Pattern-Day-Trader minimum and broker has NOT flagged this "
                "account as PDT. Day-trading is disallowed. Refusing to start."
            )

    # ── Plan builder ──────────────────────────────────────────────

    def plan(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        mid_price: float,
        atr_5min: float,
        or_high: float,
        or_low: float,
        equity: float,
    ) -> EntryPlan | None:
        """Compute the full bracket. Returns None if the setup is too loose
        or would size to zero shares."""
        offset = float(self.cfg.orders["limit_offset_cents"]) / 100.0
        if side == "long":
            entry_limit = mid_price + offset
            stop = self.stops.stop_for_long(entry=entry_limit, atr_5min=atr_5min, or_low=or_low)
        else:
            entry_limit = mid_price - offset
            stop = self.stops.stop_for_short(entry=entry_limit, atr_5min=atr_5min, or_high=or_high)

        s = self.sizing.size(equity=equity, entry=entry_limit, stop=stop)
        if s.is_skip:
            logger.info("plan rejected sym=%s side=%s binding=%s entry=%.4f stop=%.4f",
                        symbol, side, s.binding_cap, entry_limit, stop)
            return None

        per_share_R = abs(entry_limit - stop)
        t2_R = float(self.cfg.exits["t2_R"])
        tp = entry_limit + t2_R * per_share_R if side == "long" else entry_limit - t2_R * per_share_R

        return EntryPlan(
            symbol=symbol.upper(),
            side=side,
            qty=s.shares,
            entry_limit=round(entry_limit, 4),
            stop_loss=round(stop, 4),
            take_profit=round(tp, 4),
            R=round(per_share_R, 6),
            initial_risk_dollars=round(s.risk_dollars, 4),
        )

    # ── Submission with chase ────────────────────────────────────

    def submit_entry_bracket(self, plan: EntryPlan) -> SubmissionResult:
        """Submit an entry bracket. Cancel-and-replace if not filled
        within ``orders.chase_wait_seconds``, up to ``chase_max_attempts``.

        Returns ``SubmissionResult.accepted=True`` only if the entry leg
        fills.  Stop/TP placement is part of the bracket itself."""
        side = plan.side
        attempts_max = int(self.cfg.orders["chase_max_attempts"])
        wait = float(self.cfg.orders["chase_wait_seconds"])

        last_id: str | None = None
        last_coid: str | None = None
        last_status: str = "never_submitted"

        for attempt in range(1, attempts_max + 1):
            coid = make_client_order_id(plan.symbol, "entry")
            req = LimitOrderRequest(
                symbol=plan.symbol,
                qty=plan.qty,
                side=OrderSide.BUY if side == "long" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=plan.entry_limit,
                client_order_id=coid,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=plan.stop_loss),
                take_profit=TakeProfitRequest(limit_price=plan.take_profit),
            )

            try:
                order = self.tc.submit_order(req)
            except Exception as exc:  # alpaca-py raises an SDK exception
                logger.warning("submit_order failed attempt=%d sym=%s err=%s",
                               attempt, plan.symbol, exc)
                continue

            last_id = getattr(order, "id", None)
            last_coid = coid
            # Record into the trade log immediately (status='pending').
            self._record_pending(plan, coid)

            # Wait for fill.
            self._sleep(wait)
            filled, status = self._poll_fill(order_id=last_id)
            last_status = status
            if filled:
                self.store.mark_trade_filled(coid, filled_at=self._clock(), fill_price=plan.entry_limit)
                return SubmissionResult(
                    accepted=True, client_order_id=coid, order_id=last_id,
                    final_status=status, attempts=attempt,
                )

            # Not filled — cancel and retry.
            self._safe_cancel(last_id)
            self.store.cancel_trade(coid)

        return SubmissionResult(
            accepted=False, client_order_id=last_coid, order_id=last_id,
            final_status=last_status, attempts=attempts_max,
            reason="chase_exhausted",
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _record_pending(self, plan: EntryPlan, coid: str) -> None:
        self.store.record_trade(
            client_order_id=coid,
            symbol=plan.symbol, side=plan.side, tranche="entry",
            qty=Decimal(plan.qty),
            entry_price=Decimal(str(plan.entry_limit)),
            stop_loss=Decimal(str(plan.stop_loss)),
            take_profit=Decimal(str(plan.take_profit)),
            initial_risk_dollars=Decimal(str(plan.initial_risk_dollars)),
            submitted_at=self._clock(),
            status="pending",
        )

    def _poll_fill(self, *, order_id: str | None) -> tuple[bool, str]:
        if order_id is None:
            return False, "no_order_id"
        try:
            order = self.tc.get_order_by_id(order_id)
        except Exception as exc:
            logger.warning("get_order_by_id failed id=%s err=%s", order_id, exc)
            return False, "lookup_failed"
        status = getattr(order, "status", None)
        status_str = status.value if hasattr(status, "value") else str(status)
        return status_str == OrderStatus.FILLED.value, status_str

    def _safe_cancel(self, order_id: str | None) -> None:
        if order_id is None:
            return
        try:
            self.tc.cancel_order_by_id(order_id)
        except Exception as exc:
            logger.warning("cancel_order_by_id failed id=%s err=%s", order_id, exc)

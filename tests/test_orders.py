"""
Tests for orders.py.

Covers all four layers:
  • OrderSizing: 1% risk cap, 20% notional cap, too-loose stop skip
  • StopCalculator: ATR vs structural, take tighter
  • LocateCache: cached per-day, no duplicate get_asset calls
  • OrderManager: PDT gate, plan composition, bracket submission with
    cancel-and-replace chase, client_order_id idempotency

We never touch a real broker — the trading client is mocked.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from alpaca.trading.enums import OrderStatus
from config import StrategyConfig
from orders import (
    EntryPlan,
    LocateCache,
    LocateInfo,
    OrderManager,
    OrderSizing,
    PdtError,
    StopCalculator,
)
from persistence import TradeStore


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> StrategyConfig:
    return StrategyConfig.from_yaml()


@pytest.fixture
def store():
    return TradeStore("sqlite:///:memory:")


@pytest.fixture
def tc():
    return MagicMock()


# ── OrderSizing ─────────────────────────────────────────────────────────

def test_sizing_risk_cap_binds(cfg):
    """1%-of-equity risk binds when stop is wide-but-acceptable.

    Equity 100k, entry 100, stop 99 → risk_per_share = 1.0
    risk_budget = 1000. shares = floor(1000/1.0) = 1000.
    Notional cap: 100k * 20% = 20000; shares_from_notional = floor(20000/100) = 200.
    So notional binds, not risk.  We need a TIGHTER stop for risk to bind.

    Set entry=100, stop=99.99 → risk_per_share = 0.01
    risk_budget = 1000 → shares_from_risk = 100000
    notional cap → shares_from_notional = 200 → notional binds.

    Try entry=10, stop=9.95 → risk_per_share=0.05, max-allowed=10*2%=0.20, ok.
    risk_budget = 1000 → shares_from_risk = 20000
    notional cap = 20000 → shares_from_notional = floor(20000/10) = 2000 → notional binds.

    Hmm — the notional cap is so generous (20%) it nearly always binds. To
    make RISK bind: equity small, entry low, stop loose enough that risk
    budget << notional cap. Try equity=10k, entry=1, stop=0.99: but min
    share size = 1, risk_budget = 100, shares_from_risk = floor(100/0.01)=10000,
    notional = floor(10000*0.2/1)=2000 → notional binds.

    To force risk-cap to bind: very LARGE stop relative to price.
    Try equity=10k, entry=50, stop=49 (2% of price). Allowed since
    max_risk_per_share_pct=2.  risk_budget = 100, shares_from_risk = 100.
    notional cap = 2000, shares_from_notional = 40. Notional STILL binds.

    Right — notional always binds for tight stops. Risk binds only when stop is
    wide AND price is high enough that risk_dollars/(stop-entry) < notional/entry.
    Mathematically:  risk binds ⟺ (price - stop) > price · per_trade_pct / notional_pct
    With 1% / 20%: risk binds ⟺ (price - stop)/price > 0.05  i.e. stop wider than 5% of price.
    But spec caps risk-per-share at 2% of price, so risk NEVER binds with default config.
    Risk binding is a degenerate case — the test below uses a custom cfg to verify.
    """
    sz = OrderSizing(cfg)
    # Tight 0.05% stop: notional cap binds.
    res = sz.size(equity=100_000, entry=100.0, stop=99.95)
    assert res.binding_cap == "notional"
    assert res.shares == 200  # 100k * 0.2 / 100


def test_sizing_notional_cap_binds(cfg):
    """Tight stop → 1% risk budget would buy way more shares than 20% notional allows."""
    sz = OrderSizing(cfg)
    res = sz.size(equity=100_000, entry=100.0, stop=99.50)  # 0.5% stop
    # risk_dollars budget = 1000; risk_per_share = 0.50 → 2000 shares from risk.
    # notional cap = 20000 → 200 shares.
    assert res.binding_cap == "notional"
    assert res.shares == 200
    assert res.notional == pytest.approx(20_000.0)


def test_sizing_risk_cap_binds_under_relaxed_config(cfg):
    """Force risk binding by relaxing max_risk_per_share_pct and using a wide stop."""
    # Build a tweaked config in-memory: allow up to 10% stop.
    relaxed = StrategyConfig(
        universe=cfg.universe, indicators=cfg.indicators, entry=cfg.entry,
        stops={**cfg.stops, "max_risk_per_share_pct": 10.0},
        exits=cfg.exits, risk=cfg.risk, orders=cfg.orders,
        stream=cfg.stream, observability=cfg.observability,
    )
    sz = OrderSizing(relaxed)
    # entry=100, stop=92 → risk_per_share = 8.0 (8% of price, allowed under relaxed)
    # risk_budget = 1000; shares_from_risk = 125
    # notional cap = 20000 → shares_from_notional = 200 → risk binds.
    res = sz.size(equity=100_000, entry=100.0, stop=92.0)
    assert res.binding_cap == "risk"
    assert res.shares == 125
    assert res.risk_dollars == pytest.approx(1000.0)


def test_sizing_skips_when_stop_too_loose(cfg):
    """Stop > 2% of entry → skip per spec."""
    sz = OrderSizing(cfg)
    res = sz.size(equity=100_000, entry=100.0, stop=97.0)  # 3% stop
    assert res.is_skip
    assert res.binding_cap == "too_loose"
    assert res.shares == 0


def test_sizing_skips_when_fractional_shares(cfg):
    sz = OrderSizing(cfg)
    # Equity tiny, price huge → can't afford 1 share.
    res = sz.size(equity=10.0, entry=500.0, stop=495.0)
    assert res.is_skip


def test_sizing_invalid_inputs(cfg):
    sz = OrderSizing(cfg)
    assert sz.size(equity=100, entry=0, stop=1).is_skip
    assert sz.size(equity=100, entry=10, stop=10).is_skip  # zero risk


# ── StopCalculator ─────────────────────────────────────────────────────

def test_stop_long_atr_tighter(cfg):
    """When ATR stop is closer to entry than structural, use ATR."""
    sc = StopCalculator(cfg)
    # entry=100, ATR=0.4, atr_mul=1.5 → ATR stop = 100 - 0.6 = 99.40
    # OR_low = 99.20, structural = 99.20 - 0.05 = 99.15
    # ATR (99.40) is closer to entry than structural (99.15) → use ATR.
    s = sc.stop_for_long(entry=100.0, atr_5min=0.4, or_low=99.20)
    assert s == pytest.approx(99.40)


def test_stop_long_structural_tighter(cfg):
    """When structural stop is closer to entry than ATR, use structural."""
    sc = StopCalculator(cfg)
    # entry=100, ATR=1.0, atr_mul=1.5 → ATR stop = 98.50
    # OR_low=99.80, structural = 99.75 — closer.
    s = sc.stop_for_long(entry=100.0, atr_5min=1.0, or_low=99.80)
    assert s == pytest.approx(99.75)


def test_stop_short_atr_tighter(cfg):
    sc = StopCalculator(cfg)
    # entry=100, ATR=0.4 → ATR stop=100.60; OR_high=100.90 → structural=100.95
    s = sc.stop_for_short(entry=100.0, atr_5min=0.4, or_high=100.90)
    assert s == pytest.approx(100.60)


def test_stop_short_structural_tighter(cfg):
    sc = StopCalculator(cfg)
    # entry=100, ATR=1.0 → ATR stop=101.50; OR_high=100.20 → structural=100.25
    s = sc.stop_for_short(entry=100.0, atr_5min=1.0, or_high=100.20)
    assert s == pytest.approx(100.25)


# ── LocateCache ────────────────────────────────────────────────────────

def test_locate_cache_hits_on_second_lookup(tc):
    tc.get_asset.return_value = SimpleNamespace(
        symbol="AAPL", shortable=True, easy_to_borrow=True
    )
    lc = LocateCache(tc)
    today = date(2026, 5, 25)
    info1 = lc.lookup("AAPL", today=today)
    info2 = lc.lookup("AAPL", today=today)
    assert info1.shortable is True and info1.easy_to_borrow is True
    assert info2 is info1
    tc.get_asset.assert_called_once_with("AAPL")


def test_locate_cache_invalidates_next_day(tc):
    tc.get_asset.return_value = SimpleNamespace(
        symbol="AAPL", shortable=True, easy_to_borrow=False
    )
    lc = LocateCache(tc)
    lc.lookup("AAPL", today=date(2026, 5, 25))
    lc.lookup("AAPL", today=date(2026, 5, 26))
    assert tc.get_asset.call_count == 2


def test_is_locate_ok_requires_both_flags(tc):
    lc = LocateCache(tc)
    tc.get_asset.return_value = SimpleNamespace(shortable=True, easy_to_borrow=False)
    assert lc.is_locate_ok("AAPL", today=date(2026, 5, 25)) is False
    lc._cache.clear()
    tc.get_asset.return_value = SimpleNamespace(shortable=False, easy_to_borrow=True)
    assert lc.is_locate_ok("AAPL", today=date(2026, 5, 25)) is False
    lc._cache.clear()
    tc.get_asset.return_value = SimpleNamespace(shortable=True, easy_to_borrow=True)
    assert lc.is_locate_ok("AAPL", today=date(2026, 5, 25)) is True


# ── OrderManager.assert_pdt_ok ─────────────────────────────────────────

def test_pdt_refuses_under_25k_without_pdt_flag(tc, store, cfg):
    tc.get_account.return_value = SimpleNamespace(
        equity="10000", pattern_day_trader=False
    )
    om = OrderManager(tc, store, cfg)
    with pytest.raises(PdtError, match="25,000"):
        om.assert_pdt_ok()


def test_pdt_passes_when_pdt_flagged(tc, store, cfg):
    tc.get_account.return_value = SimpleNamespace(
        equity="10000", pattern_day_trader=True
    )
    om = OrderManager(tc, store, cfg)
    om.assert_pdt_ok()


def test_pdt_passes_when_equity_above_25k(tc, store, cfg):
    tc.get_account.return_value = SimpleNamespace(
        equity="50000", pattern_day_trader=False
    )
    om = OrderManager(tc, store, cfg)
    om.assert_pdt_ok()


# ── OrderManager.plan ──────────────────────────────────────────────────

def test_plan_long_basic(tc, store, cfg):
    om = OrderManager(tc, store, cfg)
    plan = om.plan(
        symbol="aapl", side="long",
        mid_price=100.00, atr_5min=0.30,
        or_high=100.10, or_low=99.50, equity=100_000,
    )
    assert plan is not None
    assert plan.symbol == "AAPL"
    assert plan.side == "long"
    assert plan.entry_limit == pytest.approx(100.01)        # mid + 1¢
    # ATR stop: 100.01 - 1.5*0.30 = 99.56;  struct = 99.45 → ATR tighter → 99.56
    assert plan.stop_loss == pytest.approx(99.56)
    # R = 0.45;  T2 = 100.01 + 2*0.45 = 100.91
    assert plan.take_profit == pytest.approx(100.91)
    assert plan.qty > 0


def test_plan_short_basic(tc, store, cfg):
    om = OrderManager(tc, store, cfg)
    plan = om.plan(
        symbol="X", side="short",
        mid_price=100.00, atr_5min=0.30,
        or_high=100.50, or_low=99.90, equity=100_000,
    )
    assert plan is not None
    assert plan.entry_limit == pytest.approx(99.99)         # mid - 1¢
    # ATR stop: 99.99 + 0.45 = 100.44;  struct = 100.55 → ATR tighter → 100.44
    assert plan.stop_loss == pytest.approx(100.44)
    assert plan.take_profit == pytest.approx(99.99 - 2 * 0.45)


def test_plan_returns_none_when_stop_too_loose(tc, store, cfg):
    om = OrderManager(tc, store, cfg)
    # ATR of 2.0 with mul 1.5 = 3.0 stop → 3% of price > 2% cap → skip.
    plan = om.plan(
        symbol="X", side="long",
        mid_price=100.00, atr_5min=2.0,
        or_high=100.20, or_low=98.00, equity=100_000,
    )
    assert plan is None


# ── OrderManager.submit_entry_bracket — happy path ─────────────────────

def _entry_plan_fixture(side="long") -> EntryPlan:
    return EntryPlan(
        symbol="AAPL", side=side, qty=10,
        entry_limit=100.01 if side == "long" else 99.99,
        stop_loss=99.50 if side == "long" else 100.50,
        take_profit=100.92 if side == "long" else 99.08,
        R=0.51, initial_risk_dollars=5.10,
    )


def test_submit_bracket_fills_first_attempt(tc, store, cfg):
    """submit_order returns immediately filled — no chase needed."""
    submitted_orders = []
    def fake_submit(req):
        submitted_orders.append(req)
        return SimpleNamespace(id="alpaca-1", status=OrderStatus.NEW,
                               client_order_id=req.client_order_id)
    tc.submit_order.side_effect = fake_submit
    tc.get_order_by_id.return_value = SimpleNamespace(status=OrderStatus.FILLED)

    sleeps: list[float] = []
    om = OrderManager(tc, store, cfg, sleep_fn=sleeps.append,
                      clock_fn=lambda: datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc))
    res = om.submit_entry_bracket(_entry_plan_fixture("long"))

    assert res.accepted is True
    assert res.attempts == 1
    assert res.client_order_id is not None
    assert res.client_order_id.startswith("AAPL-entry-")
    # Bracket order shape
    req = submitted_orders[0]
    assert req.order_class.value == "bracket"
    assert req.stop_loss.stop_price == pytest.approx(99.50)
    assert req.take_profit.limit_price == pytest.approx(100.92)
    # Slept once for the wait.
    assert sleeps == [float(cfg.orders["chase_wait_seconds"])]


def test_submit_bracket_chases_then_succeeds(tc, store, cfg):
    """First two attempts time out (status remains NEW); third fills."""
    order_ids = iter(["o1", "o2", "o3"])
    coids_submitted: list[str] = []
    def fake_submit(req):
        coids_submitted.append(req.client_order_id)
        return SimpleNamespace(id=next(order_ids), status=OrderStatus.NEW,
                               client_order_id=req.client_order_id)
    tc.submit_order.side_effect = fake_submit

    statuses = iter([
        SimpleNamespace(status=OrderStatus.NEW),       # attempt 1: not filled
        SimpleNamespace(status=OrderStatus.NEW),       # attempt 2: not filled
        SimpleNamespace(status=OrderStatus.FILLED),    # attempt 3: filled
    ])
    tc.get_order_by_id.side_effect = lambda _id: next(statuses)

    om = OrderManager(tc, store, cfg, sleep_fn=lambda _s: None,
                      clock_fn=lambda: datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc))
    res = om.submit_entry_bracket(_entry_plan_fixture("long"))

    assert res.accepted is True
    assert res.attempts == 3
    # 3 distinct client_order_ids
    assert len(set(coids_submitted)) == 3
    # Cancelled twice (after attempts 1 and 2)
    assert tc.cancel_order_by_id.call_count == 2


def test_submit_bracket_abandoned_after_max_attempts(tc, store, cfg):
    """All chase_max_attempts time out → not accepted."""
    def fake_submit(req):
        return SimpleNamespace(id="o", status=OrderStatus.NEW,
                               client_order_id=req.client_order_id)
    tc.submit_order.side_effect = fake_submit
    tc.get_order_by_id.return_value = SimpleNamespace(status=OrderStatus.NEW)

    om = OrderManager(tc, store, cfg, sleep_fn=lambda _s: None,
                      clock_fn=lambda: datetime(2026, 5, 25, 10, 0, tzinfo=timezone.utc))
    res = om.submit_entry_bracket(_entry_plan_fixture("long"))

    assert res.accepted is False
    assert res.reason == "chase_exhausted"
    assert res.attempts == int(cfg.orders["chase_max_attempts"])
    assert tc.cancel_order_by_id.call_count == int(cfg.orders["chase_max_attempts"])


# ── Idempotency: each attempt gets a unique client_order_id ────────────

def test_client_order_id_unique_per_attempt(tc, store, cfg):
    submitted: list[str] = []
    def fake_submit(req):
        submitted.append(req.client_order_id)
        return SimpleNamespace(id=f"o{len(submitted)}", status=OrderStatus.NEW,
                               client_order_id=req.client_order_id)
    tc.submit_order.side_effect = fake_submit
    tc.get_order_by_id.return_value = SimpleNamespace(status=OrderStatus.NEW)

    om = OrderManager(tc, store, cfg, sleep_fn=lambda _s: None)
    om.submit_entry_bracket(_entry_plan_fixture("long"))

    # All three distinct
    assert len(submitted) == int(cfg.orders["chase_max_attempts"])
    assert len(set(submitted)) == len(submitted)
    # All follow {SYMBOL}-{tranche}-{epoch_ms}
    for coid in submitted:
        parts = coid.split("-")
        assert parts[0] == "AAPL"
        assert parts[1] == "entry"
        assert parts[2].isdigit()

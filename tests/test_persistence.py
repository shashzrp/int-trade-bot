"""Tests for the trade-log persistence layer."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError

from persistence import NY_TZ, TradeStore, make_client_order_id


@pytest.fixture
def store():
    return TradeStore("sqlite:///:memory:")


# ── Schema / smoke ──────────────────────────────────────────────────────

def test_tables_created(store):
    """All three tables created on TradeStore init."""
    from sqlalchemy import inspect
    insp = inspect(store.engine)
    assert {"trades", "signals", "equity_curve"} <= set(insp.get_table_names())


def test_client_order_id_format():
    coid = make_client_order_id("aapl", "entry", epoch_ms=1735_000_000_000)
    assert coid == "AAPL-entry-1735000000000"


# ── Trade idempotency ──────────────────────────────────────────────────

def _now_ny() -> datetime:
    return datetime.now(tz=NY_TZ)


def test_record_trade_then_close(store):
    coid = make_client_order_id("AAPL", "entry")
    t = store.record_trade(
        client_order_id=coid, symbol="AAPL", side="long", tranche="entry",
        qty=100, entry_price=200.0, stop_loss=199.0, take_profit=202.0,
        initial_risk_dollars=100.0, submitted_at=_now_ny(),
    )
    assert t.id is not None
    assert t.status == "pending"

    store.mark_trade_filled(coid, filled_at=_now_ny(), fill_price=200.05)
    store.close_trade(
        coid, exit_price=201.5, closed_at=_now_ny(),
        realized_pnl=145.0, exit_reason="T1_target",
    )
    # Daily P&L picks it up
    today = datetime.now(tz=NY_TZ).date()
    assert store.get_daily_pnl(today) == Decimal("145")


def test_duplicate_client_order_id_rejected(store):
    coid = "AAPL-entry-1"
    store.record_trade(
        client_order_id=coid, symbol="AAPL", side="long", tranche="entry",
        qty=10, entry_price=100, stop_loss=99, take_profit=101,
        initial_risk_dollars=10, submitted_at=_now_ny(),
    )
    with pytest.raises(IntegrityError):
        store.record_trade(
            client_order_id=coid, symbol="AAPL", side="long", tranche="entry",
            qty=10, entry_price=100, stop_loss=99, take_profit=101,
            initial_risk_dollars=10, submitted_at=_now_ny(),
        )


def test_naive_datetime_refused(store):
    with pytest.raises(ValueError, match="tz-aware"):
        store.record_trade(
            client_order_id="X-entry-1", symbol="X", side="long", tranche="entry",
            qty=1, entry_price=1, stop_loss=1, take_profit=1,
            initial_risk_dollars=1, submitted_at=datetime.now(),  # naive
        )


# ── Signal log ─────────────────────────────────────────────────────────

def test_record_signal_round_trip(store):
    sig = store.record_signal(
        symbol="AAPL", evaluated_at=_now_ny(), side="long",
        passed=False, rejected_gate="rsi_band",
        gates={"price_above_or_high": True, "rsi_band": False, "vwap_slope": True},
        indicators_snapshot={"vwap": 200.1, "rsi": 75.0, "ema9": 200.05},
    )
    assert sig.id is not None
    # Read back
    with store.session() as s:
        from persistence import Signal
        from sqlalchemy import select
        row = s.execute(select(Signal).where(Signal.id == sig.id)).scalar_one()
        assert row.gates["rsi_band"] is False
        assert row.indicators_snapshot["rsi"] == 75.0


# ── Risk-manager queries ───────────────────────────────────────────────

def _quickfill(store, symbol, pnl, when, tranche="entry"):
    coid = make_client_order_id(symbol, tranche, epoch_ms=int(when.timestamp() * 1000))
    store.record_trade(
        client_order_id=coid, symbol=symbol, side="long", tranche=tranche,
        qty=1, entry_price=100, stop_loss=99, take_profit=101,
        initial_risk_dollars=1, submitted_at=when,
    )
    store.mark_trade_filled(coid, filled_at=when + timedelta(seconds=5))
    store.close_trade(
        coid, exit_price=100 + pnl, closed_at=when + timedelta(minutes=10),
        realized_pnl=pnl, exit_reason="test",
    )
    return coid


def test_daily_pnl_sums_only_todays_closed_trades(store):
    today = datetime.now(tz=NY_TZ)
    yesterday = today - timedelta(days=1)
    _quickfill(store, "AAA", Decimal("10"), today)
    _quickfill(store, "BBB", Decimal("-3"), today)
    _quickfill(store, "CCC", Decimal("99"), yesterday)
    pnl = store.get_daily_pnl(today.date())
    assert pnl == Decimal("7")


def test_consecutive_losses_walks_back_until_a_win(store):
    base = datetime.now(tz=NY_TZ)
    # Order: oldest first
    _quickfill(store, "A", Decimal("5"),  base + timedelta(minutes=1))   # win
    _quickfill(store, "B", Decimal("-1"), base + timedelta(minutes=2))   # loss
    _quickfill(store, "C", Decimal("-2"), base + timedelta(minutes=3))   # loss
    _quickfill(store, "D", Decimal("-3"), base + timedelta(minutes=4))   # loss (most recent)
    assert store.get_consecutive_losses() == 3


def test_consecutive_losses_zero_if_last_is_win(store):
    base = datetime.now(tz=NY_TZ)
    _quickfill(store, "A", Decimal("-5"), base + timedelta(minutes=1))
    _quickfill(store, "B", Decimal("5"),  base + timedelta(minutes=2))   # most recent: win
    assert store.get_consecutive_losses() == 0


def test_trades_today_counts_only_entry_tranche(store):
    today = datetime.now(tz=NY_TZ)
    _quickfill(store, "X", Decimal("1"), today, tranche="entry")
    _quickfill(store, "X", Decimal("1"), today, tranche="T1")
    _quickfill(store, "X", Decimal("1"), today, tranche="T2")
    _quickfill(store, "Y", Decimal("1"), today, tranche="entry")
    assert store.get_trades_today_count(today.date()) == 2


def test_open_positions_returns_filled_entries_only(store):
    base = datetime.now(tz=NY_TZ)
    # One open
    coid_open = make_client_order_id("OPEN", "entry", epoch_ms=1)
    store.record_trade(
        client_order_id=coid_open, symbol="OPEN", side="long", tranche="entry",
        qty=1, entry_price=100, stop_loss=99, take_profit=101,
        initial_risk_dollars=1, submitted_at=base,
    )
    store.mark_trade_filled(coid_open, filled_at=base)
    # One closed
    _quickfill(store, "CLS", Decimal("1"), base)
    # One still pending
    coid_pending = make_client_order_id("PND", "entry", epoch_ms=2)
    store.record_trade(
        client_order_id=coid_pending, symbol="PND", side="long", tranche="entry",
        qty=1, entry_price=100, stop_loss=99, take_profit=101,
        initial_risk_dollars=1, submitted_at=base,
    )

    open_trades = store.get_open_positions()
    syms = {t.symbol for t in open_trades}
    assert syms == {"OPEN"}


def test_equity_snapshot_round_trip(store):
    ts = _now_ny()
    snap = store.record_equity_snapshot(
        ts=ts, equity=100_000, cash=50_000, position_count=2, daily_pnl=1234.5
    )
    assert snap.id is not None
    assert snap.equity == Decimal("100000")
    assert snap.daily_pnl == Decimal("1234.5")


def test_weekly_pnl(store):
    monday_ny = datetime(2026, 5, 25, 9, 30, tzinfo=NY_TZ)  # 2026-05-25 is a Monday
    _quickfill(store, "A", Decimal("100"), monday_ny + timedelta(days=0))
    _quickfill(store, "B", Decimal("-30"), monday_ny + timedelta(days=2))
    _quickfill(store, "C", Decimal("50"),  monday_ny + timedelta(days=4))
    # Trade from next week should not count
    _quickfill(store, "D", Decimal("999"), monday_ny + timedelta(days=7))
    assert store.get_weekly_pnl(monday_ny.date() + timedelta(days=3)) == Decimal("120")

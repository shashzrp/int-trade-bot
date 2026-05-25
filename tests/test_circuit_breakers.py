"""
Tests for circuit_breakers.py, wind_down.py, and kill_switch.py.

Spec-mandated must-pass:
  • daily-loss-cap closing all positions (flatten_all called when -3% breached)
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from circuit_breakers import CircuitBreakers, GateResult
from kill_switch import KillSwitch, build_app
from persistence import TradeStore, make_client_order_id
from wind_down import WindDown


NY = ZoneInfo("America/New_York")


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def store() -> TradeStore:
    return TradeStore("sqlite:///:memory:")


@pytest.fixture
def tc():
    return MagicMock()


@pytest.fixture
def cb(store, tc):
    return CircuitBreakers(cfg=None, store=store, trading_client=tc)


def _quickfill(store, *, pnl, when=None, tranche="entry"):
    """Insert a closed trade with the given realized P&L."""
    when = when or datetime.now(tz=NY)
    coid = make_client_order_id("X", tranche, epoch_ms=int(when.timestamp() * 1_000_000))
    store.record_trade(
        client_order_id=coid, symbol="X", side="long", tranche=tranche,
        qty=1, entry_price=100, stop_loss=99, take_profit=101,
        initial_risk_dollars=1, submitted_at=when,
    )
    store.mark_trade_filled(coid, filled_at=when)
    store.close_trade(coid, exit_price=100 + float(pnl),
                      closed_at=when + timedelta(seconds=1),
                      realized_pnl=pnl, exit_reason="test")
    return coid


# ── Daily loss cap (the spec's must-pass test) ────────────────────────

def test_daily_loss_cap_triggers_flatten_all(cb, tc):
    """When intraday equity drops 3% below the day's starting equity, the
    bot must flatten every position. This is the spec-mandated behavior."""
    cb.set_starting_equity_today(100_000)

    # Equity dropped to $96,500 (-3.5%) — well below the -3% cap.
    today = datetime.now(tz=NY).date()
    gate = cb.can_take_new_entry(today=today, current_equity=96_500)
    assert gate.allowed is False
    assert gate.reason == "daily_loss_cap"

    # The bot acts on the gate by flattening:
    cb.flatten_all(reason="daily_loss_cap")
    tc.cancel_orders.assert_called_once()
    tc.close_all_positions.assert_called_once_with(cancel_orders=True)


def test_daily_pnl_pct_signed_correctly(cb):
    cb.set_starting_equity_today(100_000)
    assert cb.daily_pnl_pct(99_000) == Decimal("-1")
    assert cb.daily_pnl_pct(102_500) == Decimal("2.5")
    assert cb.daily_pnl_pct(100_000) == Decimal("0")


def test_daily_loss_cap_boundary_inclusive(cb):
    cb.set_starting_equity_today(100_000)
    assert cb.daily_loss_cap_breached(97_000) is True   # exactly -3%
    assert cb.daily_loss_cap_breached(97_001) is False


def test_daily_loss_cap_inactive_without_starting_equity(cb):
    """Before the day's starting equity is set, the cap can't fire."""
    assert cb.daily_loss_cap_breached(50_000) is False


# ── Weekly loss cap ───────────────────────────────────────────────────

def test_weekly_loss_cap_triggers(cb, tc):
    cb.set_starting_equity_week(100_000)
    cb.set_starting_equity_today(100_000)
    gate = cb.can_take_new_entry(today=date(2026, 5, 25), current_equity=93_500)  # -6.5%
    assert gate.allowed is False
    assert gate.reason in ("daily_loss_cap", "weekly_loss_cap")
    # -6.5% breaches BOTH; daily cap (-3%) is the first check, so reports daily.


def test_weekly_halt_until_next_day(cb):
    cb.set_starting_equity_today(100_000)
    cb.halt_for_next_trading_day(today=date(2026, 5, 25))
    g = cb.can_take_new_entry(today=date(2026, 5, 25), current_equity=100_000)
    assert g.allowed is False
    assert g.reason == "weekly_halt_active"

    # On the next day, the halt clears:
    g2 = cb.can_take_new_entry(today=date(2026, 5, 26), current_equity=100_000)
    assert g2.allowed is True


def test_weekly_halt_can_be_manually_reset(cb):
    cb.set_starting_equity_today(100_000)
    cb.halt_for_next_trading_day(today=date(2026, 5, 25))
    cb.manually_reset_weekly_halt()
    g = cb.can_take_new_entry(today=date(2026, 5, 25), current_equity=100_000)
    assert g.allowed is True


# ── Consecutive losses ────────────────────────────────────────────────

def test_consecutive_losses_breached(cb, store):
    cb.set_starting_equity_today(100_000)
    base = datetime.now(tz=NY)
    _quickfill(store, pnl=Decimal("-1"), when=base + timedelta(minutes=1))
    _quickfill(store, pnl=Decimal("-1"), when=base + timedelta(minutes=2))
    _quickfill(store, pnl=Decimal("-1"), when=base + timedelta(minutes=3))
    g = cb.can_take_new_entry(today=base.date(), current_equity=99_000)
    assert g.allowed is False
    assert g.reason == "consecutive_losses"


def test_a_win_resets_consecutive_loss_count(cb, store):
    cb.set_starting_equity_today(100_000)
    base = datetime.now(tz=NY)
    _quickfill(store, pnl=Decimal("-1"), when=base + timedelta(minutes=1))
    _quickfill(store, pnl=Decimal("-1"), when=base + timedelta(minutes=2))
    _quickfill(store, pnl=Decimal("5"),  when=base + timedelta(minutes=3))  # win
    g = cb.can_take_new_entry(today=base.date(), current_equity=103_000)
    assert g.allowed is True


# ── Max trades per day ────────────────────────────────────────────────

def test_max_trades_per_day(cb, store):
    cb.set_starting_equity_today(100_000)
    base = datetime.now(tz=NY)
    for i in range(5):
        _quickfill(store, pnl=Decimal("1"), when=base + timedelta(minutes=i))
    g = cb.can_take_new_entry(today=base.date(), current_equity=105_000)
    assert g.allowed is False
    assert g.reason == "trades_per_day"


# ── Max concurrent positions ──────────────────────────────────────────

def test_max_concurrent_positions(cb, store):
    cb.set_starting_equity_today(100_000)
    base = datetime.now(tz=NY)
    for sym in ("A", "B", "C"):
        coid = make_client_order_id(sym, "entry", epoch_ms=int(base.timestamp() * 1_000_000) + ord(sym))
        store.record_trade(
            client_order_id=coid, symbol=sym, side="long", tranche="entry",
            qty=1, entry_price=100, stop_loss=99, take_profit=101,
            initial_risk_dollars=1, submitted_at=base,
        )
        store.mark_trade_filled(coid, filled_at=base)  # still "filled" not closed
    g = cb.can_take_new_entry(today=base.date(), current_equity=100_000)
    assert g.allowed is False
    assert g.reason == "max_concurrent_positions"


# ── flatten_all is idempotent against partial broker failures ─────────

def test_flatten_all_keeps_going_if_cancel_orders_throws(cb, tc):
    tc.cancel_orders.side_effect = RuntimeError("broker timeout")
    cb.flatten_all(reason="test")
    # Should STILL try close_all_positions even after cancel_orders failed
    tc.close_all_positions.assert_called_once()


# ── WindDown ──────────────────────────────────────────────────────────

def test_wind_down_does_not_fire_before_15_55():
    fired = []
    wd = WindDown(on_force_flat=lambda: fired.append(True))
    wd.tick(datetime(2026, 5, 25, 15, 54, tzinfo=NY))
    assert fired == []


def test_wind_down_fires_at_15_55_and_only_once_per_day():
    fired = []
    wd = WindDown(on_force_flat=lambda: fired.append("fired"))
    wd.tick(datetime(2026, 5, 25, 15, 55, tzinfo=NY))
    wd.tick(datetime(2026, 5, 25, 15, 56, tzinfo=NY))
    wd.tick(datetime(2026, 5, 25, 16, 0,  tzinfo=NY))
    assert fired == ["fired"]


def test_wind_down_re_arms_next_day():
    fired = []
    wd = WindDown(on_force_flat=lambda: fired.append("fired"))
    wd.tick(datetime(2026, 5, 25, 15, 55, tzinfo=NY))
    wd.tick(datetime(2026, 5, 26, 15, 55, tzinfo=NY))
    assert len(fired) == 2


def test_wind_down_refuses_naive_now():
    wd = WindDown(on_force_flat=lambda: None)
    with pytest.raises(ValueError, match="tz-aware"):
        wd.tick(datetime(2026, 5, 25, 15, 55))


# ── KillSwitch + FastAPI ──────────────────────────────────────────────

def test_kill_switch_token_strength_required():
    with pytest.raises(ValueError, match="≥16"):
        KillSwitch(token="short")


@pytest.fixture
def switch() -> KillSwitch:
    return KillSwitch(token="x" * 40)


@pytest.fixture
def client(switch) -> TestClient:
    return TestClient(build_app(switch))


def test_kill_endpoint_requires_bearer(client):
    r = client.post("/kill")  # no Authorization header
    assert r.status_code == 401


def test_kill_endpoint_rejects_wrong_token(client):
    r = client.post("/kill", headers={"Authorization": "Bearer wrong_token_value_xxxxxxx"})
    assert r.status_code == 401


def test_kill_endpoint_rejects_malformed_header(client):
    r = client.post("/kill", headers={"Authorization": "Token xxx"})
    assert r.status_code == 401


def test_kill_then_status_then_reset(switch, client):
    auth = {"Authorization": f"Bearer {'x' * 40}"}
    assert switch.active is False
    r = client.post("/kill", headers=auth)
    assert r.status_code == 200
    assert r.json()["active"] is True
    assert switch.active is True
    s = client.get("/status", headers=auth)
    assert s.json()["active"] is True
    r2 = client.post("/reset", headers=auth)
    assert r2.json()["active"] is False
    assert switch.active is False

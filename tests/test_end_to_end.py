"""
End-to-end integration test (Step 13).

We can't run a real Alpaca paper session from inside CI, but we CAN
verify that all the wiring works:

  scanner watchlist  →  state_machine.evaluate  →  orders.submit_entry_bracket
  bar fill on EM trigger → exit_manager.evaluate_bar → exit action emitted
  signals + trades land in the SQLite store

This test constructs synthetic bars designed to fire a LONG entry on a
specific minute, hands them to the same evaluate code paths the live
bot uses, and asserts the persistence layer records the event correctly.

The actual live-paper run is done manually:

    cp .env.example .env  &&  edit .env  &&  python main.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from alpaca.trading.enums import OrderStatus

from circuit_breakers import CircuitBreakers
from exit_manager import ExitManager
from indicators import opening_range, resample_to_5min, atr
from orders import OrderManager
from persistence import TradeStore
from state_machine import StateMachine, SymbolState


NY = ZoneInfo("America/New_York")


# ── Build a bar sequence guaranteed to fire a long entry ────────────────

def _build_long_signal_bars():
    """Mirror the long-passing baseline from test_entry_gates."""
    start = datetime(2026, 5, 25, 9, 30, tzinfo=NY)
    n_or, n_post = 15, 30
    n = n_or + n_post
    times = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(n)])

    closes = np.empty(n)
    closes[:n_or] = 100.0 + 0.5 * np.sin(np.linspace(0, np.pi * 2, n_or))
    trend = np.linspace(100.10, 100.40, n_post - 1)
    wobble = 0.08 * np.sin(np.linspace(0, np.pi * 6, n_post - 1))
    closes[n_or:-1] = trend + wobble
    closes[-1] = 100.55      # breakout

    highs = closes + 0.04
    lows  = closes - 0.04
    opens = np.r_[closes[0], closes[:-1]]
    vols = np.r_[
        np.full(n_or, 1000.0),
        np.linspace(1500.0, 2000.0, n_post - 1),
        [50000.0],
    ]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": vols},
        index=times,
    )


# ── The integration test ──────────────────────────────────────────────

def test_end_to_end_long_signal_pipelines_through():
    """Construct the full pipeline with mock broker, feed it bars that
    SHOULD pass all long gates, and verify each layer recorded its event."""
    store = TradeStore("sqlite:///:memory:")

    # Mock trading client — accepts bracket orders, reports them as immediately filled.
    tc = MagicMock()
    tc.get_account.return_value = SimpleNamespace(
        equity="100000", cash="100000", pattern_day_trader=True,
        daytrading_buying_power="400000",
    )
    submitted_orders = []
    def _submit(req):
        submitted_orders.append(req)
        return SimpleNamespace(
            id=f"order-{len(submitted_orders)}",
            status=OrderStatus.NEW,
            client_order_id=req.client_order_id,
        )
    tc.submit_order.side_effect = _submit
    tc.get_order_by_id.return_value = SimpleNamespace(status=OrderStatus.FILLED)
    tc.get_asset.return_value = SimpleNamespace(
        symbol="AAPL", shortable=True, easy_to_borrow=True
    )

    cb = CircuitBreakers(cfg=None, store=store, trading_client=tc)
    cb.set_starting_equity_today(100_000)
    orders = OrderManager(
        tc, store, cfg=None, sleep_fn=lambda _s: None,
        clock_fn=lambda: datetime(2026, 5, 25, 10, 14, tzinfo=timezone.utc),
    )
    sm = StateMachine(
        cfg=None,
        daily_loss_cap_ok=lambda: not cb.daily_loss_cap_breached(100_000),
        locate_ok=lambda _s: True,
    )
    em = ExitManager(cfg=None, kill_switch_active=lambda: False)

    # ── Build bar history and record OR ─────────────────────────
    bars = _build_long_signal_bars()
    or_pair = opening_range(bars, session_date=bars.index[0].date(), minutes=15)
    assert or_pair is not None
    sm.record_or("AAPL", *or_pair)

    # Compute ATR(14) on 5-min — but the synthetic single-day bar set is
    # short; backtest-mode harness handles longer histories.  For this
    # integration test we just need atr_5min > 0.
    df5 = resample_to_5min(bars)
    atr5 = float(atr(df5, period=5).iloc[-1])  # smaller period for short series

    # ── Evaluate the final bar (the breakout) ─────────────────
    now = bars.index[-1]
    decision = sm.evaluate("AAPL", bars, now=now, atr_5min=atr5)
    assert decision.action == "long", (
        f"Long signal failed: rejected_gate={decision.rejected_gate} "
        f"failed={[k for k,v in decision.gates.items() if not v]}"
    )

    # ── Plan + submit the bracket ───────────────────────────────
    plan = orders.plan(
        symbol="AAPL", side="long",
        mid_price=float(decision.indicators["close"]),
        atr_5min=atr5,
        or_high=sm.get_or("AAPL")[0], or_low=sm.get_or("AAPL")[1],
        equity=100_000,
    )
    assert plan is not None
    assert plan.qty > 0
    result = orders.submit_entry_bracket(plan)
    assert result.accepted is True

    # ── Track in EM and state machine ───────────────────────────
    em.open_position(
        symbol="AAPL", side="long",
        entry_price=plan.entry_limit, qty=plan.qty,
        stop_price=plan.stop_loss,
    )
    sm.mark_position_opened("AAPL")
    assert sm.get_symbol_state("AAPL") == SymbolState.IN_POSITION

    # ── Verify persistence captured the entry ───────────────────
    open_trades = store.get_open_positions()
    assert len(open_trades) == 1
    t = open_trades[0]
    assert t.symbol == "AAPL"
    assert t.side == "long"
    assert t.tranche == "entry"
    assert t.qty == plan.qty
    assert t.client_order_id.startswith("AAPL-entry-")

    # ── Now feed a bar that should trigger T1 (price hits entry + 1R) ──
    t1_bar = {
        "open":   plan.entry_limit,
        "high":   plan.entry_limit + plan.R + 0.10,  # well past +1R
        "low":    plan.entry_limit - 0.01,
        "close":  plan.entry_limit + plan.R,
        "volume": 5000.0,
    }
    ind_snap = {"vwap": 100.40, "ema_fast": plan.entry_limit + 0.05,
                "rsi": 60.0, "macd_hist": 0.05, "rvol_bar": 3.0}
    action = em.evaluate_bar("AAPL", bar=t1_bar, indicators=ind_snap,
                             now=now + timedelta(minutes=1))
    assert action is not None
    assert action.tranche == "T1"
    assert action.new_stop == pytest.approx(plan.entry_limit)  # moved to BE
    assert action.qty > 0

    # ── Cleanup: close the position ─────────────────────────────
    em.close_position("AAPL")
    sm.mark_position_closed("AAPL", at=now + timedelta(minutes=1))


def test_end_to_end_short_signal_pipelines_through():
    """Mirror test for the SHORT path — locate flag and bearish baseline."""
    store = TradeStore("sqlite:///:memory:")
    tc = MagicMock()
    tc.get_account.return_value = SimpleNamespace(
        equity="100000", pattern_day_trader=True,
    )
    tc.submit_order.side_effect = lambda req: SimpleNamespace(
        id="o", status=OrderStatus.NEW, client_order_id=req.client_order_id
    )
    tc.get_order_by_id.return_value = SimpleNamespace(status=OrderStatus.FILLED)
    tc.get_asset.return_value = SimpleNamespace(
        symbol="AAPL", shortable=True, easy_to_borrow=True
    )

    orders = OrderManager(tc, store, cfg=None, sleep_fn=lambda _s: None,
                          clock_fn=lambda: datetime(2026, 5, 25, 10, 14, tzinfo=timezone.utc))
    sm = StateMachine(cfg=None, daily_loss_cap_ok=lambda: True,
                      locate_ok=lambda _s: True)
    em = ExitManager(cfg=None, kill_switch_active=lambda: False)

    # Build a SHORT-passing bar set
    start = datetime(2026, 5, 25, 9, 30, tzinfo=NY)
    n = 45
    times = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(n)])
    closes = np.empty(n)
    closes[:15] = 100.0 + 0.5 * np.sin(np.linspace(0, np.pi * 2, 15))
    trend = np.linspace(99.90, 99.60, 29)
    wobble = 0.08 * np.sin(np.linspace(0, np.pi * 6, 29))
    closes[15:-1] = trend + wobble
    closes[-1] = 99.45
    bars = pd.DataFrame({
        "open": np.r_[closes[0], closes[:-1]],
        "high": closes + 0.04, "low": closes - 0.04, "close": closes,
        "volume": np.r_[np.full(15, 1000.0), np.linspace(1500, 2000, 29), [50000.0]],
    }, index=times)

    sm.record_or("AAPL", *opening_range(bars, session_date=bars.index[0].date(), minutes=15))
    df5 = resample_to_5min(bars)
    atr5 = float(atr(df5, period=5).iloc[-1])
    now = bars.index[-1]
    decision = sm.evaluate("AAPL", bars, now=now, atr_5min=atr5)
    assert decision.action == "short", (
        f"Short signal failed: rejected_gate={decision.rejected_gate} "
        f"failed={[k for k,v in decision.gates.items() if not v]}"
    )

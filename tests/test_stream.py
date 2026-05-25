"""
Tests for the bar-close validation and heartbeat in stream.py.

The live websocket path is exercised by the Step-13 dry run; not unit-tested.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stream import BarStream, is_bar_closed


NY = ZoneInfo("America/New_York")
UTC = timezone.utc


# ── is_bar_closed ──────────────────────────────────────────────────────

def test_bar_not_closed_at_start():
    bs = datetime(2026, 5, 25, 10, 0, tzinfo=NY)
    assert is_bar_closed(bar_start=bs, now=bs) is False


def test_bar_not_closed_at_59_seconds():
    bs = datetime(2026, 5, 25, 10, 0, tzinfo=NY)
    now = bs + timedelta(seconds=59)
    assert is_bar_closed(bar_start=bs, now=now) is False


def test_bar_not_closed_at_60_seconds_due_to_grace():
    bs = datetime(2026, 5, 25, 10, 0, tzinfo=NY)
    now = bs + timedelta(seconds=60)        # default grace = 2s
    assert is_bar_closed(bar_start=bs, now=now) is False


def test_bar_closed_at_62_seconds():
    bs = datetime(2026, 5, 25, 10, 0, tzinfo=NY)
    now = bs + timedelta(seconds=62)
    assert is_bar_closed(bar_start=bs, now=now) is True


def test_bar_closed_well_after_close():
    bs = datetime(2026, 5, 25, 10, 0, tzinfo=NY)
    now = bs + timedelta(minutes=5)
    assert is_bar_closed(bar_start=bs, now=now) is True


def test_bar_closed_refuses_naive():
    with pytest.raises(ValueError, match="tz-aware"):
        is_bar_closed(
            bar_start=datetime(2026, 5, 25, 10, 0),  # naive
            now=datetime(2026, 5, 25, 10, 5, tzinfo=NY),
        )


def test_custom_grace_seconds():
    bs = datetime(2026, 5, 25, 10, 0, tzinfo=NY)
    now = bs + timedelta(seconds=61)
    assert is_bar_closed(bar_start=bs, now=now, grace_seconds=0) is True
    assert is_bar_closed(bar_start=bs, now=now, grace_seconds=2) is False


# ── BarStream._handle_bar / heartbeat ──────────────────────────────────

def _fake_bar(symbol: str, ts: datetime, *, close: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        timestamp=pd.Timestamp(ts).tz_convert(UTC),
        open=close - 0.1, high=close + 0.1, low=close - 0.2, close=close,
        volume=1500.0,
    )


@pytest.fixture
def fake_clock():
    """Mutable wall-clock for tests."""
    state = {"now": datetime(2026, 5, 25, 10, 2, tzinfo=UTC)}

    def get_now() -> datetime:
        return state["now"]
    def advance(seconds: int) -> None:
        state["now"] += timedelta(seconds=seconds)
    return SimpleNamespace(get_now=get_now, advance=advance, set=lambda d: state.__setitem__("now", d))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_handle_bar_dispatches_when_closed(fake_clock):
    received: list[tuple[str, pd.Series, datetime]] = []

    async def cb(sym, series, ts):
        received.append((sym, series, ts))

    stream_mock = MagicMock()
    bs = BarStream(stream_mock, ["AAPL"], on_closed_bar=cb, clock=fake_clock.get_now)

    # Bar started 10:01 NY (= 14:01 UTC), wall clock is 10:02 UTC = 14:02 UTC.
    # 14:02 UTC is 60s past start AT THE EARLIEST, but we need ≥ 62s (grace).
    bar_start = datetime(2026, 5, 25, 14, 1, tzinfo=UTC)
    fake_clock.set(bar_start + timedelta(seconds=63))
    _run(bs._handle_bar(_fake_bar("AAPL", bar_start)))

    assert len(received) == 1
    sym, series, ts = received[0]
    assert sym == "AAPL"
    assert series["close"] == 100.0
    # ts is in NY
    assert ts.tzinfo is not None
    assert str(ts.tzinfo) == "America/New_York"


def test_handle_bar_swallows_unclosed(fake_clock):
    received = []
    async def cb(sym, series, ts):
        received.append(sym)
    bs = BarStream(MagicMock(), ["AAPL"], on_closed_bar=cb, clock=fake_clock.get_now)

    bar_start = datetime(2026, 5, 25, 14, 1, tzinfo=UTC)
    fake_clock.set(bar_start + timedelta(seconds=30))   # well before close+grace
    _run(bs._handle_bar(_fake_bar("AAPL", bar_start)))
    assert received == []
    # Heartbeat still updated even when bar swallowed
    assert bs.last_msg_at == bar_start + timedelta(seconds=30)


def test_heartbeat_alive_when_recent(fake_clock):
    bs = BarStream(MagicMock(), ["AAPL"], on_closed_bar=lambda *_: None,
                   clock=fake_clock.get_now)
    # No bar yet → not alive
    assert bs.is_alive() is False

    bar_start = datetime(2026, 5, 25, 14, 1, tzinfo=UTC)
    fake_clock.set(bar_start + timedelta(seconds=65))
    _run(bs._handle_bar(_fake_bar("AAPL", bar_start)))
    assert bs.is_alive() is True

    # Advance 9s — still within 10s window
    fake_clock.advance(9)
    assert bs.is_alive() is True

    # Advance another 2s — total 11s, outside window
    fake_clock.advance(2)
    assert bs.is_alive() is False


def test_handle_bar_continues_when_callback_raises(fake_clock):
    """An exception in the user callback must not crash the stream loop."""
    async def angry_cb(_sym, _series, _ts):
        raise RuntimeError("boom")
    bs = BarStream(MagicMock(), ["X"], on_closed_bar=angry_cb, clock=fake_clock.get_now)

    bar_start = datetime(2026, 5, 25, 14, 1, tzinfo=UTC)
    fake_clock.set(bar_start + timedelta(seconds=65))
    # Should not raise
    _run(bs._handle_bar(_fake_bar("X", bar_start)))
    # Heartbeat still recorded
    assert bs.last_msg_at is not None


def test_subscribe_wires_through_to_alpaca():
    stream_mock = MagicMock()
    bs = BarStream(stream_mock, ["aapl", "msft"], on_closed_bar=lambda *_: None)
    bs.subscribe()
    stream_mock.subscribe_bars.assert_called_once()
    args = stream_mock.subscribe_bars.call_args.args
    # First arg is the handler; remaining are symbols
    assert set(args[1:]) == {"AAPL", "MSFT"}

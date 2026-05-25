"""
Step-5 skeleton tests for state_machine.py.

These cover the *orchestration* gates (phase transitions, cooldown,
per-symbol state).  Step 6's gate-by-gate entry tests live in a
separate file (``test_entry_gates.py``).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from state_machine import BotPhase, Decision, StateMachine, SymbolState

NY = ZoneInfo("America/New_York")


def _at(h: int, m: int, s: int = 0) -> datetime:
    """Build a tz-aware NY datetime on a fixed weekday (Mon 2026-05-25)."""
    return datetime(2026, 5, 25, h, m, s, tzinfo=NY)


@pytest.fixture
def sm() -> StateMachine:
    return StateMachine()


# ── Phase transitions ───────────────────────────────────────────────────

@pytest.mark.parametrize("hms,expected", [
    ((7, 0),    BotPhase.PRE_MARKET),
    ((9, 29, 59), BotPhase.PRE_MARKET),
    ((9, 30, 0),  BotPhase.BUILDING_OR),
    ((9, 44, 59), BotPhase.BUILDING_OR),
    ((9, 45, 0),  BotPhase.HUNTING),
    ((11, 29, 59), BotPhase.HUNTING),
    ((11, 30, 0),  BotPhase.AFTERNOON),
    ((15, 29, 59), BotPhase.AFTERNOON),
    ((15, 30, 0),  BotPhase.WIND_DOWN),
    ((15, 54, 59), BotPhase.WIND_DOWN),
    ((15, 55, 0),  BotPhase.FORCED_FLAT),
    ((16, 0, 0),   BotPhase.CLOSED),
])
def test_phase_boundaries(sm, hms, expected):
    assert sm.update_phase(_at(*hms)) == expected


def test_update_phase_requires_tz_aware(sm):
    with pytest.raises(ValueError, match="tz-aware"):
        sm.update_phase(datetime(2026, 5, 25, 10, 0))  # naive


# ── Entry-time RVOL threshold ──────────────────────────────────────────

def test_morning_entries_use_rvol_bar_min(sm):
    th = sm.entry_rvol_threshold(_at(10, 0))
    assert th == sm.cfg.entry["rvol_bar_min"]  # 1.5


def test_afternoon_entries_require_higher_rvol(sm):
    th = sm.entry_rvol_threshold(_at(14, 0))
    assert th == sm.cfg.entry["afternoon_rvol_min"]  # 3.0


def test_no_entries_in_gap_window(sm):
    """11:30–13:00 — manage existing only, no new entries."""
    assert sm.entry_rvol_threshold(_at(12, 0)) is None
    assert sm.entries_allowed(_at(12, 0)) is False


def test_no_entries_after_afternoon_end(sm):
    """15:00 onwards: tightening / wind-down only."""
    assert sm.entry_rvol_threshold(_at(15, 0)) is None
    assert sm.entry_rvol_threshold(_at(15, 30)) is None
    assert sm.entry_rvol_threshold(_at(15, 55)) is None


def test_no_entries_before_morning_start(sm):
    assert sm.entry_rvol_threshold(_at(9, 30)) is None  # BUILDING_OR
    assert sm.entry_rvol_threshold(_at(9, 44, 59)) is None


def test_entries_open_exactly_at_morning_start(sm):
    assert sm.entry_rvol_threshold(_at(9, 45)) == sm.cfg.entry["rvol_bar_min"]


# ── Opening range storage ──────────────────────────────────────────────

def test_record_and_get_or(sm):
    sm.record_or("AAPL", high=205.50, low=204.00)
    h, l = sm.get_or("AAPL")
    assert (h, l) == (205.50, 204.00)


def test_get_or_returns_none_before_record(sm):
    assert sm.get_or("NEWSYM") == (None, None)


# ── Per-symbol state machine ───────────────────────────────────────────

def test_symbol_starts_flat(sm):
    assert sm.get_symbol_state("AAPL") == SymbolState.FLAT


def test_mark_position_opened(sm):
    sm.mark_position_opened("AAPL")
    assert sm.get_symbol_state("AAPL") == SymbolState.IN_POSITION


def test_mark_position_closed_triggers_cooldown(sm):
    closed_at = _at(10, 30)
    sm.mark_position_closed("AAPL", at=closed_at)
    assert sm.get_symbol_state("AAPL", now=closed_at) == SymbolState.COOLDOWN
    # 4 minutes later: still in cooldown
    assert sm.is_in_cooldown("AAPL", now=closed_at + timedelta(minutes=4))
    # 5 minutes later: decayed back to FLAT
    after = closed_at + timedelta(minutes=5)
    assert sm.get_symbol_state("AAPL", now=after) == SymbolState.FLAT
    assert not sm.is_in_cooldown("AAPL", now=after)


def test_cooldown_minutes_match_config(sm):
    assert sm._cooldown == timedelta(minutes=sm.cfg.risk["symbol_cooldown_minutes"])


def test_mark_position_closed_refuses_naive(sm):
    with pytest.raises(ValueError, match="tz-aware"):
        sm.mark_position_closed("AAPL", at=datetime(2026, 5, 25, 10, 0))


# ── evaluate() orchestration gates ────────────────────────────────────

def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz=NY),
    )


def test_evaluate_rejects_outside_entry_window(sm):
    d = sm.evaluate("AAPL", _empty_bars(), now=_at(12, 0))
    assert d.action == "skip"
    assert d.rejected_gate == "entries_allowed_by_phase"


def test_evaluate_rejects_when_symbol_in_position(sm):
    sm.record_or("AAPL", 100, 99)
    sm.mark_position_opened("AAPL")
    d = sm.evaluate("AAPL", _empty_bars(), now=_at(10, 0))
    assert d.action == "skip"
    assert d.rejected_gate == "symbol_flat"


def test_evaluate_rejects_when_symbol_in_cooldown(sm):
    sm.record_or("AAPL", 100, 99)
    sm.mark_position_closed("AAPL", at=_at(10, 0))
    d = sm.evaluate("AAPL", _empty_bars(), now=_at(10, 2))  # within cooldown
    assert d.action == "skip"
    assert d.rejected_gate == "symbol_flat"


def test_evaluate_rejects_when_or_not_recorded(sm):
    d = sm.evaluate("AAPL", _empty_bars(), now=_at(10, 0))
    assert d.action == "skip"
    assert d.rejected_gate == "opening_range_known"


def test_evaluate_rejects_when_insufficient_history(sm):
    """Once the orchestration gates clear, the indicator snapshot will
    fail with `sufficient_history` if there aren't enough bars warmed up
    for EMA20/MACD/etc."""
    sm.record_or("AAPL", 100, 99)
    d = sm.evaluate("AAPL", _empty_bars(), now=_at(10, 0))
    assert d.action == "skip"
    assert d.rejected_gate == "sufficient_history"
    assert d.gates["entries_allowed_by_phase"] is True
    assert d.gates["symbol_flat"] is True
    assert d.gates["opening_range_known"] is True
    assert d.gates["daily_loss_cap_ok"] is True

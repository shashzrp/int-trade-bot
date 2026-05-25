"""
Tests for exit_manager.py.

Covers:
  • Tranche splitting (33/33/34 from a 100-share position, edge cases)
  • T1 trigger, stop moves to BE
  • T2 trigger via ±2R, VWAP±2σ band, prior-day H/L (each independently)
  • T3 trigger via EMA9 break and via VWAP reclaim
  • Hard exits: kill switch, forced flat at 15:55, VWAP reversal with RVOL ≥ 1.5
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from exit_manager import ExitAction, ExitManager, PositionState


NY = ZoneInfo("America/New_York")


def _at(h: int, m: int, s: int = 0) -> datetime:
    return datetime(2026, 5, 25, h, m, s, tzinfo=NY)


@pytest.fixture
def em() -> ExitManager:
    return ExitManager()


def _open_long(em, *, entry=100.0, stop=99.50, qty=99, pdh=None, pdl=None) -> PositionState:
    return em.open_position(
        symbol="AAPL", side="long", entry_price=entry, qty=qty,
        stop_price=stop, prior_day_high=pdh, prior_day_low=pdl,
    )


def _open_short(em, *, entry=100.0, stop=100.50, qty=99, pdh=None, pdl=None) -> PositionState:
    return em.open_position(
        symbol="AAPL", side="short", entry_price=entry, qty=qty,
        stop_price=stop, prior_day_high=pdh, prior_day_low=pdl,
    )


def _bar(open_=100.0, high=100.0, low=100.0, close=100.0, volume=1000) -> dict:
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _ind(vwap=99.95, ema_fast=99.97, ema_slow=99.92, rsi=60.0,
         macd_hist=0.01, rvol_bar=2.0) -> dict:
    return {"vwap": vwap, "ema_fast": ema_fast, "ema_slow": ema_slow,
            "rsi": rsi, "macd_hist": macd_hist, "rvol_bar": rvol_bar}


# ── Tranche split ──────────────────────────────────────────────────────

def test_tranche_split_99_shares(em):
    """floor(99 * 0.333) = 32, not 33 — config fractions are slightly < 1/3,
    so the rounding leaves the runner with the leftover share."""
    pos = _open_long(em, qty=99)
    assert (pos.t1_qty, pos.t2_qty, pos.t3_qty) == (32, 32, 35)
    assert sum((pos.t1_qty, pos.t2_qty, pos.t3_qty)) == 99


def test_tranche_split_100_shares(em):
    pos = _open_long(em, qty=100)
    # floor(100*0.333)=33, floor(100*0.333)=33, remainder=34
    assert (pos.t1_qty, pos.t2_qty, pos.t3_qty) == (33, 33, 34)


def test_tranche_split_too_small_collapses(em):
    pos = _open_long(em, qty=2)
    assert (pos.t1_qty, pos.t2_qty, pos.t3_qty) == (0, 0, 2)


# ── T1 ─────────────────────────────────────────────────────────────────

def test_t1_long_triggers_at_plus_1R_and_moves_stop_to_be(em):
    pos = _open_long(em, entry=100.0, stop=99.50, qty=99)  # R=0.50, target=100.50
    bar = _bar(high=100.55, low=100.10, close=100.40)
    action = em.evaluate_bar("AAPL", bar=bar, indicators=_ind(), now=_at(10, 0))
    assert action is not None
    assert action.tranche == "T1"
    assert action.qty == 32
    assert action.new_stop == pytest.approx(100.0)
    assert action.is_terminal is False
    p = em.get("AAPL")
    assert p.t1_done and p.qty_remaining == 67
    assert p.stop_price == pytest.approx(100.0)


def test_t1_short_triggers_at_minus_1R(em):
    pos = _open_short(em, entry=100.0, stop=100.50, qty=99)  # R=0.50, target=99.50
    bar = _bar(high=99.90, low=99.45, close=99.60)
    action = em.evaluate_bar("AAPL", bar=bar, indicators=_ind(vwap=100.5), now=_at(10, 0))
    assert action is not None
    assert action.tranche == "T1"
    assert action.new_stop == pytest.approx(100.0)


def test_no_t1_below_target(em):
    _open_long(em, entry=100, stop=99.5, qty=99)
    action = em.evaluate_bar("AAPL", bar=_bar(high=100.30, close=100.20),
                             indicators=_ind(), now=_at(10, 0))
    assert action is None


# ── T2 — three independent triggers ────────────────────────────────────

def test_t2_via_2R(em):
    pos = _open_long(em, entry=100.0, stop=99.50, qty=99)
    # Drive T1 first
    em.evaluate_bar("AAPL", bar=_bar(high=100.55, close=100.40),
                    indicators=_ind(), now=_at(10, 0))
    # Then T2 at +2R = 101.00
    action = em.evaluate_bar("AAPL", bar=_bar(high=101.05, close=100.90),
                             indicators=_ind(ema_fast=100.50), now=_at(10, 5))
    assert action is not None
    assert action.tranche == "T2"
    assert action.qty == 32


def test_t2_via_vwap_band(em):
    pos = _open_long(em, entry=100.0, stop=99.50, qty=99)
    pos.vwap_band_upper_2sigma = 100.65
    em.evaluate_bar("AAPL", bar=_bar(high=100.55, close=100.40),
                    indicators=_ind(), now=_at(10, 0))  # T1
    # Price tags upper 2σ band even though it didn't reach +2R (101.00).
    action = em.evaluate_bar("AAPL", bar=_bar(high=100.70, close=100.60),
                             indicators=_ind(ema_fast=100.50), now=_at(10, 5))
    assert action is not None and action.tranche == "T2"


def test_t2_via_prior_day_high(em):
    pos = _open_long(em, entry=100.0, stop=99.50, qty=99, pdh=100.60)
    em.evaluate_bar("AAPL", bar=_bar(high=100.55, close=100.40),
                    indicators=_ind(), now=_at(10, 0))  # T1
    action = em.evaluate_bar("AAPL", bar=_bar(high=100.62, close=100.55),
                             indicators=_ind(ema_fast=100.50), now=_at(10, 5))
    assert action is not None and action.tranche == "T2"


def test_t2_for_short_via_prior_day_low(em):
    pos = _open_short(em, entry=100.0, stop=100.50, qty=99, pdl=99.40)
    em.evaluate_bar("AAPL", bar=_bar(high=99.85, low=99.45, close=99.60),
                    indicators=_ind(vwap=100.2), now=_at(10, 0))  # T1
    action = em.evaluate_bar("AAPL", bar=_bar(high=99.50, low=99.35, close=99.45),
                             indicators=_ind(vwap=100.0, ema_fast=99.50), now=_at(10, 5))
    assert action is not None and action.tranche == "T2"


# ── T3 ─────────────────────────────────────────────────────────────────

def test_t3_long_via_ema9_break(em):
    """Construct a T3 bar where close ≥ VWAP (no VWAP-reclaim hard exit)
    but close < EMA9 → T3 EMA-trail trigger fires."""
    _open_long(em, entry=100.0, stop=99.50, qty=99)
    em.evaluate_bar("AAPL", bar=_bar(high=100.55, close=100.40),
                    indicators=_ind(), now=_at(10, 0))   # T1
    em.evaluate_bar("AAPL", bar=_bar(high=101.05, close=100.90),
                    indicators=_ind(ema_fast=100.50), now=_at(10, 5))  # T2
    # close=100.45 > vwap=100.40 (no VWAP reclaim)  but close < ema_fast=100.60
    action = em.evaluate_bar("AAPL", bar=_bar(close=100.45),
                             indicators=_ind(vwap=100.40, ema_fast=100.60, rvol_bar=2.0),
                             now=_at(10, 10))
    assert action is not None and action.tranche == "T3"
    assert action.is_terminal is True
    assert em.get("AAPL") is None


def test_t3_long_via_vwap_reclaim_against_position(em):
    pos = _open_long(em, entry=100.0, stop=99.50, qty=99)
    em.evaluate_bar("AAPL", bar=_bar(high=100.55, close=100.40),
                    indicators=_ind(), now=_at(10, 0))   # T1
    em.evaluate_bar("AAPL", bar=_bar(high=101.05, close=100.90),
                    indicators=_ind(ema_fast=100.50), now=_at(10, 5))  # T2
    # Close back through VWAP — but with low RVOL so the hard-exit override
    # doesn't fire; T3 path should still trigger.
    action = em.evaluate_bar("AAPL", bar=_bar(close=100.30),
                             indicators=_ind(vwap=100.50, ema_fast=100.60, rvol_bar=0.5),
                             now=_at(10, 10))
    assert action is not None and action.tranche == "T3"


# ── Hard exits ─────────────────────────────────────────────────────────

def test_forced_flat_at_15_55(em):
    _open_long(em, qty=99)
    action = em.evaluate_bar("AAPL", bar=_bar(close=100.0),
                             indicators=_ind(), now=_at(15, 55))
    assert action is not None
    assert action.tranche == "HARD"
    assert action.reason == "forced_flat"
    assert action.qty == 99  # all remaining
    assert em.get("AAPL") is None


def test_kill_switch_active_closes_immediately():
    flag = {"on": True}
    em2 = ExitManager(kill_switch_active=lambda: flag["on"])
    em2.open_position(symbol="AAPL", side="long", entry_price=100, qty=99,
                      stop_price=99.5)
    action = em2.evaluate_bar("AAPL", bar=_bar(close=100.0),
                              indicators=_ind(), now=_at(10, 0))
    assert action is not None and action.tranche == "HARD"
    assert action.reason == "kill_switch"


def test_vwap_reversal_with_high_rvol_exits_long(em):
    _open_long(em, entry=100, stop=99.5, qty=99)
    # Close BELOW VWAP, high RVOL → premature reversal hard exit
    action = em.evaluate_bar(
        "AAPL", bar=_bar(close=99.80),
        indicators=_ind(vwap=99.90, rvol_bar=1.8),  # rvol ≥ 1.5
        now=_at(10, 0),
    )
    assert action is not None
    assert action.tranche == "HARD"
    assert action.reason == "vwap_reversal_high_rvol"


def test_vwap_reversal_with_low_rvol_does_NOT_hard_exit(em):
    _open_long(em, entry=100, stop=99.5, qty=99)
    # Close below VWAP but RVOL too low → no hard exit; T1 not hit either
    action = em.evaluate_bar(
        "AAPL", bar=_bar(close=99.80),
        indicators=_ind(vwap=99.90, rvol_bar=1.0),
        now=_at(10, 0),
    )
    assert action is None


def test_vwap_reversal_for_short_exits_on_close_above_vwap(em):
    _open_short(em, entry=100, stop=100.5, qty=99)
    action = em.evaluate_bar(
        "AAPL", bar=_bar(close=100.20),
        indicators=_ind(vwap=100.10, rvol_bar=2.0),
        now=_at(10, 0),
    )
    assert action is not None and action.reason == "vwap_reversal_high_rvol"


# ── Sequencing: T1 → T2 → T3 in order ─────────────────────────────────

def test_full_sequence_t1_t2_t3(em):
    pos = _open_long(em, entry=100.0, stop=99.50, qty=99)
    # T1 — qty 32 (floor(99*0.333)); remaining = 67
    a1 = em.evaluate_bar("AAPL", bar=_bar(high=100.55, close=100.40),
                         indicators=_ind(ema_fast=100.20), now=_at(10, 0))
    assert a1.tranche == "T1"
    p = em.get("AAPL")
    assert p.qty_remaining == 67 and p.t1_done

    # T2 — qty 32; remaining = 35 (the runner)
    a2 = em.evaluate_bar("AAPL", bar=_bar(high=101.05, close=100.90),
                         indicators=_ind(ema_fast=100.50), now=_at(10, 5))
    assert a2.tranche == "T2"
    p = em.get("AAPL")
    assert p.qty_remaining == 35 and p.t2_done

    # T3 — close < EMA9 but ≥ VWAP so EMA-trail path fires (not VWAP reclaim).
    a3 = em.evaluate_bar("AAPL", bar=_bar(close=100.45),
                         indicators=_ind(vwap=100.40, ema_fast=100.60, rvol_bar=2.0),
                         now=_at(10, 10))
    assert a3.tranche == "T3"
    assert a3.is_terminal
    assert em.get("AAPL") is None


def test_no_action_when_no_position(em):
    action = em.evaluate_bar("AAPL", bar=_bar(), indicators=_ind(), now=_at(10, 0))
    assert action is None

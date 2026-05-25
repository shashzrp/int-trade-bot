"""
Per-gate entry-rule tests for state_machine.evaluate().

Strategy: build a baseline DataFrame that passes ALL long gates, take a
copy, mutate ONE input, assert evaluate() now skips with the corresponding
rejected_gate. Then mirror for shorts.

This is the spec's "Tests: all 10 long entry gates individually" requirement.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from state_machine import (
    BotPhase,
    Decision,
    StateMachine,
    SymbolState,
    _is_bearish_reversal,
    _is_bullish_reversal,
)

NY = ZoneInfo("America/New_York")


# ── Synthetic bar builders ─────────────────────────────────────────────

def _session_bars_long_passing() -> pd.DataFrame:
    """Build a 1-min bar sequence designed to pass EVERY long entry gate
    on the final bar at 10:14 ET.

    Tuning notes:
      • OR: 09:30..09:44 (15 bars), prices oscillate 99.5–100.5.
      • Post-OR: 09:45..10:13 (29 bars), gentle uptrend with sinusoidal
        wobble so RSI lands inside 55–65 (not >70).
      • Breakout bar at 10:14: closes 100.60 (above OR_high 100.50) with
        a 50× volume spike so RVOL ≥ 1.5.
    """
    start = datetime(2026, 5, 25, 9, 30, tzinfo=NY)
    n_or, n_post = 15, 30
    n = n_or + n_post
    times = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(n)])

    closes = np.empty(n)
    closes[:n_or] = 100.0 + 0.5 * np.sin(np.linspace(0, np.pi * 2, n_or))
    # Gentler uptrend with bigger wobble — keeps RSI under 70.
    trend = np.linspace(100.10, 100.40, n_post - 1)
    wobble = 0.08 * np.sin(np.linspace(0, np.pi * 6, n_post - 1))
    closes[n_or:-1] = trend + wobble
    # Breakout bar: just above OR_high (100.50)
    closes[-1] = 100.55

    highs = closes + 0.04
    lows  = closes - 0.04
    opens = np.r_[closes[0], closes[:-1]]
    # OR volume 1000; consolidation 1500–2000; breakout bar 50×
    vols = np.r_[
        np.full(n_or, 1000.0),
        np.linspace(1500.0, 2000.0, n_post - 1),
        [50000.0],
    ]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=times,
    )


def _session_bars_short_passing() -> pd.DataFrame:
    """Mirror of the long passer — gentle downtrend after the OR."""
    start = datetime(2026, 5, 25, 9, 30, tzinfo=NY)
    n_or, n_post = 15, 30
    n = n_or + n_post
    times = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(n)])

    closes = np.empty(n)
    closes[:n_or] = 100.0 + 0.5 * np.sin(np.linspace(0, np.pi * 2, n_or))
    trend = np.linspace(99.90, 99.60, n_post - 1)
    wobble = 0.08 * np.sin(np.linspace(0, np.pi * 6, n_post - 1))
    closes[n_or:-1] = trend + wobble
    closes[-1] = 99.45  # just below OR_low (99.50)

    highs = closes + 0.04
    lows  = closes - 0.04
    opens = np.r_[closes[0], closes[:-1]]
    vols = np.r_[
        np.full(n_or, 1000.0),
        np.linspace(1500.0, 2000.0, n_post - 1),
        [50000.0],
    ]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=times,
    )


# ── Fixtures ───────────────────────────────────────────────────────────

NOW = datetime(2026, 5, 25, 10, 14, tzinfo=NY)  # inside HUNTING phase


@pytest.fixture
def sm_long():
    sm = StateMachine()
    sm.record_or("AAPL", high=100.50, low=99.50)
    return sm


@pytest.fixture
def sm_short():
    sm = StateMachine()
    sm.record_or("AAPL", high=100.50, low=99.50)
    return sm


# ── Baseline: the long passer actually passes everything ───────────────

def test_long_baseline_passes(sm_long):
    bars = _session_bars_long_passing()
    d = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d.action == "long", (
        f"baseline must pass; rejected_gate={d.rejected_gate!r} "
        f"gates={ {k:v for k,v in d.gates.items() if not v} }"
    )


def test_short_baseline_passes(sm_short):
    bars = _session_bars_short_passing()
    d = sm_short.evaluate("AAPL", bars, now=NOW)
    assert d.action == "short", (
        f"baseline must pass; rejected_gate={d.rejected_gate!r} "
        f"gates={ {k:v for k,v in d.gates.items() if not v} }"
    )


# ── LONG gates: each gate broken in isolation ──────────────────────────

def test_long_gate_trigger(sm_long):
    """Gate: price closes ≤ OR_high → trigger_long fails."""
    bars = _session_bars_long_passing()
    bars.iloc[-1, bars.columns.get_loc("close")] = 100.40  # under OR_high 100.50
    bars.iloc[-1, bars.columns.get_loc("high")]  = 100.45
    d = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d.action == "skip"
    assert d.gates["trigger_long"] is False


def test_long_gate_price_above_vwap(sm_long):
    """Gate: close ≤ VWAP → price_above_vwap fails."""
    bars = _session_bars_long_passing()
    # Drag the last bar's close well below VWAP
    bars.iloc[-1, bars.columns.get_loc("close")] = 98.0
    bars.iloc[-1, bars.columns.get_loc("low")]   = 97.9
    d = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d.action == "skip"
    assert d.gates["price_above_vwap"] is False


def test_long_gate_vwap_slope_positive(sm_long):
    """Gate: VWAP_now ≤ VWAP_5min_ago → slope gate fails.

    Force this by making volume HUGE on a single down-bar 5 min before
    the active bar so the running VWAP gets pulled UP at that moment and
    then drifts back down.
    """
    bars = _session_bars_long_passing()
    # Find the bar 5 minutes before the last, pump its price and volume.
    idx = bars.index[-6]
    bars.loc[idx, "close"] = 110.0
    bars.loc[idx, "high"]  = 110.05
    bars.loc[idx, "low"]   = 109.95
    bars.loc[idx, "volume"] = 5_000_000.0
    d = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d.action == "skip"
    assert d.gates["vwap_slope_positive"] is False


def test_long_gate_ema_alignment(sm_long):
    """Gate: EMA9 ≤ EMA20.  Force by inserting a deep dip immediately
    before the active bar so the fast EMA crosses below the slow EMA."""
    bars = _session_bars_long_passing()
    # Crush closes for the post-OR section so EMA9 drops below EMA20.
    crash_start = -10
    bars.iloc[crash_start:, bars.columns.get_loc("close")] = 95.0
    bars.iloc[crash_start:, bars.columns.get_loc("high")]  = 95.05
    bars.iloc[crash_start:, bars.columns.get_loc("low")]   = 94.95
    # Re-pop the LAST bar above OR_high so trigger_long evaluates separately
    bars.iloc[-1, bars.columns.get_loc("close")] = 100.60
    bars.iloc[-1, bars.columns.get_loc("high")]  = 100.65
    bars.iloc[-1, bars.columns.get_loc("low")]   = 95.0  # big lower wick

    d = sm_long.evaluate("AAPL", bars, now=NOW)
    # Should fail ema_alignment_long (even if it also fails others)
    assert d.gates.get("ema_alignment_long") is False


def test_long_gate_rsi_band(sm_long):
    """Gate: RSI not in (50, 70).  Easiest: drive an overheated rally so
    RSI exceeds 70."""
    bars = _session_bars_long_passing()
    # Make every post-OR bar a strong up-bar so RSI shoots > 70.
    n_post = 30
    bars.iloc[15:, bars.columns.get_loc("close")] = np.linspace(100.5, 110.0, n_post)
    bars.iloc[15:, bars.columns.get_loc("high")]  = bars.iloc[15:, bars.columns.get_loc("close")] + 0.05
    bars.iloc[15:, bars.columns.get_loc("low")]   = bars.iloc[15:, bars.columns.get_loc("close")] - 0.05
    d = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d.gates.get("rsi_in_band_long") is False


def test_long_gate_macd_hist_positive(sm_long):
    """Gate: MACD hist > 0.  Use a clean post-OR downtrend so MACD line
    sits well below its signal line. The trigger gate also fails here —
    that's fine; we just need `macd_hist_positive` recorded as False."""
    bars = _session_bars_long_passing()
    # Pure downtrend post-OR — no last-bar pop (that would create a bullish
    # MACD divergence and surprise us with a positive hist).
    bars.iloc[15:, bars.columns.get_loc("close")] = np.linspace(100.40, 98.00, 30)
    bars.iloc[15:, bars.columns.get_loc("high")] = bars.iloc[15:, bars.columns.get_loc("close")] + 0.04
    bars.iloc[15:, bars.columns.get_loc("low")]  = bars.iloc[15:, bars.columns.get_loc("close")] - 0.04
    d = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d.gates.get("macd_hist_positive") is False


def test_long_gate_rvol_bar_ok(sm_long):
    """Gate: rvol_bar ≥ threshold.  Shrink the last bar's volume."""
    bars = _session_bars_long_passing()
    bars.iloc[-1, bars.columns.get_loc("volume")] = 100.0  # tiny
    d = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d.gates["rvol_bar_ok"] is False


def test_long_gate_daily_loss_cap():
    """Gate: daily_loss_cap_ok callable returns False."""
    sm = StateMachine(daily_loss_cap_ok=lambda: False)
    sm.record_or("AAPL", 100.5, 99.5)
    d = sm.evaluate("AAPL", _session_bars_long_passing(), now=NOW)
    assert d.action == "skip"
    assert d.rejected_gate == "daily_loss_cap_ok"


def test_long_gate_symbol_flat_when_in_position(sm_long):
    """Gate 9: not already IN_POSITION."""
    sm_long.mark_position_opened("AAPL")
    d = sm_long.evaluate("AAPL", _session_bars_long_passing(), now=NOW)
    assert d.action == "skip"
    assert d.rejected_gate == "symbol_flat"


def test_long_gate_symbol_flat_when_in_cooldown(sm_long):
    """Gate 9: not in COOLDOWN."""
    sm_long.mark_position_closed("AAPL", at=NOW - timedelta(minutes=2))
    d = sm_long.evaluate("AAPL", _session_bars_long_passing(), now=NOW)
    assert d.action == "skip"
    assert d.rejected_gate == "symbol_flat"


def test_long_gate_entry_window():
    """Gate 1: outside HUNTING / afternoon-RVOL window."""
    sm = StateMachine()
    sm.record_or("AAPL", 100.5, 99.5)
    d = sm.evaluate("AAPL", _session_bars_long_passing(),
                    now=datetime(2026, 5, 25, 12, 30, tzinfo=NY))  # midday gap
    assert d.action == "skip"
    assert d.rejected_gate == "entries_allowed_by_phase"


# ── Afternoon RVOL threshold ───────────────────────────────────────────

def test_afternoon_requires_higher_rvol(sm_long):
    """Afternoon entries (13:00–15:00) need RVOL ≥ 3.0.  At 2.0 in afternoon
    the rvol_bar_ok gate should fail; at 2.0 in morning it would pass."""
    bars = _session_bars_long_passing()
    # Reduce the last bar's volume so rvol ≈ 2.0 (above 1.5 morning, below 3 afternoon)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 3500.0  # roughly 2.0× the rolling MA
    afternoon = datetime(2026, 5, 25, 14, 30, tzinfo=NY)
    d = sm_long.evaluate("AAPL", bars, now=afternoon)
    assert d.gates["rvol_bar_ok"] is False


# ── SHORT gates ────────────────────────────────────────────────────────

def test_short_gate_locate_required():
    """Short gate: locate_ok must be True (shortable AND easy_to_borrow)."""
    sm = StateMachine(locate_ok=lambda _sym: False)
    sm.record_or("AAPL", 100.5, 99.5)
    d = sm.evaluate("AAPL", _session_bars_short_passing(), now=NOW)
    assert d.action == "skip"
    assert d.gates.get("locate_ok") is False


def test_short_gate_trigger_short(sm_short):
    bars = _session_bars_short_passing()
    bars.iloc[-1, bars.columns.get_loc("close")] = 99.60  # above OR_low
    bars.iloc[-1, bars.columns.get_loc("low")]   = 99.55
    d = sm_short.evaluate("AAPL", bars, now=NOW)
    assert d.gates["trigger_short"] is False


def test_short_gate_price_below_vwap(sm_short):
    bars = _session_bars_short_passing()
    bars.iloc[-1, bars.columns.get_loc("close")] = 105.0  # well above VWAP
    bars.iloc[-1, bars.columns.get_loc("high")]  = 105.1
    d = sm_short.evaluate("AAPL", bars, now=NOW)
    assert d.gates["price_below_vwap"] is False


def test_short_gate_ema_alignment_short(sm_short):
    bars = _session_bars_short_passing()
    # Crush the early-trend section to make EMA9 > EMA20 (long alignment)
    bars.iloc[15:, bars.columns.get_loc("close")] = np.linspace(100.5, 102.0, 30)
    bars.iloc[15:, bars.columns.get_loc("high")] = bars.iloc[15:, bars.columns.get_loc("close")] + 0.05
    bars.iloc[15:, bars.columns.get_loc("low")]  = bars.iloc[15:, bars.columns.get_loc("close")] - 0.05
    bars.iloc[-1, bars.columns.get_loc("close")] = 99.40  # break OR_low to trigger
    bars.iloc[-1, bars.columns.get_loc("low")]   = 99.35
    bars.iloc[-1, bars.columns.get_loc("high")]  = 102.0
    d = sm_short.evaluate("AAPL", bars, now=NOW)
    assert d.gates.get("ema_alignment_short") is False


def test_short_gate_rsi_band_short(sm_short):
    """Drive RSI > 50 by extending the up-leg before crashing the last bar."""
    bars = _session_bars_short_passing()
    bars.iloc[15:-1, bars.columns.get_loc("close")] = np.linspace(100.5, 102.0, 29)
    bars.iloc[15:-1, bars.columns.get_loc("high")] = bars.iloc[15:-1, bars.columns.get_loc("close")] + 0.05
    bars.iloc[15:-1, bars.columns.get_loc("low")]  = bars.iloc[15:-1, bars.columns.get_loc("close")] - 0.05
    bars.iloc[-1, bars.columns.get_loc("close")] = 99.40
    bars.iloc[-1, bars.columns.get_loc("low")]   = 99.35
    d = sm_short.evaluate("AAPL", bars, now=NOW)
    assert d.gates.get("rsi_in_band_short") is False


def test_short_gate_macd_hist_negative(sm_short):
    """Clean uptrend post-OR → MACD hist stays positive → short gate fails."""
    bars = _session_bars_short_passing()
    bars.iloc[15:, bars.columns.get_loc("close")] = np.linspace(99.70, 102.00, 30)
    bars.iloc[15:, bars.columns.get_loc("high")] = bars.iloc[15:, bars.columns.get_loc("close")] + 0.04
    bars.iloc[15:, bars.columns.get_loc("low")]  = bars.iloc[15:, bars.columns.get_loc("close")] - 0.04
    d = sm_short.evaluate("AAPL", bars, now=NOW)
    assert d.gates.get("macd_hist_negative") is False


# ── Breakout-state: second close above OR requires retest pattern ─────

def test_long_first_breakout_works_then_second_requires_retest(sm_long):
    """First close above OR_high passes; the immediately-next bar (also
    above OR_high, no reversal pattern) must take the retest path and
    fail since the prior bar didn't dip back to OR_high / VWAP."""
    bars = _session_bars_long_passing()
    d1 = sm_long.evaluate("AAPL", bars, now=NOW)
    assert d1.action == "long"

    # Extend by one bar: still > OR_high, no reversal pattern, no retest.
    next_idx = bars.index[-1] + timedelta(minutes=1)
    next_bar = pd.DataFrame(
        {"open": [101.40], "high": [101.50], "low": [101.35],
         "close": [101.45], "volume": [5000.0]},
        index=[next_idx],
    )
    bars2 = pd.concat([bars, next_bar])
    d2 = sm_long.evaluate("AAPL", bars2, now=NOW + timedelta(minutes=1))
    # The trigger should fail — breakout state already True, no retest.
    assert d2.gates.get("trigger_long") is False


# ── Candlestick helpers ────────────────────────────────────────────────

def test_bullish_hammer():
    # Tiny body 100.02–100.04, long lower wick to 99.0, almost no upper wick.
    bar = {"open": 100.02, "high": 100.045, "low": 99.0, "close": 100.04,
           "prev_open": 100.0, "prev_close": 100.0}
    assert _is_bullish_reversal(bar)


def test_bullish_engulfing():
    bar = {"open": 99.5, "high": 100.6, "low": 99.4, "close": 100.5,
           "prev_open": 100.0, "prev_close": 99.6}
    assert _is_bullish_reversal(bar)


def test_no_bullish_signal_on_plain_up_bar():
    bar = {"open": 100.0, "high": 100.5, "low": 99.95, "close": 100.4,
           "prev_open": 99.0, "prev_close": 99.5}
    assert not _is_bullish_reversal(bar)


def test_bearish_shooting_star():
    bar = {"open": 100.0, "high": 101.0, "low": 99.97, "close": 99.98,
           "prev_open": 99.5, "prev_close": 100.0}
    assert _is_bearish_reversal(bar)


def test_bearish_engulfing():
    bar = {"open": 100.5, "high": 100.6, "low": 99.4, "close": 99.5,
           "prev_open": 99.6, "prev_close": 100.0}
    assert _is_bearish_reversal(bar)


# ── Indicator snapshot exposed on the Decision ────────────────────────

def test_indicator_snapshot_populated_on_pass(sm_long):
    d = sm_long.evaluate("AAPL", _session_bars_long_passing(), now=NOW)
    assert d.action == "long"
    keys = {"vwap", "vwap_5m_ago", "ema_fast", "ema_slow", "rsi",
            "macd_hist", "rvol_bar", "close", "high", "low", "open"}
    assert keys <= set(d.indicators.keys())

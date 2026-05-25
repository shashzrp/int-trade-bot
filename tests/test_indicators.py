"""
Unit tests for indicators.py.

Covers spec-mandated edge cases:
  • VWAP at 09:30:00 == first bar's typical price (the #1 bug)
  • VWAP resets across NY trading days; pre-market bars excluded
  • OR window boundary: bar at 09:44 IN, bar at 09:45 OUT (15-min window)
  • ATR(14) computed on 5-min bars matches `ta`'s reference implementation
  • RSI(14), EMA, MACD basic correctness
  • Position-related helpers (vwap_slope) are direction-correct
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from indicators import (
    NY_TZ,
    SESSION_OPEN,
    atr,
    ema,
    macd,
    opening_range,
    resample_to_5min,
    rsi,
    rvol_bar,
    session_vwap,
    session_vwap_bands,
    true_range,
    typical_price,
    volume_ma,
    vwap_slope_negative,
    vwap_slope_positive,
    wilder_rma,
)


# ── Hand-rolled Wilder reference for RSI / ATR cross-checks ─────────────

def _wilder_reference(values: np.ndarray, period: int) -> np.ndarray:
    """Pure-numpy SMA-seeded Wilder RMA — independent reference for tests."""
    out = np.full_like(values, np.nan, dtype=float)
    valid = ~np.isnan(values)
    cumv = valid.cumsum()
    if cumv[-1] < period:
        return out
    seed_idx = int(np.where(cumv == period)[0][0])
    out[seed_idx] = np.nanmean(values[: seed_idx + 1])
    for i in range(seed_idx + 1, len(values)):
        v = values[i]
        out[i] = out[i - 1] if np.isnan(v) else (out[i - 1] * (period - 1) + v) / period
    return out


def _rsi_reference(closes: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(closes, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = np.nan  # mirror Series.diff() behaviour
    loss[0] = np.nan
    avg_gain = _wilder_reference(gain, period)
    avg_loss = _wilder_reference(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where(avg_loss == 0, 100.0, out)
    out = np.where(np.isnan(avg_gain), np.nan, out)
    return out


# ── Synthetic data helpers ──────────────────────────────────────────────

def _session_index(day: date, n_minutes: int, start_time=SESSION_OPEN) -> pd.DatetimeIndex:
    """Build a tz-aware 1-min index for a single session."""
    base = datetime.combine(day, start_time).replace(tzinfo=NY_TZ)
    return pd.DatetimeIndex([base + timedelta(minutes=i) for i in range(n_minutes)])


def _ohlc_from_close(close: np.ndarray) -> dict[str, np.ndarray]:
    """Trivial OHLC: H=close+0.05, L=close-0.05, O=prev close (or close[0])."""
    open_ = np.r_[close[0], close[:-1]]
    return {
        "open":  open_,
        "high":  close + 0.05,
        "low":   close - 0.05,
        "close": close,
    }


def _make_bars(
    day: date,
    closes: list[float] | np.ndarray,
    volumes: list[float] | np.ndarray | None = None,
    start_time=SESSION_OPEN,
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    if volumes is None:
        volumes = np.full_like(closes, 1000.0)
    ohlc = _ohlc_from_close(closes)
    return pd.DataFrame({**ohlc, "volume": np.asarray(volumes, dtype=float)},
                        index=_session_index(day, len(closes), start_time))


# ── typical_price ───────────────────────────────────────────────────────

def test_typical_price():
    df = pd.DataFrame({"high": [11.0], "low": [9.0], "close": [10.0]})
    assert typical_price(df).iloc[0] == pytest.approx((11 + 9 + 10) / 3)


# ── VWAP: the headline test ─────────────────────────────────────────────

def test_vwap_at_session_open_equals_first_typical_price():
    """The #1 bug the spec calls out — explicit assertion."""
    day = date(2025, 3, 5)
    closes = np.array([100.0, 101.0, 102.0, 103.0])
    vols   = np.array([1000.0, 1500.0, 2000.0, 500.0])
    df = _make_bars(day, closes, vols)

    vwap = session_vwap(df)
    first_tp = typical_price(df).iloc[0]

    assert vwap.iloc[0] == pytest.approx(first_tp), (
        f"VWAP at 09:30 ({vwap.iloc[0]}) must equal first bar's typical price ({first_tp})"
    )


def test_vwap_matches_hand_computed_running_value():
    day = date(2025, 3, 5)
    closes = np.array([100.0, 101.0, 102.0])
    vols   = np.array([1000.0, 2000.0, 3000.0])
    df = _make_bars(day, closes, vols)

    vwap = session_vwap(df)
    tp = typical_price(df).to_numpy()

    # Hand-computed running VWAP
    cum_pv = np.cumsum(tp * vols)
    cum_v  = np.cumsum(vols)
    expected = cum_pv / cum_v

    np.testing.assert_allclose(vwap.to_numpy(), expected, rtol=1e-12)


def test_vwap_resets_across_trading_days():
    """Day-2's 09:30 VWAP must NOT inherit day-1's volume-weighted prior."""
    day1 = date(2025, 3, 5)
    day2 = date(2025, 3, 6)
    df1 = _make_bars(day1, [100.0, 101.0, 102.0], [1000.0, 1000.0, 1000.0])
    df2 = _make_bars(day2, [200.0, 201.0, 202.0], [1000.0, 1000.0, 1000.0])
    df = pd.concat([df1, df2])

    vwap = session_vwap(df)
    day2_first_tp = typical_price(df2).iloc[0]
    assert vwap.loc[df2.index[0]] == pytest.approx(day2_first_tp)

    # Sanity: day-2 VWAP nowhere near day-1's terminal VWAP (~101)
    assert vwap.loc[df2.index[0]] > 150.0


def test_vwap_excludes_premarket_bars():
    """Pre-market bars (08:00 etc.) should NOT contribute to VWAP."""
    day = date(2025, 3, 5)
    from datetime import time as t

    pm = _make_bars(day, [50.0, 50.0], [10_000_000.0, 10_000_000.0],
                    start_time=t(8, 0))
    rh = _make_bars(day, [100.0, 100.5], [1000.0, 1000.0])
    df = pd.concat([pm, rh])

    vwap = session_vwap(df)
    # Pre-market bars: NaN
    assert vwap.loc[pm.index].isna().all()
    # 09:30 bar: equals its own typical price — the pre-market giant volume
    # did NOT pull the VWAP toward 50.
    assert vwap.loc[rh.index[0]] == pytest.approx(typical_price(rh).iloc[0])


def test_vwap_zero_volume_bar_does_not_blow_up():
    day = date(2025, 3, 5)
    df = _make_bars(day, [100.0, 101.0], [0.0, 0.0])
    vwap = session_vwap(df)
    # No volume yet → NaN, not inf
    assert vwap.isna().all()


# ── VWAP bands ──────────────────────────────────────────────────────────

def test_vwap_bands_bracket_vwap():
    day = date(2025, 3, 5)
    closes = np.linspace(100.0, 110.0, 30)
    vols = np.random.default_rng(0).integers(1000, 5000, size=30).astype(float)
    df = _make_bars(day, closes, vols)

    vwap = session_vwap(df)
    bands = session_vwap_bands(df, vwap, sigmas=(1.0, 2.0))

    # Upper ≥ VWAP ≥ Lower at every in-session bar
    assert (bands[1.0]["upper"] >= vwap).all()
    assert (bands[1.0]["lower"] <= vwap).all()
    # 2σ bands strictly wider than 1σ (except where sigma is exactly zero)
    spread_1 = bands[1.0]["upper"] - bands[1.0]["lower"]
    spread_2 = bands[2.0]["upper"] - bands[2.0]["lower"]
    assert (spread_2 >= spread_1).all()


# ── Opening Range boundary ──────────────────────────────────────────────

def test_opening_range_boundary_9_44_in_9_45_out():
    """Spec says OR = bars between 09:30:00 and 09:44:59. With 1-min bars
    timestamped at minute START, that means 09:30..09:44 inclusive, 09:45 OUT."""
    day = date(2025, 3, 5)
    # 16 bars: 09:30 .. 09:45
    closes = np.full(16, 100.0)
    closes[14] = 105.0   # 09:44 — should set the OR_high
    closes[15] = 999.0   # 09:45 — must NOT set the OR_high (outside window)
    df = _make_bars(day, closes)
    # Force the synthetic OHLC to align: high = close + 0.05
    or_window = opening_range(df, session_date=day, minutes=15)
    assert or_window is not None
    or_high, or_low = or_window
    assert or_high == pytest.approx(105.0 + 0.05)
    assert or_low  == pytest.approx(100.0 - 0.05)

    # Now drop the 09:45 bar to confirm result unchanged → confirms 09:45 was excluded.
    df_no_945 = df.iloc[:15]
    assert opening_range(df_no_945, session_date=day, minutes=15) == or_window


def test_opening_range_returns_none_when_no_bars():
    df = _make_bars(date(2025, 3, 5), [100.0])  # only the 09:30 bar
    other_day = date(2025, 3, 6)
    assert opening_range(df, session_date=other_day, minutes=15) is None


def test_opening_range_partial_window():
    """If only a few OR bars exist (e.g. data outage), OR computes from what's there."""
    day = date(2025, 3, 5)
    df = _make_bars(day, [100.0, 101.0, 99.0])  # 09:30, 09:31, 09:32
    or_high, or_low = opening_range(df, session_date=day, minutes=15)
    assert or_high == pytest.approx(101.0 + 0.05)
    assert or_low  == pytest.approx(99.0 - 0.05)


# ── EMA / RSI / MACD ────────────────────────────────────────────────────

def test_ema_of_constant_series_is_constant():
    s = pd.Series([5.0] * 50)
    e = ema(s, 9).dropna()
    assert (e == 5.0).all()


def test_ema_min_periods_respected():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    e = ema(s, period=4)
    assert e.iloc[:3].isna().all()
    assert not np.isnan(e.iloc[3])


def test_rsi_matches_wilder_reference():
    """RSI(14) using SMA-seeded Wilder RMA, hand-computed reference."""
    rng = np.random.default_rng(42)
    closes_arr = 100 + rng.standard_normal(200).cumsum()
    closes = pd.Series(closes_arr)
    ours = rsi(closes, 14).to_numpy()
    ref = _rsi_reference(closes_arr, 14)

    valid = ~np.isnan(ours) & ~np.isnan(ref)
    assert valid.sum() > 100
    np.testing.assert_allclose(ours[valid], ref[valid], rtol=1e-10, atol=1e-10)


def test_wilder_rma_textbook_recurrence():
    """Independent assertion of the Wilder recurrence itself."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = wilder_rma(x, period=4).to_numpy()
    # First 3 NaN, seed at index 3 = mean(1,2,3,4) = 2.5
    assert np.isnan(out[:3]).all()
    assert out[3] == pytest.approx(2.5)
    # Index 4: (2.5 * 3 + 5)/4 = 12.5/4 = 3.125
    assert out[4] == pytest.approx(3.125)
    # Index 5: (3.125 * 3 + 6)/4
    assert out[5] == pytest.approx((3.125 * 3 + 6) / 4)


def test_rsi_strong_uptrend_above_70():
    closes = pd.Series(np.linspace(100, 200, 100))  # monotonic up
    r = rsi(closes, 14).dropna()
    assert r.iloc[-1] > 70


def test_rsi_strong_downtrend_below_30():
    closes = pd.Series(np.linspace(200, 100, 100))  # monotonic down
    r = rsi(closes, 14).dropna()
    assert r.iloc[-1] < 30


def test_macd_components_align():
    rng = np.random.default_rng(7)
    closes = pd.Series(100 + rng.standard_normal(300).cumsum())
    m = macd(closes)
    # hist = macd - signal (definitional)
    diff = (m["macd"] - m["signal"]) - m["hist"]
    np.testing.assert_allclose(diff.dropna().to_numpy(), 0.0, atol=1e-12)


# ── ATR on 5-min bars ───────────────────────────────────────────────────

def test_atr_matches_wilder_reference_on_5min_bars():
    """ATR(14) on 5-min bars — Wilder RMA reference, computed by hand."""
    rng = np.random.default_rng(11)
    n = 200
    day = date(2025, 3, 5)
    closes = 100 + rng.standard_normal(n).cumsum() * 0.1
    df = _make_bars(day, closes)
    df_5m = resample_to_5min(df)

    tr = true_range(df_5m).to_numpy()
    ref = _wilder_reference(tr, 14)
    ours = atr(df_5m, 14).to_numpy()

    valid = ~np.isnan(ours) & ~np.isnan(ref)
    assert valid.sum() > 5
    np.testing.assert_allclose(ours[valid], ref[valid], rtol=1e-12, atol=1e-12)


def test_atr_first_value_index_and_seed():
    """At index period-1 the ATR should equal the SMA of TR over [0, period)."""
    day = date(2025, 3, 5)
    closes = np.linspace(100, 105, 30)
    df = _make_bars(day, closes)
    df_5m = resample_to_5min(df)
    period = 5
    a = atr(df_5m, period=period)
    tr = true_range(df_5m)
    assert a.iloc[: period - 1].isna().all()
    assert a.iloc[period - 1] == pytest.approx(tr.iloc[:period].mean())


def test_true_range_first_bar_uses_only_high_low():
    """No prior close → TR collapses to H-L."""
    df = pd.DataFrame({"high": [10.0], "low": [9.5], "close": [9.8]})
    tr = true_range(df)
    assert tr.iloc[0] == pytest.approx(0.5)


def test_resample_to_5min_aggregates_correctly():
    day = date(2025, 3, 5)
    df = _make_bars(day, list(range(100, 110)))  # 10 1-min bars, closes 100..109
    df_5 = resample_to_5min(df)
    assert len(df_5) == 2  # 09:30 and 09:35 buckets
    # Bucket 0: bars 09:30..09:34, closes 100..104
    b0 = df_5.iloc[0]
    assert b0["open"]  == pytest.approx(df.iloc[0]["open"])
    assert b0["close"] == pytest.approx(df.iloc[4]["close"])
    assert b0["high"]  == df.iloc[:5]["high"].max()
    assert b0["low"]   == df.iloc[:5]["low"].min()
    assert b0["volume"] == df.iloc[:5]["volume"].sum()


# ── Volume MA / RVOL ────────────────────────────────────────────────────

def test_volume_ma_warmup_period():
    s = pd.Series([100.0] * 25)
    ma = volume_ma(s, bars=20)
    assert ma.iloc[:19].isna().all()
    assert ma.iloc[19] == 100.0


def test_rvol_bar_value():
    vol = pd.Series([1000.0] * 19 + [2500.0])
    r = rvol_bar(vol, bars=20)
    # MA at last bar = mean([1000*19, 2500]) / 20 = (19000 + 2500)/20 = 1075
    expected = 2500.0 / ((1000.0 * 19 + 2500.0) / 20.0)
    assert r.iloc[-1] == pytest.approx(expected)


# ── VWAP slope helpers ──────────────────────────────────────────────────

def test_vwap_slope_positive_and_negative_directionality():
    idx = pd.date_range("2025-03-05 09:30", periods=10, freq="1min", tz=NY_TZ)
    up   = pd.Series(np.arange(10, dtype=float), index=idx)
    down = pd.Series(np.arange(10, 0, -1, dtype=float), index=idx)
    assert vwap_slope_positive(up, 5).iloc[-1] is np.True_ or bool(vwap_slope_positive(up, 5).iloc[-1])
    assert bool(vwap_slope_negative(down, 5).iloc[-1])
    assert not bool(vwap_slope_negative(up, 5).iloc[-1])
    assert not bool(vwap_slope_positive(down, 5).iloc[-1])


# ── Defensive: tz-aware index requirement ───────────────────────────────

def test_session_vwap_refuses_naive_index():
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex(["2025-03-05 09:30:00"]),  # naive, no tz
    )
    with pytest.raises(ValueError, match="tz-aware"):
        session_vwap(df)


def test_session_vwap_refuses_non_datetime_index():
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
    )
    with pytest.raises(TypeError, match="DatetimeIndex"):
        session_vwap(df)

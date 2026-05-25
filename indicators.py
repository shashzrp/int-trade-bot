"""
Vectorized indicator engine.

Design rules:
  • All inputs are pandas DataFrames or Series with a **tz-aware** index in
    America/New_York. Bars carry the **start** timestamp of the minute they
    represent (Alpaca convention).
  • Everything is pure / functional — no hidden state, no caching. Live mode
    re-evaluates on each closed bar by passing the rolling window in.
  • The VWAP path resets at each NY trading-day's 09:30. Pre-market bars do
    NOT contribute. Wrong reset is the #1 bug the spec calls out — see
    `session_vwap` and the matching test.

Hand-rolled (not delegated to `ta`):
  • session_vwap — needs a session-aware reset `ta` does not provide.
  • vwap_bands  — volume-weighted variance, parallel-axis form.
  • opening_range — boundary semantics matter (see spec §indicators).
  • rsi, atr — classical **Wilder's RMA** (SMA seed for the first `period`
    observations, then  rma[t] = (rma[t-1]·(period-1) + x[t]) / period).
    This matches TradingView's `rma()` and the values most pro platforms
    (ThinkOrSwim, NinjaTrader) display. Note that the `ta` library uses
    pandas `ewm(adjust=False)` for RSI which differs in the warmup region;
    we cross-check against hand-computed Wilder values in tests, not `ta`.
"""
from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


# ── Helpers ──────────────────────────────────────────────────────────────

def _require_ny_index(df: pd.DataFrame | pd.Series) -> None:
    """Bars must carry a tz-aware index. We do NOT silently convert naive
    timestamps — the bot's contract is that ingest already localized to NY."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(idx).__name__}.")
    if idx.tz is None:
        raise ValueError(
            "Bar index must be tz-aware (America/New_York). "
            "Convert at ingest, not here."
        )


def typical_price(df: pd.DataFrame) -> pd.Series:
    """(H + L + C) / 3 — the price used by VWAP."""
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _session_mask(
    df: pd.DataFrame | pd.Series,
    *,
    start: time = SESSION_OPEN,
    end: time = SESSION_CLOSE,
) -> pd.Series:
    """True for bars whose timestamp falls in [start, end) on each NY day."""
    _require_ny_index(df)
    local = df.index.tz_convert(NY_TZ)
    mask = (local.time >= start) & (local.time < end)
    return pd.Series(mask, index=df.index)


# ── VWAP & bands ─────────────────────────────────────────────────────────

def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP. Resets each NY trading day at 09:30 ET.

    Pre-market bars contribute NaN (they are not part of the regular-hours
    VWAP). Post-close bars (>= 16:00) also NaN.

    Critical invariant (test asserted): at the bar timestamped 09:30:00,
    VWAP equals that bar's typical price.
    """
    _require_ny_index(df)
    tp = typical_price(df)
    vol = df["volume"].astype(float)
    in_sess = _session_mask(df)

    pv = (tp * vol).where(in_sess, 0.0)
    v  = vol.where(in_sess, 0.0)

    # Group by NY calendar date so cumsum restarts every morning.
    ny_date = df.index.tz_convert(NY_TZ).date
    cum_pv = pv.groupby(ny_date).cumsum()
    cum_v  = v.groupby(ny_date).cumsum()

    # Where volume is zero (e.g. first bar with zero volume, or out-of-session),
    # surface NaN rather than divide-by-zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = cum_pv / cum_v.replace(0.0, np.nan)

    return vwap.where(in_sess)


def session_vwap_bands(
    df: pd.DataFrame,
    vwap: pd.Series | None = None,
    sigmas: tuple[float, ...] = (1.0, 2.0),
) -> dict[float, dict[str, pd.Series]]:
    """Volume-weighted std bands around VWAP.

    Uses the parallel-axis form  Var = E[X²] − (E[X])²  with weights = volume:
        σ²(t) = Σᵢ≤ₜ volᵢ·tpᵢ² / Σᵢ≤ₜ volᵢ   −   VWAP(t)²
    Returns ``{1.0: {"upper": …, "lower": …}, 2.0: {…}}``.
    """
    _require_ny_index(df)
    if vwap is None:
        vwap = session_vwap(df)

    tp = typical_price(df)
    vol = df["volume"].astype(float)
    in_sess = _session_mask(df)

    pv2 = (tp * tp * vol).where(in_sess, 0.0)
    v   = vol.where(in_sess, 0.0)

    ny_date = df.index.tz_convert(NY_TZ).date
    cum_pv2 = pv2.groupby(ny_date).cumsum()
    cum_v   = v.groupby(ny_date).cumsum()

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sq = cum_pv2 / cum_v.replace(0.0, np.nan)

    variance = (mean_sq - vwap ** 2).clip(lower=0.0)  # crush tiny negatives
    sigma = np.sqrt(variance)
    return {
        float(s): {"upper": vwap + s * sigma, "lower": vwap - s * sigma}
        for s in sigmas
    }


# ── Opening Range ────────────────────────────────────────────────────────

def opening_range(
    df: pd.DataFrame,
    *,
    session_date: date,
    minutes: int = 15,
) -> tuple[float, float] | None:
    """High and low across the first `minutes` of the regular session.

    Spec defines the OR as bars whose start timestamp lies in
    [09:30:00, 09:30:00 + minutes).  At minutes=15 this is
    [09:30:00, 09:45:00) — i.e. the 09:44 bar is INCLUDED, the 09:45 bar is
    NOT. Tested in ``test_indicators.py::test_opening_range_boundary``.

    Returns None if no bars fall in the window (e.g. half-day, data outage).
    """
    _require_ny_index(df)
    open_dt = pd.Timestamp.combine(session_date, SESSION_OPEN).tz_localize(NY_TZ)
    end_dt  = open_dt + pd.Timedelta(minutes=minutes)
    window = df.loc[(df.index >= open_dt) & (df.index < end_dt)]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min())


# ── Wilder's RMA (Running Moving Average) ────────────────────────────────

def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Classical Wilder's smoothing — the basis for RSI and ATR.

    Pads NaN until `period` non-NaN observations exist; the seed at that
    index is the SMA of those observations.  Subsequent values use the
    recurrence  rma[t] = (rma[t-1]·(period-1) + x[t]) / period.
    """
    arr = np.asarray(series, dtype=float)
    out = np.full_like(arr, np.nan)
    if arr.size < period:
        return pd.Series(out, index=series.index, name=series.name)

    valid = ~np.isnan(arr)
    cumv = valid.cumsum()
    seeds = np.where(cumv == period)[0]
    if seeds.size == 0:
        return pd.Series(out, index=series.index, name=series.name)
    seed_idx = int(seeds[0])

    out[seed_idx] = float(arr[: seed_idx + 1][valid[: seed_idx + 1]].mean())
    inv_p = 1.0 / period
    weight = (period - 1) * inv_p
    for i in range(seed_idx + 1, arr.size):
        val = arr[i]
        out[i] = out[i - 1] if np.isnan(val) else out[i - 1] * weight + val * inv_p
    return pd.Series(out, index=series.index, name=series.name)


# ── EMA / RSI / MACD ─────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using classical SMA-seeded RMA smoothing.

    The first bar has no prior close so `delta[0]` is NaN — we keep that
    NaN (via ``clip``) so the seed period spans the first `period` REAL
    price changes (indices 1..period), matching textbook Wilder."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)              # NaN at idx 0 preserved
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # No down-moves yet → loss==0 → RSI = 100 by convention.
    out = out.where(avg_loss != 0.0, 100.0)
    return out.where(avg_gain.notna())


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Classic MACD on 1-min close.  Returns {macd, signal, hist}."""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


# ── True Range / ATR ─────────────────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    """TR = max(H-L, |H − prev_C|, |L − prev_C|)."""
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"]  - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(period) — Wilder smoothing of True Range using SMA-seeded RMA.

    Spec requires this on **5-minute** bars (not 1-min).  The caller is
    responsible for passing 5-min OHLC — see ``resample_to_5min``.
    """
    return wilder_rma(true_range(df), period)


def resample_to_5min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min OHLCV into 5-min bars (left-labelled, left-closed)."""
    _require_ny_index(df_1min)
    rules = {"open": "first", "high": "max", "low": "min", "close": "last",
             "volume": "sum"}
    return (
        df_1min.resample("5min", label="left", closed="left")
        .agg(rules)
        .dropna(subset=["open", "high", "low", "close"])
    )


# ── Volume / RVOL ────────────────────────────────────────────────────────

def volume_ma(volume: pd.Series, bars: int = 20) -> pd.Series:
    """20-bar SMA of bar volume."""
    return volume.rolling(window=bars, min_periods=bars).mean()


def rvol_bar(volume: pd.Series, bars: int = 20) -> pd.Series:
    """RVOL_bar = volume / volume_ma(volume, bars).

    NaN until `bars` history exists. NaN where the MA is exactly zero (rare
    on liquid names but possible on pre-market / halt edge cases)."""
    ma = volume_ma(volume, bars)
    return volume / ma.replace(0.0, np.nan)


# ── VWAP slope helper (used by entry gates) ──────────────────────────────

def vwap_slope_positive(vwap: pd.Series, lookback_minutes: int = 5) -> pd.Series:
    """True where VWAP_now > VWAP_{lookback_minutes ago}.  NaN-aware."""
    return vwap > vwap.shift(lookback_minutes)


def vwap_slope_negative(vwap: pd.Series, lookback_minutes: int = 5) -> pd.Series:
    return vwap < vwap.shift(lookback_minutes)

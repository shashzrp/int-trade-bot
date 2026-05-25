"""
Historical 1-min bar loader.

Fetches from Alpaca's StockHistoricalDataClient in 30-day windows (to avoid
single-request size limits) and returns a ``{symbol: DataFrame}`` keyed by
NY-tz timestamps — the same shape the live stream produces, so the rest of
the backtest can be data-source-agnostic.

For offline / CI runs, a sidecar CSV cache directory keeps roundtrips down.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")


def load_1min_bars(
    data_client: StockHistoricalDataClient,
    symbols: list[str],
    *,
    start: date,
    end: date,
    feed: str = "iex",
    cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load 1-min bars for ``symbols`` covering ``[start, end]`` inclusive.

    Bars are localized to NY time, indexed on the bar start timestamp,
    with columns ``open, high, low, close, volume``.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym = sym.upper()
        if cache_dir is not None:
            cached = _read_cache(cache_dir, sym, start, end)
            if cached is not None:
                out[sym] = cached
                continue

        df = _fetch_in_windows(data_client, sym, start, end, feed=feed)
        if df is None or df.empty:
            logger.warning("no bars returned for %s on [%s, %s]", sym, start, end)
            continue
        out[sym] = df
        if cache_dir is not None:
            _write_cache(cache_dir, sym, start, end, df)
    return out


def _fetch_in_windows(
    dc: StockHistoricalDataClient,
    sym: str,
    start: date,
    end: date,
    *,
    feed: str,
    window_days: int = 30,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=window_days - 1), end)
        req = StockBarsRequest(
            symbol_or_symbols=[sym],
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
            start=datetime.combine(cur, time(4, 0), tzinfo=NY_TZ),
            end=datetime.combine(chunk_end + timedelta(days=1), time(0, 0), tzinfo=NY_TZ),
            feed=feed,
        )
        barset = dc.get_stock_bars(req)
        data = getattr(barset, "data", None) or {}
        bars = data.get(sym, [])
        if bars:
            df = pd.DataFrame(
                [{"timestamp": b.timestamp, "open": b.open, "high": b.high,
                  "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY_TZ)
            df = df.set_index("timestamp").sort_index()
            chunks.append(df)
        cur = chunk_end + timedelta(days=1)
    return pd.concat(chunks) if chunks else pd.DataFrame()


def _cache_path(cache_dir: Path, sym: str, start: date, end: date) -> Path:
    return cache_dir / f"{sym}_{start.isoformat()}_{end.isoformat()}.parquet"


def _read_cache(cache_dir: Path, sym: str, start: date, end: date) -> pd.DataFrame | None:
    p = _cache_path(cache_dir, sym, start, end)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _write_cache(cache_dir: Path, sym: str, start: date, end: date, df: pd.DataFrame) -> None:
    p = _cache_path(cache_dir, sym, start, end)
    try:
        df.to_parquet(p)
    except Exception as exc:
        logger.warning("cache write failed for %s: %s", sym, exc)

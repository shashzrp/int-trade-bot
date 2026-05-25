"""
Unit tests for the pre-market scanner.

We never touch the real Alpaca API here — the data client and trading
client are mocked; the scanner's per-symbol scoring logic is exercised
against synthetic daily/pre-market bars whose shape mirrors what
``StockHistoricalDataClient`` returns.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config import StrategyConfig
from scanner import (
    NY_TZ,
    Candidate,
    ScanResult,
    UniverseScanner,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> StrategyConfig:
    return StrategyConfig.from_yaml()


@pytest.fixture
def asof() -> datetime:
    # Monday 2026-05-25, 09:15 ET — solid weekday, no DST quirks.
    return datetime(2026, 5, 25, 9, 15, tzinfo=NY_TZ)


# ── Synthetic bar helpers ───────────────────────────────────────────────

def _daily_frame(prior_close: float, *, daily_atr_pct: float = 3.0,
                  adv: float = 5_000_000, n_days: int = 22) -> pd.DataFrame:
    """Build a daily frame ending one day before `asof`. The LAST row's
    close becomes the prior_close used by gap calculation."""
    end = date(2026, 5, 22)  # Friday before asof Monday
    dates = pd.bdate_range(end=end, periods=n_days, tz=NY_TZ)
    # Construct OHLC so daily ATR(14) ≈ daily_atr_pct of price.
    rng = prior_close * daily_atr_pct / 100.0
    closes = np.full(n_days, prior_close, dtype=float)
    highs  = closes + rng / 2
    lows   = closes - rng / 2
    opens  = closes
    vols   = np.full(n_days, adv, dtype=float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=dates,
    )


def _premkt_frame(total_volume: float) -> pd.DataFrame:
    """Synthetic pre-market 1-min bars on 2026-05-25 from 04:00 to 09:29."""
    start = datetime(2026, 5, 25, 4, 0, tzinfo=NY_TZ)
    idx = pd.date_range(start=start, periods=10, freq="1min")  # arbitrary 10 bars
    per_bar = total_volume / len(idx)
    return pd.DataFrame(
        {"open": [100.0] * 10, "high": [101.0] * 10, "low": [99.0] * 10,
         "close": [100.5] * 10, "volume": [per_bar] * 10},
        index=idx,
    )


def _quote(bid: float, ask: float):
    return SimpleNamespace(bid_price=bid, ask_price=ask)


def _make_scanner(cfg, *,
                  symbols: list[str],
                  daily: dict[str, pd.DataFrame],
                  premkt: dict[str, pd.DataFrame],
                  quotes: dict[str, SimpleNamespace],
                  earnings_today: set[str] | None = None,
                  asset_overrides: dict[str, dict] | None = None):
    """Build a UniverseScanner with internals stubbed for deterministic tests."""
    tc = MagicMock()
    dc = MagicMock()

    asset_overrides = asset_overrides or {}
    def fake_get_asset(sym):
        defaults = {"symbol": sym, "shortable": True, "easy_to_borrow": True}
        defaults.update(asset_overrides.get(sym, {}))
        return SimpleNamespace(**defaults)
    tc.get_asset.side_effect = fake_get_asset

    scn = UniverseScanner(
        tc, dc, cfg,
        earnings_today_provider=(lambda d: earnings_today) if earnings_today is not None else None,
        candidate_symbols=symbols,
    )
    # Stub the cheap-filter stage so all supplied symbols pass.
    scn._candidate_universe = lambda result: symbols
    # Stub the network calls.
    scn._fetch_daily_bars = lambda syms, day: daily
    scn._fetch_premarket_bars = lambda syms, day: premkt
    # Quote response: a dict is fine — `_quote()` handles `__getitem__`.
    scn._fetch_latest_quotes = lambda syms: quotes
    return scn


# ── Happy-path: top-N sort + cap ────────────────────────────────────────

def test_watchlist_sorted_by_rvol_desc(cfg, asof):
    # Three symbols, all passing thresholds, with different RVOLs
    syms = ["AAA", "BBB", "CCC"]
    daily = {s: _daily_frame(prior_close=100.0, adv=2_000_000) for s in syms}
    # PM volumes give RVOL: 0.05, 0.10, 0.075 → BBB > CCC > AAA
    premkt = {
        "AAA": _premkt_frame(100_000),    # 100k / 2M = 0.05
        "BBB": _premkt_frame(200_000),    # 0.10
        "CCC": _premkt_frame(150_000),    # 0.075
    }
    # Gap = (price - prior_close)/prior_close. To pass the 2-20% gap filter,
    # set price 5% above prior_close = 105.
    quotes = {s: _quote(bid=104.95, ask=105.05) for s in syms}

    scn = _make_scanner(cfg, symbols=syms, daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)

    # Lower the rvol_min for this test so all three pass (default is 2.0)
    # — instead, just confirm SORT and ordering on whatever passes.
    # All three may fail rvol_min — let's verify intent: if all pass, sorted.
    # Force rvol_min lower at runtime:
    scn.cfg = {**scn.cfg, "rvol_min": 0.01}
    res = scn.scan(asof=asof)
    syms_in = [c.symbol for c in res.watchlist]
    assert syms_in == ["BBB", "CCC", "AAA"]


def test_watchlist_capped_at_max_size(cfg, asof):
    syms = [f"S{i}" for i in range(15)]
    daily  = {s: _daily_frame(100.0, adv=2_000_000) for s in syms}
    # Increasing PM volume so order is deterministic
    premkt = {s: _premkt_frame(50_000 + 1000 * i) for i, s in enumerate(syms)}
    quotes = {s: _quote(104.95, 105.05) for s in syms}
    scn = _make_scanner(cfg, symbols=syms, daily=daily, premkt=premkt, quotes=quotes)
    scn.cfg = {**scn.cfg, "rvol_min": 0.001, "max_watchlist_size": 10}
    res = scn.scan(asof=asof)
    assert len(res.watchlist) == 10
    # The five lowest-RVOL symbols should be in rejections with size cap reason
    overflow = {c for c, r in res.rejections.items() if r == "exceeded_max_watchlist_size"}
    assert len(overflow) == 5


# ── Individual filters ─────────────────────────────────────────────────

@pytest.mark.parametrize("price,reason", [(3.0, "price_out_of_band"),
                                          (600.0, "price_out_of_band")])
def test_price_filter(cfg, asof, price, reason):
    daily = {"X": _daily_frame(prior_close=price * 0.97, adv=2_000_000)}
    premkt = {"X": _premkt_frame(200_000)}
    quotes = {"X": _quote(bid=price - 0.05, ask=price + 0.05)}
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    scn.cfg = {**scn.cfg, "rvol_min": 0.01}
    res = scn.scan(asof=asof)
    assert res.rejections["X"] == reason


def test_adv_filter(cfg, asof):
    # ADV below the 1M minimum
    daily = {"X": _daily_frame(100.0, adv=500_000)}
    premkt = {"X": _premkt_frame(200_000)}
    quotes = {"X": _quote(104.95, 105.05)}
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert "adv_below" in res.rejections["X"]


def test_premarket_volume_filter(cfg, asof):
    daily = {"X": _daily_frame(100.0, adv=2_000_000)}
    premkt = {"X": _premkt_frame(10_000)}  # under 50k threshold
    quotes = {"X": _quote(104.95, 105.05)}
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert "pm_vol_below" in res.rejections["X"]


def test_rvol_filter(cfg, asof):
    # PM vol 60k vs ADV 2M → RVOL = 0.03, well under 2.0x
    daily = {"X": _daily_frame(100.0, adv=2_000_000)}
    premkt = {"X": _premkt_frame(60_000)}
    quotes = {"X": _quote(104.95, 105.05)}
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert "rvol_below" in res.rejections["X"]


def test_gap_filter_too_small(cfg, asof):
    # 0.5% gap (price 100.5 vs prior_close 100) — below 2% min
    daily = {"X": _daily_frame(100.0, adv=2_000_000)}
    premkt = {"X": _premkt_frame(5_000_000)}  # huge to clear rvol
    quotes = {"X": _quote(100.45, 100.55)}
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert res.rejections["X"] == "gap_out_of_band"


def test_gap_filter_too_large(cfg, asof):
    # 25% gap
    daily = {"X": _daily_frame(100.0, adv=2_000_000)}
    premkt = {"X": _premkt_frame(5_000_000)}
    quotes = {"X": _quote(124.95, 125.05)}
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert res.rejections["X"] == "gap_out_of_band"


def test_mega_cap_gets_relaxed_gap_threshold(cfg, asof):
    """AAPL (mega cap) at 1.5% gap should PASS; a non-mega at 1.5% should FAIL."""
    daily = {
        "AAPL": _daily_frame(prior_close=200.0, adv=50_000_000, daily_atr_pct=3.0),
        "XYZ":  _daily_frame(prior_close=200.0, adv=50_000_000, daily_atr_pct=3.0),
    }
    premkt = {s: _premkt_frame(150_000_000) for s in ("AAPL", "XYZ")}
    quotes = {
        "AAPL": _quote(202.95, 203.05),  # +1.5% gap
        "XYZ":  _quote(202.95, 203.05),  # same +1.5% gap, non-mega
    }
    scn = _make_scanner(cfg, symbols=["AAPL", "XYZ"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    syms_in = [c.symbol for c in res.watchlist]
    assert "AAPL" in syms_in
    assert "XYZ" not in syms_in
    assert res.rejections["XYZ"] == "gap_out_of_band"


def test_negative_gap_passes_when_magnitude_in_band(cfg, asof):
    """A −5% gap must pass — spec uses absolute value."""
    daily = {"X": _daily_frame(100.0, adv=2_000_000)}
    premkt = {"X": _premkt_frame(5_000_000)}
    quotes = {"X": _quote(94.98, 95.02)}  # ~-5% gap, spread 0.04% < 0.1% cap
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert res.watchlist and res.watchlist[0].symbol == "X"
    assert res.watchlist[0].gap_pct < 0


def test_atr_filter(cfg, asof):
    """Daily ATR < 2% of price should reject."""
    daily = {"X": _daily_frame(100.0, adv=2_000_000, daily_atr_pct=1.0)}  # too low
    premkt = {"X": _premkt_frame(5_000_000)}
    quotes = {"X": _quote(104.95, 105.05)}
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert res.rejections["X"] == "atr_pct_below_min"


def test_spread_filter(cfg, asof):
    """Spread > 0.1% of price should reject."""
    daily = {"X": _daily_frame(100.0, adv=2_000_000)}
    premkt = {"X": _premkt_frame(5_000_000)}
    quotes = {"X": _quote(104.0, 106.0)}  # ~1.9% spread on ~$105 mid
    scn = _make_scanner(cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes)
    res = scn.scan(asof=asof)
    assert res.rejections["X"] == "spread_too_wide"


def test_earnings_today_excluded(cfg, asof):
    daily = {s: _daily_frame(100.0, adv=2_000_000) for s in ("AAA", "BBB")}
    premkt = {s: _premkt_frame(5_000_000) for s in ("AAA", "BBB")}
    quotes = {s: _quote(104.95, 105.05) for s in ("AAA", "BBB")}
    scn = _make_scanner(cfg, symbols=["AAA", "BBB"], daily=daily, premkt=premkt,
                        quotes=quotes, earnings_today={"AAA"})
    res = scn.scan(asof=asof)
    assert {c.symbol for c in res.watchlist} == {"BBB"}
    assert res.rejections["AAA"] == "earnings_today"


def test_locate_fields_populated(cfg, asof):
    daily = {"X": _daily_frame(100.0, adv=2_000_000)}
    premkt = {"X": _premkt_frame(5_000_000)}
    quotes = {"X": _quote(104.95, 105.05)}
    scn = _make_scanner(
        cfg, symbols=["X"], daily=daily, premkt=premkt, quotes=quotes,
        asset_overrides={"X": {"shortable": False, "easy_to_borrow": False}},
    )
    res = scn.scan(asof=asof)
    assert res.watchlist
    c = res.watchlist[0]
    assert c.is_shortable is False
    assert c.easy_to_borrow is False


# ── Defensive ──────────────────────────────────────────────────────────

def test_scan_requires_tz_aware_asof(cfg):
    tc, dc = MagicMock(), MagicMock()
    scn = UniverseScanner(tc, dc, cfg)
    with pytest.raises(ValueError, match="tz-aware"):
        scn.scan(asof=datetime(2026, 5, 25, 9, 15))  # naive


def test_insufficient_history_rejected(cfg, asof):
    # Only 5 daily bars — too few for ADV(20) / ATR(14)
    short = _daily_frame(100.0, adv=2_000_000, n_days=5)
    scn = _make_scanner(cfg, symbols=["X"], daily={"X": short},
                        premkt={"X": _premkt_frame(5_000_000)},
                        quotes={"X": _quote(104.95, 105.05)})
    res = scn.scan(asof=asof)
    assert res.rejections["X"] == "insufficient_daily_history"

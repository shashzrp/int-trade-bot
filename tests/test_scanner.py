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
    DEFAULT_CANDIDATE_SYMBOLS,
    MAX_SYMBOLS_PER_SCAN,
    NY_TZ,
    Candidate,
    ScanResult,
    UniverseScanner,
    YFinanceEarningsCalendar,
    _yfinance_unverified_session,
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

    # Always supply a deterministic predicate so the scanner never falls
    # through to the default yfinance lookup during tests.
    earn_set = {s.upper() for s in (earnings_today or set())}
    earnings_predicate = lambda sym, day: sym.upper() in earn_set  # noqa: E731

    scn = UniverseScanner(
        tc, dc, cfg,
        earnings_today_provider=earnings_predicate,
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


# ── YFinanceEarningsCalendar ───────────────────────────────────────────

def _mock_earnings_df(*dates_ny: datetime) -> pd.DataFrame:
    """Build a yfinance-shaped earnings DataFrame indexed by tz-aware ts."""
    idx = pd.DatetimeIndex(dates_ny)
    return pd.DataFrame(
        {"EPS Estimate": [None] * len(idx),
         "Reported EPS": [None] * len(idx),
         "Surprise(%)": [None] * len(idx)},
        index=idx,
    )


def _patch_yf_ticker(monkeypatch, *, by_symbol: dict[str, object]):
    """Patch scanner.yf.Ticker so each symbol returns a canned mock.

    Each value in ``by_symbol`` is either a DataFrame (returned from
    ``get_earnings_dates``) or an Exception instance (raised from it).
    """
    def fake_ticker(sym: str, *args, **kwargs):
        # Accept (and ignore) the session= kwarg the scanner now passes.
        mock = MagicMock()
        canned = by_symbol.get(sym.upper())
        if isinstance(canned, BaseException):
            mock.get_earnings_dates.side_effect = canned
        else:
            mock.get_earnings_dates.return_value = canned
        return mock

    import scanner as scanner_mod
    monkeypatch.setattr(scanner_mod.yf, "Ticker", fake_ticker)


def test_yf_calendar_hits_on_matching_session_day(monkeypatch, asof):
    """A symbol whose earnings calendar lists the session day → True."""
    session_day = asof.date()  # 2026-05-25
    df = _mock_earnings_df(datetime(2026, 5, 25, 16, 30, tzinfo=NY_TZ))
    _patch_yf_ticker(monkeypatch, by_symbol={"AAA": df})

    cal = YFinanceEarningsCalendar()
    assert cal("AAA", session_day) is True


def test_yf_calendar_misses_on_other_days(monkeypatch, asof):
    """No matching date in the calendar → False."""
    session_day = asof.date()
    df = _mock_earnings_df(
        datetime(2026, 5, 26, 16, 30, tzinfo=NY_TZ),
        datetime(2026, 5, 24, 9, 0, tzinfo=NY_TZ),
    )
    _patch_yf_ticker(monkeypatch, by_symbol={"BBB": df})

    cal = YFinanceEarningsCalendar()
    assert cal("BBB", session_day) is False


def test_yf_calendar_bmo_and_amc_both_count(monkeypatch, asof):
    """Before-market and after-market both share the calendar date."""
    session_day = asof.date()
    bmo = _mock_earnings_df(datetime(2026, 5, 25, 7, 0, tzinfo=NY_TZ))   # 07:00 ET
    amc = _mock_earnings_df(datetime(2026, 5, 25, 16, 30, tzinfo=NY_TZ)) # 16:30 ET
    _patch_yf_ticker(monkeypatch, by_symbol={"BMO": bmo, "AMC": amc})

    cal = YFinanceEarningsCalendar()
    assert cal("BMO", session_day) is True
    assert cal("AMC", session_day) is True


def test_yf_calendar_caches_per_session_day(monkeypatch, asof):
    """Two calls for the same (symbol, day) must not re-query yfinance."""
    session_day = asof.date()
    df = _mock_earnings_df(datetime(2026, 5, 25, 16, 30, tzinfo=NY_TZ))

    call_count = {"n": 0}
    def counting_ticker(sym, *args, **kwargs):
        call_count["n"] += 1
        m = MagicMock()
        m.get_earnings_dates.return_value = df
        return m
    import scanner as scanner_mod
    monkeypatch.setattr(scanner_mod.yf, "Ticker", counting_ticker)

    cal = YFinanceEarningsCalendar()
    cal("AAA", session_day)
    cal("AAA", session_day)
    cal("AAA", session_day)
    assert call_count["n"] == 1

    # A different session day must miss the cache and query again.
    cal("AAA", session_day + timedelta(days=1))
    assert call_count["n"] == 2


def test_yf_calendar_handles_network_failure(monkeypatch, asof):
    """A lookup exception → fails open (False) and caches the negative."""
    session_day = asof.date()
    err = ConnectionError("simulated outage")
    _patch_yf_ticker(monkeypatch, by_symbol={"ZZZ": err})

    cal = YFinanceEarningsCalendar()
    assert cal("ZZZ", session_day) is False
    # Cached, so a second call doesn't re-raise / re-query either.
    assert cal("ZZZ", session_day) is False


def test_unverified_session_has_verify_false_and_closes():
    """The context manager must yield a curl_cffi session with TLS
    verification disabled and clean it up on exit."""
    with _yfinance_unverified_session() as session:
        assert session.verify is False
        # Should be a curl_cffi session, not a plain requests.Session.
        from curl_cffi.requests import Session as CurlSession
        assert isinstance(session, CurlSession)


def test_query_passes_unverified_session_to_yfinance(monkeypatch, asof):
    """Calendar's _query must hand the unverified session to yf.Ticker
    so SSL bypass actually reaches the underlying HTTP layer."""
    captured: dict[str, object] = {}

    def spy_ticker(sym, *args, **kwargs):
        captured["sym"] = sym
        captured["session"] = kwargs.get("session")
        m = MagicMock()
        m.get_earnings_dates.return_value = pd.DataFrame()
        return m

    import scanner as scanner_mod
    monkeypatch.setattr(scanner_mod.yf, "Ticker", spy_ticker)

    YFinanceEarningsCalendar()("AAPL", asof.date())
    assert captured["sym"] == "AAPL"
    assert captured["session"] is not None
    assert captured["session"].verify is False


def test_yf_calendar_handles_empty_response(monkeypatch, asof):
    """get_earnings_dates returning None or empty df → False."""
    session_day = asof.date()
    _patch_yf_ticker(monkeypatch, by_symbol={"NONE": None,
                                              "EMPTY": pd.DataFrame()})
    cal = YFinanceEarningsCalendar()
    assert cal("NONE", session_day) is False
    assert cal("EMPTY", session_day) is False


def test_default_provider_is_yfinance_when_none_supplied(cfg):
    """Scanner with no explicit provider must use YFinanceEarningsCalendar
    so the exclude_earnings_today flag is active out of the box."""
    tc, dc = MagicMock(), MagicMock()
    scn = UniverseScanner(tc, dc, cfg)
    assert isinstance(scn.earnings_today, YFinanceEarningsCalendar)


def test_default_candidate_universe_is_hardcoded_shortlist(cfg):
    """No candidate_symbols passed → scanner falls back to the
    operator-vetted DEFAULT_CANDIDATE_SYMBOLS list, not an all-assets sweep."""
    tc, dc = MagicMock(), MagicMock()
    scn = UniverseScanner(tc, dc, cfg)
    assert scn._candidate_symbols == [s.upper() for s in DEFAULT_CANDIDATE_SYMBOLS]
    # Spot-check a few names from the spec are present.
    for required in ("AAPL", "NVDA", "OKLO", "MRVL"):
        assert required in scn._candidate_symbols
    # No duplicates after dedup.
    assert len(scn._candidate_symbols) == len(set(scn._candidate_symbols))


# ── MAX_SYMBOLS_PER_SCAN cap ───────────────────────────────────────────

def test_max_symbols_per_scan_is_100():
    """The published cap must be 100 — exposed as MAX_SYMBOLS_PER_SCAN."""
    assert MAX_SYMBOLS_PER_SCAN == 100


def test_candidate_symbols_truncated_at_cap(cfg, caplog):
    """A list larger than the cap is truncated (with a warning)."""
    big = [f"S{i:04d}" for i in range(150)]
    tc, dc = MagicMock(), MagicMock()
    with caplog.at_level("WARNING", logger="scanner"):
        scn = UniverseScanner(tc, dc, cfg, candidate_symbols=big)
    assert len(scn._candidate_symbols) == 100
    # Order preserved — first 100 input symbols survive.
    assert scn._candidate_symbols == [s.upper() for s in big[:100]]
    assert any("truncated from 150 to 100" in rec.message for rec in caplog.records)


def test_candidate_symbols_under_cap_pass_through(cfg, caplog):
    """A list at or below the cap is left intact with no warning."""
    fifty = [f"S{i:04d}" for i in range(50)]
    tc, dc = MagicMock(), MagicMock()
    with caplog.at_level("WARNING", logger="scanner"):
        scn = UniverseScanner(tc, dc, cfg, candidate_symbols=fifty)
    assert len(scn._candidate_symbols) == 50
    assert not any("truncated" in rec.message for rec in caplog.records)


def test_duplicates_deduped_before_counting_against_cap(cfg):
    """Duplicates must not consume cap quota — order-preserving dedup."""
    inputs = ["AAPL", "msft", "AAPL", "NVDA", "MSFT"]
    tc, dc = MagicMock(), MagicMock()
    scn = UniverseScanner(tc, dc, cfg, candidate_symbols=inputs)
    assert scn._candidate_symbols == ["AAPL", "MSFT", "NVDA"]


def test_at_cap_exactly_not_truncated(cfg, caplog):
    """100 symbols passes through exactly — boundary check."""
    hundred = [f"S{i:04d}" for i in range(100)]
    tc, dc = MagicMock(), MagicMock()
    with caplog.at_level("WARNING", logger="scanner"):
        scn = UniverseScanner(tc, dc, cfg, candidate_symbols=hundred)
    assert len(scn._candidate_symbols) == 100
    assert not any("truncated" in rec.message for rec in caplog.records)

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
from requests.exceptions import RequestException

from config import StrategyConfig
from scanner import (
    DEFAULT_CANDIDATE_SYMBOLS,
    FINNHUB_EARNINGS_URL,
    FinnhubEarningsCalendar,
    MAX_SYMBOLS_PER_SCAN,
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

    # Always supply a deterministic predicate so the scanner never falls
    # through to the default Finnhub lookup during tests.
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
    # config.yaml may have exclude_earnings_today=false; this test specifically
    # exercises the filter, so force it on regardless of YAML state.
    scn.cfg = {**scn.cfg, "exclude_earnings_today": True}
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


# ── FinnhubEarningsCalendar ────────────────────────────────────────────

def _finnhub_session(payload: dict | None = None, *,
                     get_exception: BaseException | None = None,
                     raise_for_status: BaseException | None = None) -> MagicMock:
    """Build a mock ``requests.Session`` that returns a canned Finnhub
    response. Either ``payload`` is returned via ``.json()``, or ``.get()``
    raises ``get_exception`` outright, or ``.raise_for_status()`` raises."""
    session = MagicMock()
    if get_exception is not None:
        session.get.side_effect = get_exception
        return session
    response = MagicMock()
    if raise_for_status is not None:
        response.raise_for_status.side_effect = raise_for_status
    else:
        response.raise_for_status.return_value = None
    response.json.return_value = payload or {}
    session.get.return_value = response
    return session


def _finnhub_payload(*items: dict) -> dict:
    """Shape Finnhub's /calendar/earnings JSON: {"earningsCalendar": [...]}."""
    return {"earningsCalendar": list(items)}


def test_finnhub_returns_true_when_symbol_in_calendar(asof):
    """A symbol whose Finnhub calendar entry matches session_day → True."""
    session_day = asof.date()  # 2026-05-25
    payload = _finnhub_payload(
        {"symbol": "AAA", "date": "2026-05-25", "hour": "amc"},
        {"symbol": "BBB", "date": "2026-05-25", "hour": "bmo"},
    )
    cal = FinnhubEarningsCalendar(api_key="test-key",
                                   session=_finnhub_session(payload))
    assert cal("AAA", session_day) is True
    assert cal("BBB", session_day) is True


def test_finnhub_returns_false_when_symbol_not_in_calendar(asof):
    """Symbol absent from the Finnhub set → False."""
    session_day = asof.date()
    payload = _finnhub_payload({"symbol": "OTHER", "date": "2026-05-25"})
    cal = FinnhubEarningsCalendar(api_key="test-key",
                                   session=_finnhub_session(payload))
    assert cal("AAA", session_day) is False


def test_finnhub_bmo_and_amc_both_count(asof):
    """BMO and AMC entries on the session day are both flagged."""
    session_day = asof.date()
    payload = _finnhub_payload(
        {"symbol": "BMO", "date": "2026-05-25", "hour": "bmo"},
        {"symbol": "AMC", "date": "2026-05-25", "hour": "amc"},
    )
    cal = FinnhubEarningsCalendar(api_key="test-key",
                                   session=_finnhub_session(payload))
    assert cal("BMO", session_day) is True
    assert cal("AMC", session_day) is True


def test_finnhub_filters_next_day_entries_out(asof):
    """Entries from session_day+1 (returned because we fetch a 2-day range)
    must NOT count against today — date equality filter kicks them out."""
    session_day = asof.date()  # 2026-05-25
    payload = _finnhub_payload(
        {"symbol": "TODAY",    "date": "2026-05-25"},
        {"symbol": "TOMORROW", "date": "2026-05-26"},
    )
    cal = FinnhubEarningsCalendar(api_key="test-key",
                                   session=_finnhub_session(payload))
    assert cal("TODAY",    session_day) is True
    assert cal("TOMORROW", session_day) is False


def test_finnhub_single_fetch_per_session_day(asof):
    """Spec: one Alpaca/Finnhub call per session_day, regardless of how
    many _has_earnings_today queries fire. Subsequent lookups are O(1)."""
    session_day = asof.date()
    session = _finnhub_session(_finnhub_payload(
        {"symbol": "AAA", "date": "2026-05-25"},
    ))
    cal = FinnhubEarningsCalendar(api_key="test-key", session=session)

    cal("AAA", session_day)
    cal("BBB", session_day)
    cal("CCC", session_day)
    cal("AAA", session_day)
    assert session.get.call_count == 1

    # New session day forces a second call.
    cal("AAA", session_day + timedelta(days=1))
    assert session.get.call_count == 2


def test_finnhub_request_params_match_spec(asof):
    """Sanity-check the wire call: correct URL, from/to/token params."""
    session_day = asof.date()  # 2026-05-25
    session = _finnhub_session(_finnhub_payload())
    cal = FinnhubEarningsCalendar(api_key="secret-token", session=session)
    cal("AAA", session_day)

    call_args = session.get.call_args
    assert call_args.args[0] == FINNHUB_EARNINGS_URL
    params = call_args.kwargs["params"]
    assert params["from"] == "2026-05-25"
    assert params["to"]   == "2026-05-26"   # [session_day, session_day+1]
    assert params["token"] == "secret-token"


def test_finnhub_network_failure_disables_filter_for_session(asof, caplog):
    """Network exception → log warning, return False for every query that
    day, AND don't crash. Filter is a no-op rather than blocking."""
    session_day = asof.date()
    session = _finnhub_session(get_exception=ConnectionError("simulated outage"))
    cal = FinnhubEarningsCalendar(api_key="test-key", session=session)
    with caplog.at_level("WARNING", logger="scanner"):
        assert cal("AAA", session_day) is False
        assert cal("BBB", session_day) is False
    assert any("Finnhub earnings fetch failed" in r.message for r in caplog.records)
    # No retry within the session — exactly one HTTP attempt.
    assert session.get.call_count == 1


def test_finnhub_http_error_disables_filter(asof, caplog):
    """HTTP 4xx/5xx (raise_for_status) → same fail-open behavior."""
    session_day = asof.date()
    http_err = RequestException("HTTP 500")
    session = _finnhub_session(_finnhub_payload(), raise_for_status=http_err)
    cal = FinnhubEarningsCalendar(api_key="test-key", session=session)
    with caplog.at_level("WARNING", logger="scanner"):
        assert cal("AAA", session_day) is False
    assert any("Finnhub earnings fetch failed" in r.message for r in caplog.records)


def test_finnhub_missing_api_key_disables_filter(asof, caplog, monkeypatch):
    """No FINNHUB_API_KEY env var and no explicit key → filter no-op + warning."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    session_day = asof.date()
    cal = FinnhubEarningsCalendar()    # api_key=None, falls back to env
    with caplog.at_level("WARNING", logger="scanner"):
        assert cal("AAA", session_day) is False
    assert any("FINNHUB_API_KEY not set" in r.message for r in caplog.records)


def test_finnhub_malformed_response_disables_filter(asof, caplog):
    """Response missing/changed schema → fail open with warning."""
    session_day = asof.date()
    # `earningsCalendar` returns None — the .get(...) yields None which we
    # treat as empty; that should NOT be an error. Build an actively-broken
    # case: items list contains a non-dict (causes TypeError on .get).
    session = _finnhub_session({"earningsCalendar": ["not-a-dict"]})
    cal = FinnhubEarningsCalendar(api_key="test-key", session=session)
    with caplog.at_level("WARNING", logger="scanner"):
        assert cal("AAA", session_day) is False
    assert any("malformed" in r.message for r in caplog.records)


def test_finnhub_empty_calendar_is_not_an_error(asof, caplog):
    """An empty earnings calendar is a valid response — every symbol
    returns False, no warning logged."""
    session_day = asof.date()
    session = _finnhub_session(_finnhub_payload())   # zero entries
    cal = FinnhubEarningsCalendar(api_key="test-key", session=session)
    with caplog.at_level("WARNING", logger="scanner"):
        assert cal("AAA", session_day) is False
        assert cal("BBB", session_day) is False
    assert not any("Finnhub" in r.message for r in caplog.records)


def test_default_provider_is_finnhub_when_none_supplied(cfg):
    """Scanner with no explicit provider must default to FinnhubEarningsCalendar
    so the exclude_earnings_today flag is active out of the box."""
    tc, dc = MagicMock(), MagicMock()
    scn = UniverseScanner(tc, dc, cfg)
    assert isinstance(scn.earnings_today, FinnhubEarningsCalendar)


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

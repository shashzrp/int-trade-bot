"""
Pre-market universe scanner.

Runs around 09:15 ET each day and emits the watchlist of 3–10 symbols the
bot will actually evaluate during HUNTING.  Every threshold comes from
``config.yaml``; the scanner itself is dumb — it just applies them.

Inputs
------
trading_client  — `alpaca.trading.client.TradingClient`. Used for asset
                  metadata (tradable, exchange, class, easy_to_borrow).
data_client     — `alpaca.data.historical.StockHistoricalDataClient`.
                  Used for daily bars (ADV, ATR, gap), pre-market 1-min
                  bars (PM volume), and latest quote (price, spread).
cfg.universe    — every filter threshold.
earnings_today_provider — optional callable ``date -> Iterable[str]`` that
                  yields symbols with earnings releasing post-close today.
                  Alpaca has no earnings endpoint; wire your own source
                  (Polygon, Finnhub, CSV).  If None, the filter is a no-op
                  and a warning is logged.

Output: ``ScanResult`` with `watchlist` (top-N sorted by RVOL desc) and
        `rejections` (symbol → first failing filter, for debugging).
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from config import StrategyConfig, get_strategy_config
from indicators import atr, true_range

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
PREMARKET_START = time(4, 0)
SESSION_OPEN = time(9, 30)


# ── Public types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    symbol: str
    price: float
    adv_20d: float
    premarket_vol: float
    rvol: float
    gap_pct: float          # signed: positive = gap up, negative = gap down
    daily_atr_pct: float
    spread_pct: float
    is_mega_cap: bool
    prior_close: float
    is_shortable: bool
    easy_to_borrow: bool


@dataclass
class ScanResult:
    watchlist: list[Candidate] = field(default_factory=list)
    rejections: dict[str, str] = field(default_factory=dict)


# ── Scanner ─────────────────────────────────────────────────────────────

class UniverseScanner:
    def __init__(
        self,
        trading_client: TradingClient,
        data_client: StockHistoricalDataClient,
        cfg: StrategyConfig | None = None,
        *,
        earnings_today_provider: Callable[[date], Iterable[str]] | None = None,
        candidate_symbols: Iterable[str] | None = None,
    ) -> None:
        self.tc = trading_client
        self.dc = data_client
        self.cfg = (cfg or get_strategy_config()).universe
        self.earnings_today = earnings_today_provider
        # Optional pre-filter — if supplied, skip the asset-metadata sweep.
        self._candidate_symbols = (
            None if candidate_symbols is None else [s.upper() for s in candidate_symbols]
        )

    # ── Entry point ────────────────────────────────────────────────

    def scan(self, *, asof: datetime) -> ScanResult:
        """Run the full filter pipeline for the session containing ``asof``."""
        if asof.tzinfo is None:
            raise ValueError("asof must be tz-aware (America/New_York).")
        session_day = asof.astimezone(NY_TZ).date()
        result = ScanResult()

        # Stage 1 — candidate symbols via asset metadata (cheap filter)
        symbols = self._candidate_universe(result)
        if not symbols:
            logger.warning("No symbols survived asset-metadata pre-filter.")
            return result

        # Stage 2 — earnings exclusion (cheap, in-process)
        if self.cfg.get("exclude_earnings_today", True):
            if self.earnings_today is None:
                logger.warning(
                    "exclude_earnings_today=true but no earnings_today_provider supplied — "
                    "filter is a no-op for this run."
                )
            else:
                earn = {s.upper() for s in self.earnings_today(session_day)}
                survivors = []
                for s in symbols:
                    if s in earn:
                        result.rejections[s] = "earnings_today"
                    else:
                        survivors.append(s)
                symbols = survivors

        if not symbols:
            return result

        # Stage 3 — bulk market data (daily bars, premarket bars, quotes)
        daily = self._fetch_daily_bars(symbols, session_day)
        premkt = self._fetch_premarket_bars(symbols, session_day)
        quotes = self._fetch_latest_quotes(symbols)

        # Stage 4 — per-symbol metric computation + threshold checks
        candidates: list[Candidate] = []
        mega = {s.upper() for s in self.cfg.get("mega_caps", [])}
        for sym in symbols:
            try:
                cand = self._evaluate(sym, daily, premkt, quotes, is_mega=sym in mega)
            except _Reject as r:
                result.rejections[sym] = r.reason
                continue
            candidates.append(cand)

        # Stage 5 — sort and cap
        candidates.sort(key=lambda c: c.rvol, reverse=True)
        max_size = int(self.cfg.get("max_watchlist_size", 10))
        result.watchlist = candidates[:max_size]

        for c in candidates[max_size:]:
            result.rejections[c.symbol] = "exceeded_max_watchlist_size"

        logger.info(
            "scan complete day=%s candidates_pass=%d watchlist=%d rejected=%d",
            session_day, len(candidates), len(result.watchlist), len(result.rejections),
        )
        return result

    # ── Stage 1 ────────────────────────────────────────────────────

    def _candidate_universe(self, result: ScanResult) -> list[str]:
        if self._candidate_symbols is not None:
            # Caller supplied a pre-screened list — still apply asset filters.
            assets = [self._safe_get_asset(s, result) for s in self._candidate_symbols]
            assets = [a for a in assets if a is not None]
        else:
            req = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
            )
            assets = self.tc.get_all_assets(req)

        allowed_ex = {AssetExchange.NYSE, AssetExchange.NASDAQ, AssetExchange.ARCA}
        out: list[str] = []
        for a in assets:
            sym = a.symbol.upper()
            if not a.tradable:
                result.rejections[sym] = "not_tradable"
                continue
            if a.status != AssetStatus.ACTIVE:
                result.rejections[sym] = "inactive"
                continue
            if a.exchange not in allowed_ex:
                result.rejections[sym] = f"exchange:{a.exchange}"
                continue
            if a.asset_class != AssetClass.US_EQUITY:
                result.rejections[sym] = f"class:{a.asset_class}"
                continue
            # Cheap symbol-shape filter for warrants/units/rights — these end
            # in 'W', 'U', 'R' on Alpaca for compound tickers like 'XYZW'.
            # We use the conservative test: any non-A-Z character in the
            # ticker (a dot, slash, plus) is excluded.
            if not sym.isalpha():
                result.rejections[sym] = "non_alpha_ticker"
                continue
            out.append(sym)
        return out

    def _safe_get_asset(self, sym: str, result: ScanResult):
        try:
            return self.tc.get_asset(sym)
        except Exception:  # alpaca-py raises a SDK-specific HTTP exception
            result.rejections[sym] = "asset_lookup_failed"
            return None

    # ── Stage 3 — bulk market data fetches ─────────────────────────

    def _fetch_daily_bars(
        self, symbols: list[str], session_day: date
    ) -> dict[str, pd.DataFrame]:
        """30 calendar days of daily bars; enough for ADV(20) and ATR(14)."""
        start_d = session_day - timedelta(days=45)  # extra buffer for holidays
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.combine(start_d, time(0, 0), tzinfo=NY_TZ),
            end=datetime.combine(session_day, time(0, 0), tzinfo=NY_TZ),
        )
        return self._bars_to_frames(self.dc.get_stock_bars(req))

    def _fetch_premarket_bars(
        self, symbols: list[str], session_day: date
    ) -> dict[str, pd.DataFrame]:
        """1-min bars from 04:00 ET to 09:30 ET on session_day."""
        start = datetime.combine(session_day, PREMARKET_START, tzinfo=NY_TZ)
        end   = datetime.combine(session_day, SESSION_OPEN,    tzinfo=NY_TZ)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
            start=start,
            end=end,
        )
        return self._bars_to_frames(self.dc.get_stock_bars(req))

    def _fetch_latest_quotes(self, symbols: list[str]):
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        return self.dc.get_stock_latest_quote(req)

    @staticmethod
    def _bars_to_frames(barset) -> dict[str, pd.DataFrame]:
        """Convert alpaca-py BarSet → {symbol: DataFrame with NY-tz index}."""
        data = getattr(barset, "data", None) or {}
        out: dict[str, pd.DataFrame] = {}
        for sym, bars in data.items():
            if not bars:
                continue
            df = pd.DataFrame(
                [{"open": b.open, "high": b.high, "low": b.low,
                  "close": b.close, "volume": b.volume,
                  "timestamp": b.timestamp} for b in bars]
            ).set_index("timestamp")
            df.index = pd.DatetimeIndex(df.index).tz_convert(NY_TZ)
            out[sym] = df
        return out

    # ── Stage 4 — per-symbol scoring ───────────────────────────────

    def _evaluate(
        self,
        sym: str,
        daily: dict[str, pd.DataFrame],
        premkt: dict[str, pd.DataFrame],
        quotes,
        *,
        is_mega: bool,
    ) -> Candidate:
        ddf = daily.get(sym)
        if ddf is None or len(ddf) < 21:
            raise _Reject("insufficient_daily_history")

        # Prior close = last completed daily bar BEFORE session_day
        prior_close = float(ddf["close"].iloc[-1])

        # ADV(20): mean of last 20 daily volumes
        adv_20 = float(ddf["volume"].iloc[-20:].mean())
        if adv_20 < self.cfg["adv_min_shares"]:
            raise _Reject(f"adv_below_{self.cfg['adv_min_shares']}")

        # Daily ATR(14) — uses 14-period Wilder on daily TR
        a14 = atr(ddf, period=14)
        daily_atr = float(a14.iloc[-1]) if not pd.isna(a14.iloc[-1]) else 0.0

        # Latest quote = current price (mid) + spread
        q = self._quote(quotes, sym)
        bid, ask = float(q.bid_price), float(q.ask_price)
        if ask <= 0 or bid <= 0:
            raise _Reject("invalid_quote")
        price = (bid + ask) / 2
        spread_pct = (ask - bid) / price * 100

        # Pre-market volume
        pdf = premkt.get(sym)
        pm_vol = float(pdf["volume"].sum()) if pdf is not None else 0.0

        # RVOL — pre-market vol expressed as fraction of 20-day ADV
        rvol = pm_vol / adv_20 if adv_20 > 0 else 0.0

        # Gap = (price − prior_close) / prior_close × 100
        gap_pct = (price - prior_close) / prior_close * 100

        # ── Threshold checks ───────────────────────────────────
        if not (self.cfg["price_min"] <= price <= self.cfg["price_max"]):
            raise _Reject("price_out_of_band")

        if pm_vol < self.cfg["premarket_volume_min"]:
            raise _Reject(f"pm_vol_below_{self.cfg['premarket_volume_min']}")

        if rvol < self.cfg["rvol_min"]:
            raise _Reject(f"rvol_below_{self.cfg['rvol_min']}")

        gap_min = self.cfg["mega_cap_gap_min_pct"] if is_mega else self.cfg["gap_min_pct"]
        if not (gap_min <= abs(gap_pct) <= self.cfg["gap_max_pct"]):
            raise _Reject("gap_out_of_band")

        daily_atr_pct = (daily_atr / price * 100) if price > 0 else 0.0
        if daily_atr_pct < self.cfg["daily_atr_pct_min"]:
            raise _Reject("atr_pct_below_min")

        if spread_pct > self.cfg["spread_max_pct"]:
            raise _Reject("spread_too_wide")

        # Locate flags — useful for the SHORT path later, but fetched here
        # so we don't repeat the asset call per signal.
        asset = self._safe_get_asset(sym, ScanResult())
        shortable = bool(getattr(asset, "shortable", False))
        etb = bool(getattr(asset, "easy_to_borrow", False))

        return Candidate(
            symbol=sym, price=price, adv_20d=adv_20, premarket_vol=pm_vol,
            rvol=rvol, gap_pct=gap_pct, daily_atr_pct=daily_atr_pct,
            spread_pct=spread_pct, is_mega_cap=is_mega, prior_close=prior_close,
            is_shortable=shortable, easy_to_borrow=etb,
        )

    @staticmethod
    def _quote(quotes, sym: str):
        """Robust accessor for alpaca-py's quote response shape."""
        if hasattr(quotes, "__getitem__"):
            try:
                return quotes[sym]
            except (KeyError, TypeError):
                pass
        if hasattr(quotes, "data"):
            return quotes.data[sym]
        raise _Reject("no_quote_returned")


class _Reject(Exception):
    """Internal flow-control: one filter rejected this symbol."""
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ── CLI: live scan against Alpaca for a real session ────────────────────

def _main() -> int:  # pragma: no cover — exercised manually with real creds
    parser = argparse.ArgumentParser(description="Run the pre-market scanner.")
    parser.add_argument("--asof", type=str, default=None,
                        help="ISO datetime in NY (e.g. 2026-05-25T09:15). Defaults to now.")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated candidate list (skips asset-metadata sweep).")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    from clients import get_data_client, get_trading_client
    tc = get_trading_client()
    dc = get_data_client()

    if args.asof:
        asof = datetime.fromisoformat(args.asof)
        if asof.tzinfo is None:
            asof = asof.replace(tzinfo=NY_TZ)
    else:
        asof = datetime.now(tz=NY_TZ)

    candidates = (args.symbols.split(",") if args.symbols else None)
    scanner = UniverseScanner(tc, dc, candidate_symbols=candidates)
    result = scanner.scan(asof=asof)

    print(f"Watchlist ({len(result.watchlist)}):")
    for c in result.watchlist:
        print(f"  {c.symbol:6s} price={c.price:7.2f} rvol={c.rvol:5.2f}x "
              f"gap={c.gap_pct:+6.2f}% atr={c.daily_atr_pct:5.2f}% "
              f"spread={c.spread_pct:5.3f}%")
    print(f"\nRejections ({len(result.rejections)}):")
    by_reason: dict[str, int] = {}
    for sym, reason in result.rejections.items():
        by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {reason:32s} {count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

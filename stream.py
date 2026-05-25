"""
Live 1-minute bar streaming with bar-close validation + heartbeat.

Two responsibilities:

  1. **Bar-close validation** — alpaca-py emits each bar as its minute
     closes, but we add a small wall-clock grace before dispatching:
     a bar at start-timestamp T is treated as closed when
     ``now ≥ T + 60s + bar_close_grace_seconds``.  This guards against
     a clock skew / out-of-order websocket frame.  Pure function so it
     can be unit-tested with frozen time.

  2. **Heartbeat** — track the wall-clock of the last received bar
     across ALL subscribed symbols.  If it goes silent > 10 s during
     market hours, the caller must halt new entries and alert.

The actual websocket subscription is thin glue over alpaca-py — covered
by the manual paper-trading dry run rather than unit tests.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca.data.live import StockDataStream

from config import StrategyConfig, get_strategy_config

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")

BarCallback = Callable[[str, pd.Series, datetime], Awaitable[None]]


def is_bar_closed(*, bar_start: datetime, now: datetime, grace_seconds: int = 2) -> bool:
    """A 1-min bar with start timestamp ``bar_start`` is considered closed
    once wall-clock reaches ``bar_start + 60s + grace``."""
    if bar_start.tzinfo is None or now.tzinfo is None:
        raise ValueError("bar_start and now must both be tz-aware.")
    return now >= bar_start + timedelta(seconds=60 + int(grace_seconds))


class BarStream:
    def __init__(
        self,
        data_stream: StockDataStream,
        symbols: list[str],
        on_closed_bar: BarCallback,
        cfg: StrategyConfig | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc),
    ) -> None:
        self.stream = data_stream
        self.symbols = [s.upper() for s in symbols]
        self.on_closed_bar = on_closed_bar
        self.cfg = (cfg or get_strategy_config()).stream
        self._clock = clock
        self._last_msg_at: datetime | None = None
        self._heartbeat_timeout_s = int(self.cfg["heartbeat_timeout_seconds"])
        self._grace_s = int(self.cfg["bar_close_grace_seconds"])

    # ── Subscription ──────────────────────────────────────────────

    def subscribe(self) -> None:
        """Register the bar handler with alpaca-py."""
        self.stream.subscribe_bars(self._handle_bar, *self.symbols)

    async def run(self) -> None:  # pragma: no cover — live websocket
        self.subscribe()
        await self.stream._run_forever()    # alpaca-py's internal loop

    async def stop(self) -> None:  # pragma: no cover
        await self.stream.stop_ws()

    # ── Bar dispatch ──────────────────────────────────────────────

    async def _handle_bar(self, bar) -> None:
        """Internal callback from alpaca-py.  Validates closure, updates
        heartbeat, then dispatches to the user callback."""
        now = self._clock()
        self._last_msg_at = now

        # alpaca-py Bar carries .symbol, .timestamp (UTC), and OHLCV fields.
        try:
            sym = str(bar.symbol).upper()
            ts_utc = bar.timestamp
        except AttributeError:
            logger.warning("malformed bar payload: %r", bar)
            return
        ts_utc = pd.Timestamp(ts_utc).tz_convert(timezone.utc) if pd.Timestamp(ts_utc).tzinfo else pd.Timestamp(ts_utc, tz=timezone.utc)
        ts_ny = ts_utc.tz_convert(NY_TZ).to_pydatetime()

        if not is_bar_closed(bar_start=ts_ny, now=now, grace_seconds=self._grace_s):
            logger.debug("bar not yet closed sym=%s ts=%s now=%s", sym, ts_ny, now)
            return

        series = pd.Series({
            "open":   float(bar.open),
            "high":   float(bar.high),
            "low":    float(bar.low),
            "close":  float(bar.close),
            "volume": float(bar.volume),
        }, name=ts_ny)

        try:
            await self.on_closed_bar(sym, series, ts_ny)
        except Exception as exc:
            logger.exception("on_closed_bar callback failed sym=%s err=%s", sym, exc)

    # ── Heartbeat ─────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """True if any bar arrived within the last
        ``stream.heartbeat_timeout_seconds`` wall-clock seconds."""
        if self._last_msg_at is None:
            return False
        elapsed = (self._clock() - self._last_msg_at).total_seconds()
        return elapsed <= self._heartbeat_timeout_s

    @property
    def last_msg_at(self) -> datetime | None:
        return self._last_msg_at

"""
End-of-session wind-down scheduler.

Per spec:
  • WIND_DOWN  (15:30 – 15:54 ET): no new entries; tighten trailing stops.
    The state machine already disallows new entries at this time (the
    `entry_rvol_threshold` returns None past 15:00); trail-tightening is
    handled by the ExitManager's EMA9 trail.  Nothing extra to do here.
  • FORCED_FLAT (15:55 ET): cancel ALL open orders, close ALL positions.

This module is a tiny coordinator: ``tick(now)`` is called every bar (or
every loop tick); when wall-clock crosses the force-flat boundary, the
callback fires exactly once.
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Callable
from zoneinfo import ZoneInfo

from config import StrategyConfig, get_strategy_config

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")


def _parse_time(hhmm: str) -> time:
    parts = hhmm.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) > 2 else 0
    return time(h, m, s)


class WindDown:
    def __init__(
        self,
        cfg: StrategyConfig | None = None,
        *,
        on_force_flat: Callable[[], None],
    ) -> None:
        self.cfg = cfg or get_strategy_config()
        self.force_flat_time = _parse_time(self.cfg.exits["forced_flat_time"])
        self._fired_on: date | None = None
        self._on_force_flat = on_force_flat

    def tick(self, now: datetime) -> bool:
        """Returns True iff the force-flat callback was invoked this tick."""
        if now.tzinfo is None:
            raise ValueError("now must be tz-aware")
        ny = now.astimezone(NY_TZ)
        today = ny.date()
        if self._fired_on == today:
            return False
        if ny.time() >= self.force_flat_time:
            logger.warning("wind_down: forcing flat at %s", ny.isoformat())
            self._fired_on = today
            self._on_force_flat()
            return True
        return False

    def reset(self) -> None:
        """For tests / new trading day."""
        self._fired_on = None


# Local import to avoid circulars at module load time.
from datetime import date  # noqa: E402

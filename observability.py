"""
Structured logging + Prometheus metrics.

All logs go through ``structlog`` and emit one JSON line per record.  Every
field the spec mentions is present:

  signals_evaluated_total      Counter{symbol}
  signals_passed_total         Counter{symbol, side}
  orders_submitted_total       Counter{symbol, side}
  orders_filled_total          Counter{symbol, side}
  fills_slippage_bps           Histogram{symbol}
  daily_pnl_pct                Gauge
  position_count               Gauge
  circuit_breaker_active       Gauge (0/1)

A tiny ``configure_logging`` helper switches Python's stdlib logging onto
structlog's JSON renderer once at startup.
"""
from __future__ import annotations

import logging
import sys

import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server


# ── Logging ─────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    """Idempotent — safe to call multiple times in tests."""
    level_num = getattr(logging, level.upper(), logging.INFO)

    # Stdlib root logger: emit raw messages to stderr; structlog wraps them.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    # Don't duplicate handlers on repeat calls.
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level_num)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


# ── Prometheus metrics ──────────────────────────────────────────────────

signals_eval_counter = Counter(
    "signals_evaluated_total",
    "Total signal evaluations (one per closed bar per watched symbol).",
    ["symbol"],
)

signals_passed_counter = Counter(
    "signals_passed_total",
    "Signals that passed every gate.",
    ["symbol", "side"],
)

orders_submitted_counter = Counter(
    "orders_submitted_total",
    "Orders submitted to the broker (excludes chase retries).",
    ["symbol", "side"],
)

orders_filled_counter = Counter(
    "orders_filled_total",
    "Orders that resulted in a fill.",
    ["symbol", "side"],
)

# Alias used by older imports — keep both names for backward compat.
fills_counter = orders_filled_counter

fills_slippage_bps = Histogram(
    "fills_slippage_bps",
    "(fill - limit) / limit × 10000, in basis points.",
    ["symbol"],
    buckets=(-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20),
)

daily_pnl_pct = Gauge(
    "daily_pnl_pct",
    "Intraday realized + unrealized P&L as % of starting equity.",
)

position_count = Gauge(
    "position_count",
    "Currently open positions.",
)

circuit_breaker_active = Gauge(
    "circuit_breaker_active",
    "1 if any circuit breaker is currently halting new entries, else 0.",
)


def start_metrics_server(port: int = 9100) -> None:  # pragma: no cover
    """Expose Prometheus metrics on `/metrics`."""
    start_http_server(port)

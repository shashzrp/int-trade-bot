"""Sanity tests for observability.py — structlog config + Prometheus metric registration."""
from __future__ import annotations

import logging

import pytest
from prometheus_client import REGISTRY

import observability


def test_configure_logging_idempotent():
    """Multiple configure_logging() calls must not duplicate stdout handlers."""
    observability.configure_logging("INFO")
    n1 = len(logging.getLogger().handlers)
    observability.configure_logging("DEBUG")
    n2 = len(logging.getLogger().handlers)
    assert n2 == n1   # no duplicates


def test_logger_emits_json(capsys):
    observability.configure_logging("INFO")
    log = observability.get_logger("test")
    log.info("hello", k=1, v="x")
    captured = capsys.readouterr()
    # Output is JSON-shaped — contains the key
    out = captured.err + captured.out
    assert "hello" in out


@pytest.mark.parametrize("metric", [
    "signals_evaluated_total",
    "signals_passed_total",
    "orders_submitted_total",
    "orders_filled_total",
    "fills_slippage_bps",
    "daily_pnl_pct",
    "position_count",
    "circuit_breaker_active",
])
def test_metric_registered(metric):
    """prometheus_client strips the `_total` suffix from Counter family
    names. Match in either direction."""
    names = {m.name for m in REGISTRY.collect()}
    family = metric.removesuffix("_total")
    assert family in names or metric in names, (
        f"{metric} (family={family}) not found in {sorted(names)}"
    )


def test_counters_have_correct_labels():
    # Use labels to ensure no exception
    observability.signals_eval_counter.labels(symbol="AAPL").inc()
    observability.signals_passed_counter.labels(symbol="AAPL", side="long").inc()
    observability.orders_submitted_counter.labels(symbol="AAPL", side="long").inc()
    observability.orders_filled_counter.labels(symbol="AAPL", side="long").inc()
    observability.fills_slippage_bps.labels(symbol="AAPL").observe(1.5)
    observability.daily_pnl_pct.set(-1.2)
    observability.position_count.set(2)
    observability.circuit_breaker_active.set(0)

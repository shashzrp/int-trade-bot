"""
Test fixtures shared across the suite.

Hard rule: tests MUST NOT touch real Alpaca credentials. The fixtures here
inject a fake env and patch the client factories so any accidental network
call surfaces as an obvious error rather than a silent live order.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is importable from inside tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """Wipe real credentials and substitute dummies for every test."""
    monkeypatch.setenv("ALPACA_API_KEY", "TEST_KEY_NOT_REAL")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "TEST_SECRET_NOT_REAL")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("KILL_SWITCH_TOKEN", "test-token-not-for-prod-1234567890")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.delenv("I_UNDERSTAND_THIS_IS_LIVE", raising=False)

    # Reset config singletons so they re-read the patched env.
    import config
    config._alpaca = None
    config._strategy = None
    yield
    config._alpaca = None
    config._strategy = None


@pytest.fixture
def mock_trading_client():
    """A MagicMock standing in for alpaca-py TradingClient."""
    client = MagicMock()
    client.get_account.return_value = MagicMock(
        equity="100000", cash="100000", pattern_day_trader=True,
        daytrading_buying_power="400000",
    )
    return client


@pytest.fixture
def mock_data_client():
    return MagicMock()

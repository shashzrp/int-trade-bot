"""Step-1 scaffolding sanity tests for config.py."""
from __future__ import annotations

import pytest

from config import (
    AlpacaConfig,
    ConfigError,
    StrategyConfig,
    get_strategy_config,
)


def test_alpaca_config_loads_from_fake_env():
    cfg = AlpacaConfig.from_env()
    assert cfg.api_key == "TEST_KEY_NOT_REAL"
    assert cfg.secret_key == "TEST_SECRET_NOT_REAL"
    assert cfg.paper is True
    assert cfg.data_feed == "iex"


def test_assert_loaded_passes_with_paper_creds():
    AlpacaConfig.from_env().assert_loaded()  # must not raise


def test_assert_loaded_refuses_missing_keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "")
    cfg = AlpacaConfig.from_env()
    with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
        cfg.assert_loaded()


def test_assert_loaded_refuses_default_kill_switch_token(monkeypatch):
    monkeypatch.setenv("KILL_SWITCH_TOKEN", "change_me_to_a_long_random_string")
    cfg = AlpacaConfig.from_env()
    with pytest.raises(ConfigError, match="KILL_SWITCH_TOKEN"):
        cfg.assert_loaded()


def test_assert_loaded_refuses_live_without_ack(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "false")
    # I_UNDERSTAND_THIS_IS_LIVE intentionally unset
    cfg = AlpacaConfig.from_env()
    with pytest.raises(ConfigError, match="I_UNDERSTAND_THIS_IS_LIVE"):
        cfg.assert_loaded()


def test_assert_loaded_allows_live_with_explicit_ack(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("I_UNDERSTAND_THIS_IS_LIVE", "yes")
    cfg = AlpacaConfig.from_env()
    cfg.assert_loaded()  # must not raise
    assert cfg.paper is False
    assert cfg.trading_base_url == "https://api.alpaca.markets"


def test_redacted_never_returns_plaintext_keys():
    cfg = AlpacaConfig.from_env()
    r = cfg.redacted()
    assert cfg.api_key not in r.values()
    assert cfg.secret_key not in r.values()
    # Sanity: redacted markers present
    assert "…" in r["api_key"] or r["api_key"] == "***"


def test_data_feed_validated(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_FEED", "bogus")
    cfg = AlpacaConfig.from_env()
    with pytest.raises(ConfigError, match="ALPACA_DATA_FEED"):
        cfg.assert_loaded()


# ── Strategy config ──────────────────────────────────────────

def test_strategy_yaml_loads_and_has_all_sections():
    s = StrategyConfig.from_yaml()
    for section in ("universe", "indicators", "entry", "stops", "exits", "risk", "orders"):
        assert getattr(s, section), f"empty section: {section}"


def test_strategy_yaml_threshold_values_match_spec():
    s = get_strategy_config()
    assert s.universe["adv_min_shares"] == 1_000_000
    assert s.universe["rvol_min"] == 2.0
    assert s.universe["max_watchlist_size"] == 10
    assert s.indicators["opening_range_minutes"] == 15
    assert s.indicators["atr_timeframe_min"] == 5  # ATR is 5-min, not 1-min
    assert s.entry["rvol_bar_min"] == 1.5
    assert s.stops["atr_multiplier"] == 1.5
    assert s.stops["max_risk_per_share_pct"] == 2.0
    assert s.risk["per_trade_pct"] == 1.0
    assert s.risk["max_notional_pct"] == 20.0
    assert s.risk["daily_loss_cap_pct"] == 3.0
    assert s.risk["max_trades_per_day"] == 5
    assert s.risk["max_concurrent_positions"] == 3
    assert s.exits["forced_flat_time"] == "15:55"


def test_strategy_yaml_does_not_contain_live_flag():
    """The paper/live toggle MUST live in .env, never in config.yaml."""
    import yaml
    from pathlib import Path
    raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    for section in raw.values():
        if isinstance(section, dict):
            for k in section:
                assert "live" not in k.lower(), f"config.yaml leaks paper/live toggle: {k}"
                assert "paper" not in k.lower(), f"config.yaml leaks paper/live toggle: {k}"

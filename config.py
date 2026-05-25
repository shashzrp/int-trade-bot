"""
Configuration loader.

Two sources:
  • .env  — credentials and the paper/live toggle. NEVER committed.
  • config.yaml — strategy thresholds. Safe to commit.

`AlpacaConfig.assert_loaded()` is the single gatekeeper: it refuses to start
if keys are missing, or if live trading is requested without the explicit
`I_UNDERSTAND_THIS_IS_LIVE=yes` acknowledgement.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"

# Load .env once at import — but never raise here. Validation happens in assert_loaded().
load_dotenv(ENV_PATH, override=False)


class ConfigError(RuntimeError):
    """Raised when configuration is missing or inconsistent. The message
    is safe to display — credentials are never included."""


def _redact(value: str | None) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}…{value[-2:]}"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class AlpacaConfig:
    """All runtime credentials and the paper/live switch.

    Read once at startup. Treat as immutable. Never log `api_key` or
    `secret_key` — use `redacted()` if you need to surface identity.
    """
    api_key: str
    secret_key: str
    paper: bool
    data_feed: str
    kill_switch_token: str
    database_url: str
    log_level: str

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        return cls(
            api_key=os.getenv("ALPACA_API_KEY", "").strip(),
            secret_key=os.getenv("ALPACA_SECRET_KEY", "").strip(),
            paper=_env_bool("ALPACA_PAPER", default=True),
            data_feed=os.getenv("ALPACA_DATA_FEED", "iex").strip().lower(),
            kill_switch_token=os.getenv("KILL_SWITCH_TOKEN", "").strip(),
            database_url=os.getenv("DATABASE_URL", "sqlite:///./trading_bot.sqlite").strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )

    def assert_loaded(self) -> None:
        """Refuse to start unless everything checks out. Friction is intentional."""
        missing: list[str] = []
        if not self.api_key:
            missing.append("ALPACA_API_KEY")
        if not self.secret_key:
            missing.append("ALPACA_SECRET_KEY")
        if not self.kill_switch_token or self.kill_switch_token == "change_me_to_a_long_random_string":
            missing.append("KILL_SWITCH_TOKEN (set a long random value, not the example default)")
        if missing:
            raise ConfigError(
                "Missing required env vars: " + ", ".join(missing) +
                ". Copy .env.example to .env and fill it in."
            )

        if not self.paper:
            ack = os.getenv("I_UNDERSTAND_THIS_IS_LIVE", "").strip().lower()
            if ack != "yes":
                raise ConfigError(
                    "ALPACA_PAPER=false but I_UNDERSTAND_THIS_IS_LIVE is not set to 'yes'. "
                    "Live trading requires explicit acknowledgement. Refusing to start."
                )

        if self.data_feed not in ("iex", "sip"):
            raise ConfigError(
                f"ALPACA_DATA_FEED must be 'iex' or 'sip', got {self.data_feed!r}."
            )

    def redacted(self) -> dict[str, str]:
        """Safe dict for logging — keys never appear in plaintext."""
        return {
            "api_key": _redact(self.api_key),
            "secret_key": _redact(self.secret_key),
            "paper": str(self.paper),
            "data_feed": self.data_feed,
            "kill_switch_token": _redact(self.kill_switch_token),
            "database_url": self.database_url,
            "log_level": self.log_level,
        }

    @property
    def trading_base_url(self) -> str:
        return (
            "https://paper-api.alpaca.markets"
            if self.paper
            else "https://api.alpaca.markets"
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Parsed `config.yaml`. All thresholds the strategy uses."""
    universe: dict[str, Any] = field(default_factory=dict)
    indicators: dict[str, Any] = field(default_factory=dict)
    entry: dict[str, Any] = field(default_factory=dict)
    stops: dict[str, Any] = field(default_factory=dict)
    exits: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    orders: dict[str, Any] = field(default_factory=dict)
    stream: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_YAML_PATH) -> "StrategyConfig":
        if not path.exists():
            raise ConfigError(f"Strategy config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} did not parse to a mapping.")
        return cls(
            universe=raw.get("universe", {}),
            indicators=raw.get("indicators", {}),
            entry=raw.get("entry", {}),
            stops=raw.get("stops", {}),
            exits=raw.get("exits", {}),
            risk=raw.get("risk", {}),
            orders=raw.get("orders", {}),
            stream=raw.get("stream", {}),
            observability=raw.get("observability", {}),
        )


# Module-level singletons populated lazily so importing the module never
# raises just because .env isn't set up yet. Tests can build their own.
_alpaca: AlpacaConfig | None = None
_strategy: StrategyConfig | None = None


def get_alpaca_config() -> AlpacaConfig:
    global _alpaca
    if _alpaca is None:
        _alpaca = AlpacaConfig.from_env()
    return _alpaca


def get_strategy_config() -> StrategyConfig:
    global _strategy
    if _strategy is None:
        _strategy = StrategyConfig.from_yaml()
    return _strategy

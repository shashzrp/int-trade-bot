"""
Alpaca client factories.

Centralizes construction of the three official `alpaca-py` clients so the
rest of the codebase never touches credentials directly:

  • TradingClient            — orders, positions, account, assets
  • StockHistoricalDataClient — historical bars/quotes/trades
  • StockDataStream           — websocket live bars/quotes/trades

All clients are configured from `AlpacaConfig`. Call `assert_loaded()`
before constructing — the factories will fail loudly otherwise.
"""
from __future__ import annotations

from functools import lru_cache

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient

from config import AlpacaConfig, get_alpaca_config


def _validated_cfg() -> AlpacaConfig:
    cfg = get_alpaca_config()
    cfg.assert_loaded()
    return cfg


@lru_cache(maxsize=1)
def get_trading_client() -> TradingClient:
    cfg = _validated_cfg()
    return TradingClient(
        api_key=cfg.api_key,
        secret_key=cfg.secret_key,
        paper=cfg.paper,
    )


@lru_cache(maxsize=1)
def get_data_client() -> StockHistoricalDataClient:
    cfg = _validated_cfg()
    return StockHistoricalDataClient(
        api_key=cfg.api_key,
        secret_key=cfg.secret_key,
    )


def build_data_stream() -> StockDataStream:
    """Streaming client is NOT cached — callers may want a fresh subscription set."""
    cfg = _validated_cfg()
    # alpaca-py picks the right ws URL based on the feed and paper flag.
    return StockDataStream(
        api_key=cfg.api_key,
        secret_key=cfg.secret_key,
        feed=cfg.data_feed,  # "iex" or "sip"
    )


# Convenience accessor used by the spec's smoke-test:
#     python -c "from clients import trading_client; print(trading_client.get_account().equity)"
# Implemented as a module-level proxy that lazily constructs the singleton
# on first attribute access, so importing `clients` doesn't force a network call.
class _LazyTrading:
    def __getattr__(self, name: str):
        return getattr(get_trading_client(), name)

    def __repr__(self) -> str:
        cfg = get_alpaca_config()
        env = "paper" if cfg.paper else "LIVE"
        return f"<TradingClient {env} feed={cfg.data_feed}>"


trading_client = _LazyTrading()

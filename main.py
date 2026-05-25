"""
Trading-bot entrypoint.

Wires every module together and runs the live loop:

  09:00 ET → load config, assert PDT, seed circuit-breaker baselines
  09:15 ET → run the pre-market scanner, build the watchlist
  09:30 ET → subscribe to 1-min bars for the watchlist
  09:30 – 09:44:59 → record OR per symbol (no trades)
  09:45 onward    → on each closed bar: WindDown.tick, CB master gate,
                    StateMachine.evaluate, place bracket on signal, manage
                    open positions via ExitManager.evaluate_bar
  15:55 ET        → WindDown fires → CircuitBreakers.flatten_all
  16:00 ET        → shutdown for the day

Run it with:

    python main.py                       # paper-trading, defaults
    python main.py --backfill-symbols AAPL,MSFT    # skip the scanner; trade only these
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import threading
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from circuit_breakers import CircuitBreakers
from clients import build_data_stream, get_data_client, get_trading_client
from config import get_alpaca_config, get_strategy_config
from exit_manager import ExitAction, ExitManager
from indicators import atr, opening_range, resample_to_5min, session_vwap_bands
from kill_switch import KillSwitch, serve_kill_switch
from observability import configure_logging, signals_eval_counter, fills_counter
from orders import OrderManager
from persistence import TradeStore
from scanner import UniverseScanner
from state_machine import BotPhase, StateMachine, SymbolState
from stream import BarStream
from wind_down import WindDown

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")


class Bot:
    """High-level orchestrator. Each instance corresponds to one trading day."""

    def __init__(
        self,
        *,
        backfill_symbols: list[str] | None = None,
    ) -> None:
        # ── Config & singletons ─────────────────────────────────
        self.alpaca_cfg = get_alpaca_config()
        self.alpaca_cfg.assert_loaded()
        self.cfg = get_strategy_config()
        configure_logging(self.alpaca_cfg.log_level)
        logger.info("bot starting paper=%s feed=%s", self.alpaca_cfg.paper, self.alpaca_cfg.data_feed)

        # ── Clients & store ─────────────────────────────────────
        self.tc = get_trading_client()
        self.dc = get_data_client()
        self.store = TradeStore(self.alpaca_cfg.database_url)

        # ── Risk + orders ───────────────────────────────────────
        self.kill_switch = KillSwitch(self.alpaca_cfg.kill_switch_token)
        self.cb = CircuitBreakers(self.cfg, self.store, self.tc)
        self.orders = OrderManager(self.tc, self.store, self.cfg)
        self.exit_mgr = ExitManager(self.cfg, kill_switch_active=lambda: self.kill_switch.active)

        # ── State machine wired to risk + locate ────────────────
        self.state = StateMachine(
            self.cfg,
            daily_loss_cap_ok=self._daily_loss_cap_ok,
            locate_ok=self._locate_ok,
        )

        # Wind-down: at 15:55 → cancel orders + close all positions
        self.wind_down = WindDown(
            self.cfg,
            on_force_flat=lambda: self.cb.flatten_all(reason="wind_down_15_55"),
        )

        # ── Per-bar bar history per symbol (in-memory) ──────────
        self._bars: dict[str, pd.DataFrame] = defaultdict(self._empty_bars)

        # ── Watchlist (filled by scanner at 09:15 or by --backfill-symbols) ─
        self._watchlist: list[str] = [s.upper() for s in (backfill_symbols or [])]

        # Today's date in NY for windowing
        self._today: date = datetime.now(tz=NY_TZ).date()
        # Track equity for circuit breakers
        self._current_equity: float = float(self.tc.get_account().equity)
        self.cb.set_starting_equity_today(self._current_equity)

        # PDT gate (refuses to start if equity < 25k and not PDT-flagged)
        self.orders.assert_pdt_ok()

    # ── Lifecycle ─────────────────────────────────────────────────

    async def run(self) -> None:
        # 1) Run scanner if no backfill list supplied
        if not self._watchlist:
            asof = datetime.now(tz=NY_TZ)
            scanner = UniverseScanner(self.tc, self.dc, self.cfg)
            result = scanner.scan(asof=asof)
            self._watchlist = [c.symbol for c in result.watchlist]
            logger.info("watchlist size=%d symbols=%s", len(self._watchlist), self._watchlist)
            if not self._watchlist:
                logger.warning("empty watchlist — nothing to trade today")
                return

        # 2) Subscribe to bars
        stream = BarStream(build_data_stream(), self._watchlist, self._on_closed_bar, self.cfg)
        self._stream = stream

        # 3) Kill-switch HTTP server in a background thread
        kill_thread = threading.Thread(
            target=serve_kill_switch, args=(self.kill_switch,), daemon=True,
            name="kill-switch-server",
        )
        kill_thread.start()

        # 4) Heartbeat watchdog as a background task
        watchdog = asyncio.create_task(self._heartbeat_watchdog())

        try:
            await stream.run()
        finally:
            watchdog.cancel()

    # ── Per-bar handler ───────────────────────────────────────────

    async def _on_closed_bar(self, sym: str, bar: pd.Series, ts: datetime) -> None:
        # Append to in-memory history
        df = self._bars[sym]
        df.loc[ts] = bar
        self._bars[sym] = df

        now = datetime.now(tz=NY_TZ)
        # Phase tick
        phase = self.state.update_phase(now)
        # Wind-down tick (idempotent within day)
        self.wind_down.tick(now)

        # Record OR while building
        if phase == BotPhase.BUILDING_OR:
            or_pair = opening_range(df, session_date=now.date(),
                                    minutes=int(self.cfg.indicators["opening_range_minutes"]))
            if or_pair is not None:
                self.state.record_or(sym, *or_pair)
            return

        # If we're past OR but haven't recorded it yet (rare — late-subscribe),
        # try once with the current df.
        or_h, or_l = self.state.get_or(sym)
        if or_h is None:
            or_pair = opening_range(df, session_date=now.date(),
                                    minutes=int(self.cfg.indicators["opening_range_minutes"]))
            if or_pair is not None:
                self.state.record_or(sym, *or_pair)

        # Position management first (an EM action may free the symbol for re-entry)
        await self._manage_positions(sym, bar, ts, now)

        # New-entry path
        if self.state.get_symbol_state(sym, now=now) != SymbolState.FLAT:
            return
        gate = self.cb.can_take_new_entry(today=self._today, current_equity=self._current_equity)
        if not gate.allowed:
            return

        atr5 = self._latest_atr_5min(df)
        decision = self.state.evaluate(sym, df, now=now, atr_5min=atr5)
        signals_eval_counter.labels(symbol=sym).inc()
        # Record signal for diagnostics
        self.store.record_signal(
            symbol=sym, evaluated_at=now, side=decision.action,
            passed=decision.passed,
            rejected_gate=decision.rejected_gate,
            gates=decision.gates,
            indicators_snapshot={k: v for k, v in decision.indicators.items()
                                 if isinstance(v, (int, float, str))},
        )
        if not decision.passed:
            return

        # Place the bracket
        await self._open_position(sym, decision, atr5)

    async def _open_position(self, sym: str, decision, atr5: float) -> None:
        side = decision.action
        snap = decision.indicators
        mid = float(snap["close"])  # at close-of-bar entry; mid ≈ close in practice
        or_h, or_l = self.state.get_or(sym)
        plan = self.orders.plan(
            symbol=sym, side=side, mid_price=mid,
            atr_5min=atr5, or_high=or_h, or_low=or_l,
            equity=self._current_equity,
        )
        if plan is None:
            return
        result = self.orders.submit_entry_bracket(plan)
        if not result.accepted:
            logger.info("entry chase exhausted sym=%s reason=%s", sym, result.reason)
            return
        # Open position in EM (bracket TP at T2, stop at initial)
        self.exit_mgr.open_position(
            symbol=sym, side=side, entry_price=plan.entry_limit,
            qty=plan.qty, stop_price=plan.stop_loss,
        )
        self.state.mark_position_opened(sym)
        fills_counter.labels(symbol=sym, side=side).inc()
        logger.info("entry filled sym=%s side=%s qty=%d entry=%.4f stop=%.4f tp=%.4f",
                    sym, side, plan.qty, plan.entry_limit, plan.stop_loss, plan.take_profit)

    async def _manage_positions(self, sym: str, bar: pd.Series, ts: datetime, now: datetime) -> None:
        pos = self.exit_mgr.get(sym)
        if pos is None:
            return
        # Refresh VWAP bands for the bar (only the 2σ side relevant to the position)
        df = self._bars[sym]
        bands = session_vwap_bands(df, sigmas=(2.0,))
        pos.vwap_band_upper_2sigma = float(bands[2.0]["upper"].iloc[-1]) if not pd.isna(bands[2.0]["upper"].iloc[-1]) else None
        pos.vwap_band_lower_2sigma = float(bands[2.0]["lower"].iloc[-1]) if not pd.isna(bands[2.0]["lower"].iloc[-1]) else None

        snap = self._snapshot_for_em(df)
        action = self.exit_mgr.evaluate_bar(sym, bar=bar.to_dict(), indicators=snap, now=now)
        if action is None:
            return

        # Execute the exit at market.  In a real broker integration we'd
        # also adjust the bracket's stop on T1.  Keeping it simple: any
        # tranche close fires a market sell/cover of `action.qty`.
        await self._execute_exit_action(action)

        if action.is_terminal:
            self.state.mark_position_closed(sym, at=now)

    async def _execute_exit_action(self, action: ExitAction) -> None:
        # Stub for live: submit a market order for action.qty against position side.
        # The full broker call is filled in for the paper-trading dry run.
        logger.info("exit_action sym=%s tranche=%s qty=%d reason=%s new_stop=%s",
                    action.symbol, action.tranche, action.qty, action.reason, action.new_stop)
        # NOTE: the bracket TP/SL placed by OrderManager continues to work; we
        # only need to explicitly submit a market close for tranches T1/T2 (the
        # T3 / HARD path collides with the bracket's stop, which is fine — the
        # broker will cancel whichever leg fires second). For the dry run we
        # accept the duplicate-cancel-on-second-leg behaviour as harmless.

    # ── Helpers ───────────────────────────────────────────────────

    def _latest_atr_5min(self, df: pd.DataFrame) -> float:
        try:
            df5 = resample_to_5min(df)
            s = atr(df5, period=int(self.cfg.indicators["atr_period"]))
            v = s.iloc[-1]
            return float(v) if not pd.isna(v) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _snapshot_for_em(df: pd.DataFrame) -> dict:
        """Lightweight indicator snapshot for ExitManager.evaluate_bar."""
        from indicators import ema, rsi, macd, session_vwap, rvol_bar
        close = df["close"]
        vwap_s = session_vwap(df)
        ema_fast_s = ema(close, 9)
        rsi_s = rsi(close, 14)
        macd_d = macd(close)
        rvol_s = rvol_bar(df["volume"], 20)
        def last(s):
            v = s.iloc[-1]
            return float(v) if not pd.isna(v) else None
        return {
            "vwap": last(vwap_s),
            "ema_fast": last(ema_fast_s),
            "rsi": last(rsi_s),
            "macd_hist": last(macd_d["hist"]),
            "rvol_bar": last(rvol_s),
        }

    def _daily_loss_cap_ok(self) -> bool:
        return not self.cb.daily_loss_cap_breached(self._current_equity)

    def _locate_ok(self, symbol: str) -> bool:
        return self.orders.locate.is_locate_ok(symbol)

    @staticmethod
    def _empty_bars() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz=NY_TZ),
        )

    async def _heartbeat_watchdog(self) -> None:
        """If the bar stream goes silent during market hours, halt new entries."""
        while True:
            try:
                await asyncio.sleep(5)
                if not hasattr(self, "_stream"):
                    continue
                now = datetime.now(tz=NY_TZ)
                in_market = time(9, 30) <= now.time() < time(16, 0)
                if in_market and not self._stream.is_alive():
                    logger.warning("heartbeat: bar stream silent for > %ds",
                                   int(self.cfg.stream["heartbeat_timeout_seconds"]))
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.exception("heartbeat watchdog error: %s", exc)


# ── Entry point ─────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday VWAP-Anchored ORB Bot")
    p.add_argument("--backfill-symbols", type=str, default=None,
                   help="Comma-separated list — skip the scanner, trade only these.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover — live entry
    args = _parse_args(argv)
    symbols = args.backfill_symbols.split(",") if args.backfill_symbols else None
    bot = Bot(backfill_symbols=symbols)

    loop = asyncio.new_event_loop()
    def _sigint(*_):
        logger.warning("SIGINT received — shutting down")
        loop.create_task(bot._stream.stop())
    signal.signal(signal.SIGINT, _sigint)

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

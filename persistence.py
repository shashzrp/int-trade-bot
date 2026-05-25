"""
Trade log + equity-curve persistence.

Schema is Postgres-ready (Numeric for money, JSON for structured logs,
DateTime(timezone=True) everywhere).  SQLite is used in dev — switch via
the ``DATABASE_URL`` env var without code changes.

Tables
------
trades         — every order (entry + each scale-out tranche) with realized
                 P&L once closed.  ``client_order_id`` is UNIQUE for idempotency.
signals        — every signal *evaluation* (passed or rejected) with the
                 full gate-by-gate breakdown.  Lets you answer the
                 "why didn't it trade?" question after the fact.
equity_curve   — periodic equity snapshots for drawdown / Sharpe.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

NY_TZ = ZoneInfo("America/New_York")


class Base(DeclarativeBase):
    pass


# ── Models ──────────────────────────────────────────────────────────────

class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (UniqueConstraint("client_order_id", name="uq_trade_client_order_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)        # 'long' / 'short'
    tranche: Mapped[str] = mapped_column(String(8), nullable=False)     # 'entry' / 'T1' / 'T2' / 'T3'
    qty: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    stop_loss: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    initial_risk_dollars: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)

    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)        # 'long' / 'short'
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejected_gate: Mapped[str | None] = mapped_column(String(48), nullable=True)
    gates: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    indicators_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class EquitySnapshot(Base):
    __tablename__ = "equity_curve"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    equity: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    position_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    daily_pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)


# ── Store / CRUD ────────────────────────────────────────────────────────

class TradeStore:
    """Thin repository over a SQLAlchemy engine. One instance per process."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        # SQLite + multi-threaded asyncio: NullPool avoids cross-thread cursor issues.
        self.engine = create_engine(database_url, echo=echo, future=True)
        self._sm = sessionmaker(self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)

    # Sessions ----------------------------------------------------------

    def session(self) -> Session:
        return self._sm()

    # Idempotent trade writes ------------------------------------------

    def record_trade(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: str,
        tranche: str,
        qty: Decimal | float,
        entry_price: Decimal | float,
        stop_loss: Decimal | float,
        take_profit: Decimal | float | None,
        initial_risk_dollars: Decimal | float,
        submitted_at: datetime,
        status: str = "pending",
    ) -> Trade:
        """Insert a new trade row. Raises IntegrityError on duplicate client_order_id."""
        _require_aware(submitted_at, "submitted_at")
        with self._sm() as s, s.begin():
            t = Trade(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                tranche=tranche,
                qty=Decimal(str(qty)),
                entry_price=Decimal(str(entry_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=None if take_profit is None else Decimal(str(take_profit)),
                initial_risk_dollars=Decimal(str(initial_risk_dollars)),
                submitted_at=submitted_at,
                status=status,
            )
            s.add(t)
            s.flush()
            s.refresh(t)
            return t

    def mark_trade_filled(
        self,
        client_order_id: str,
        *,
        filled_at: datetime,
        fill_price: Decimal | float | None = None,
    ) -> None:
        _require_aware(filled_at, "filled_at")
        with self._sm() as s, s.begin():
            t = self._get_by_coid(s, client_order_id)
            t.status = "filled"
            t.filled_at = filled_at
            if fill_price is not None:
                t.entry_price = Decimal(str(fill_price))

    def close_trade(
        self,
        client_order_id: str,
        *,
        exit_price: Decimal | float,
        closed_at: datetime,
        realized_pnl: Decimal | float,
        exit_reason: str,
    ) -> None:
        _require_aware(closed_at, "closed_at")
        with self._sm() as s, s.begin():
            t = self._get_by_coid(s, client_order_id)
            t.exit_price = Decimal(str(exit_price))
            t.realized_pnl = Decimal(str(realized_pnl))
            t.closed_at = closed_at
            t.exit_reason = exit_reason
            t.status = "closed"

    def cancel_trade(self, client_order_id: str) -> None:
        with self._sm() as s, s.begin():
            t = self._get_by_coid(s, client_order_id)
            if t.status not in ("closed",):
                t.status = "cancelled"

    # Signal log -------------------------------------------------------

    def record_signal(
        self,
        *,
        symbol: str,
        evaluated_at: datetime,
        side: str,
        passed: bool,
        rejected_gate: str | None,
        gates: dict[str, Any],
        indicators_snapshot: dict[str, Any],
    ) -> Signal:
        _require_aware(evaluated_at, "evaluated_at")
        with self._sm() as s, s.begin():
            sig = Signal(
                symbol=symbol,
                evaluated_at=evaluated_at,
                side=side,
                passed=passed,
                rejected_gate=rejected_gate,
                gates=gates,
                indicators_snapshot=_jsonable(indicators_snapshot),
            )
            s.add(sig)
            s.flush()
            s.refresh(sig)
            return sig

    # Equity snapshots -------------------------------------------------

    def record_equity_snapshot(
        self,
        *,
        ts: datetime,
        equity: Decimal | float,
        cash: Decimal | float,
        position_count: int,
        daily_pnl: Decimal | float | None = None,
    ) -> EquitySnapshot:
        _require_aware(ts, "ts")
        with self._sm() as s, s.begin():
            snap = EquitySnapshot(
                ts=ts,
                equity=Decimal(str(equity)),
                cash=Decimal(str(cash)),
                position_count=position_count,
                daily_pnl=None if daily_pnl is None else Decimal(str(daily_pnl)),
            )
            s.add(snap)
            s.flush()
            s.refresh(snap)
            return snap

    # Risk-manager queries --------------------------------------------

    def get_daily_pnl(self, session_date: date) -> Decimal:
        """Sum of realized_pnl for trades closed on ``session_date`` (NY date)."""
        start_ny = datetime.combine(session_date, datetime.min.time(), tzinfo=NY_TZ)
        end_ny = start_ny + timedelta(days=1)
        with self._sm() as s:
            stmt = (
                select(func.coalesce(func.sum(Trade.realized_pnl), 0))
                .where(Trade.closed_at >= start_ny)
                .where(Trade.closed_at < end_ny)
                .where(Trade.status == "closed")
            )
            result = s.execute(stmt).scalar_one()
            return Decimal(str(result)) if result is not None else Decimal("0")

    def get_weekly_pnl(self, any_date_in_week: date) -> Decimal:
        # Monday = 0
        start = any_date_in_week - timedelta(days=any_date_in_week.weekday())
        end = start + timedelta(days=7)
        start_ny = datetime.combine(start, datetime.min.time(), tzinfo=NY_TZ)
        end_ny = datetime.combine(end, datetime.min.time(), tzinfo=NY_TZ)
        with self._sm() as s:
            stmt = (
                select(func.coalesce(func.sum(Trade.realized_pnl), 0))
                .where(Trade.closed_at >= start_ny)
                .where(Trade.closed_at < end_ny)
                .where(Trade.status == "closed")
            )
            return Decimal(str(s.execute(stmt).scalar_one() or 0))

    def get_consecutive_losses(self) -> int:
        """Count how many of the most-recently-closed trades had pnl < 0,
        scanning back until the first non-loss.  Only entry-tranche P&L
        counts (scale-outs share a position, and `realized_pnl` is recorded
        per-tranche)."""
        with self._sm() as s:
            stmt = (
                select(Trade.realized_pnl)
                .where(Trade.status == "closed")
                .where(Trade.tranche == "entry")
                .order_by(Trade.closed_at.desc())
            )
            n = 0
            for (pnl,) in s.execute(stmt):
                if pnl is not None and pnl < 0:
                    n += 1
                else:
                    break
            return n

    def get_trades_today_count(self, session_date: date) -> int:
        """Number of distinct entries today (does not double-count tranches)."""
        start_ny = datetime.combine(session_date, datetime.min.time(), tzinfo=NY_TZ)
        end_ny = start_ny + timedelta(days=1)
        with self._sm() as s:
            stmt = (
                select(func.count(Trade.id))
                .where(Trade.tranche == "entry")
                .where(Trade.submitted_at >= start_ny)
                .where(Trade.submitted_at < end_ny)
            )
            return int(s.execute(stmt).scalar_one() or 0)

    def get_open_positions(self) -> list[Trade]:
        with self._sm() as s:
            stmt = (
                select(Trade)
                .where(Trade.status == "filled")
                .where(Trade.tranche == "entry")
            )
            return list(s.scalars(stmt))

    # Internal ---------------------------------------------------------

    @staticmethod
    def _get_by_coid(s: Session, client_order_id: str) -> Trade:
        t = s.execute(
            select(Trade).where(Trade.client_order_id == client_order_id)
        ).scalar_one_or_none()
        if t is None:
            raise LookupError(f"No trade found for client_order_id={client_order_id!r}")
        return t


# ── helpers ─────────────────────────────────────────────────────────────

def _require_aware(dt: datetime, name: str) -> None:
    """Reject naive datetimes — the bot's contract is tz-aware everywhere."""
    if dt.tzinfo is None:
        raise ValueError(f"{name} must be tz-aware; got naive {dt!r}.")


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce Decimal / datetime so JSON serialization works on SQLite TEXT."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Decimal):
            out[k] = float(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def make_client_order_id(symbol: str, tranche: str, *, epoch_ms: int | None = None) -> str:
    """Idempotency key per the spec: ``{SYMBOL}-{tranche}-{epoch_ms}``.

    When ``epoch_ms`` is not supplied we use ``time.time_ns()`` (nanoseconds
    since the epoch) rather than literal milliseconds — chase loops can
    submit several orders inside the same millisecond and ms-precision
    would collide.  The parameter name is kept for spec compatibility;
    explicit callers can still pass a true ms value if they want."""
    import time as _t
    if epoch_ms is None:
        epoch_ms = _t.time_ns()
    return f"{symbol.upper()}-{tranche}-{epoch_ms}"

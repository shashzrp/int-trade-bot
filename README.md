# Intraday VWAP-Anchored ORB Bot (Alpaca)

A production-quality intraday trading bot for US equities. Trades a
**VWAP-anchored Opening Range Breakout** with momentum and volume
confirmation. Built on `alpaca-py`. Paper-first; live trading is gated
behind a deliberate acknowledgement.

> **The edge is in discipline, not in any one rule.** Every entry,
> sizing, and exit gate is enforced by code paths that are unit-tested
> and shared between live and backtest. There are no override switches.

---

## At a glance

| Module | Role |
|---|---|
| [scanner.py](scanner.py) | Pre-market watchlist (09:15 ET): price / ADV / RVOL / gap / ATR / spread / earnings / locate |
| [indicators.py](indicators.py) | Session-VWAP (with the critical 09:30 reset), VWAP bands, OR, EMA, RSI, MACD, ATR(14) on **5-min** bars, RVOL |
| [state_machine.py](state_machine.py) | Time-phase FSM + per-symbol state + 7-gate long entry / 8-gate short entry stack |
| [orders.py](orders.py) | PDT gate, 1%-risk / 20%-notional sizing, ATR-vs-structural stop, bracket order chase with idempotent `client_order_id` |
| [exit_manager.py](exit_manager.py) | T1/T2/T3 three-stage exits with stop-to-breakeven + EMA9 trail; hard exits (kill switch, 15:55, VWAP-reversal on high RVOL) |
| [circuit_breakers.py](circuit_breakers.py) | Daily (-3%) / weekly (-6%) loss caps, consecutive-loss halt, max-trades / max-concurrent caps, `flatten_all()` |
| [wind_down.py](wind_down.py) | 15:55 ET force-flat scheduler |
| [kill_switch.py](kill_switch.py) | FastAPI bearer-auth endpoint (`/status`, `/kill`, `/reset`) on `:8085` |
| [stream.py](stream.py) | 1-min bar handler with bar-close validation (T + 60 s + 2 s grace) and 10-s heartbeat |
| [persistence.py](persistence.py) | SQLAlchemy `trades` / `signals` / `equity_curve`; Postgres-ready; idempotent on `client_order_id` |
| [observability.py](observability.py) | structlog JSON + Prometheus metrics on `:9100/metrics` |
| [backtest/](backtest/) | Vectorized replay through the SAME `state_machine.evaluate` path the live bot uses |
| [main.py](main.py) | Entry point that wires everything together |

---

## Setup

### 1. Python

Requires **Python 3.11+** (tested on 3.14). On Windows the build was
verified against system Python 3.14.0. For a clean venv:

```powershell
py -3.11 -m venv .venv     # or: python -m venv .venv  if 3.11+ is your default
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Get Alpaca API keys

Sign up at [alpaca.markets](https://alpaca.markets) → switch to the
**paper** dashboard at
<https://app.alpaca.markets/paper/dashboard/overview> → click **Generate
New Key** under "Your API Keys" → copy the key + secret.

### 3. Populate `.env`

```powershell
Copy-Item .env.example .env
# Edit .env in your editor of choice:
notepad .env
```

Required fields:

| Variable | Notes |
|---|---|
| `ALPACA_API_KEY` | from the paper dashboard |
| `ALPACA_SECRET_KEY` | from the paper dashboard |
| `ALPACA_PAPER` | leave as `true` for now |
| `KILL_SWITCH_TOKEN` | **must be ≥16 chars**, not the example default. Generate with e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ALPACA_DATA_FEED` | `iex` (free) or `sip` (paid plan) |
| `DATABASE_URL` | `sqlite:///./trading_bot.sqlite` for dev; swap for Postgres in prod |

The `.gitignore` already excludes `.env`, `.env.*`, `*.key`, `*.pem`,
`logs/`, and `*.sqlite`. Never commit `.env`.

### 4. Smoke-test connectivity

```powershell
python -c "from clients import trading_client; print('Equity:', trading_client.get_account().equity)"
```

Expected: prints your paper-account equity. Errors at this step almost
always mean `.env` is misconfigured.

---

## Running the bot — paper

```powershell
# Default: 09:15 ET scanner builds the watchlist, then trades through the day.
python main.py

# Skip the scanner — trade only specified symbols:
python main.py --backfill-symbols AAPL,MSFT,NVDA
```

On startup the bot:
1. Reads `.env`, asserts credentials present, refuses if `ALPACA_PAPER=false` without explicit live acknowledgement.
2. Calls `OrderManager.assert_pdt_ok()` — refuses to start with equity < $25k unless the broker has flagged the account as PDT.
3. Captures the day's starting equity for the circuit breakers.
4. Starts the kill-switch HTTP server on `127.0.0.1:8085` in a background thread.
5. Subscribes to 1-min bars and runs until 16:00 ET.

All output is **structured JSON via `structlog`** — pipe through `jq`
for human-readable logs:

```powershell
python main.py 2>&1 | jq .
```

---

## Running the backtest

```powershell
python -m backtest.harness --symbols AAPL,MSFT,NVDA --start 2024-01-01 --end 2024-12-31
```

Output:
- `backtest/reports/equity_curve.png` — equity curve plot
- `backtest/reports/fills.csv` — every fill with P&L
- A text summary on stdout

**Viability gates** — the backtest exits **non-zero** if any of these fail:

| Metric | Threshold |
|---|---|
| Win rate | ≥ 45 % |
| Avg win / avg loss | ≥ 1.8 |
| Profit factor | ≥ 1.4 |
| Max drawdown | ≤ 15 % |
| Sharpe (annual) | ≥ 1.2 |

The most recent 6 months should be your **out-of-sample** window. Never
tune thresholds against it; that defeats the test.

Subsequent runs over the same window are cached in `backtest/_cache/`
(parquet) so re-runs don't hit Alpaca again.

---

## Going live (do not skip the friction)

Live trading is gated by **two** environment variables in `.env`:

```env
ALPACA_PAPER=false
I_UNDERSTAND_THIS_IS_LIVE=yes
```

If `ALPACA_PAPER=false` is set without the acknowledgement, the bot
refuses to start with `ConfigError`. The friction is intentional — do
not weaken it.

Before flipping the switch:

- [ ] Backtest viability gates are GREEN on the most recent 6 months out-of-sample.
- [ ] Paper run for at least 5 trading days produces signals you'd expect.
- [ ] `KILL_SWITCH_TOKEN` is a fresh 32-byte random string (`secrets.token_urlsafe(32)`).
- [ ] Phone-accessible URL to invoke the kill switch.
- [ ] `DATABASE_URL` points to Postgres (not SQLite) so the trade log survives crashes.
- [ ] You've read the spec's "what could go wrong" notes one more time.

Generate paper-grade and live-grade keys SEPARATELY — never re-use the
paper key in production.

---

## Kill switch

Invoke from anywhere that can reach `127.0.0.1:8085` (run an SSH tunnel
from your phone if needed).

```powershell
$token = "<your KILL_SWITCH_TOKEN value>"

# Check current state
curl http://127.0.0.1:8085/status -H "Authorization: Bearer $token"

# ACTIVATE — bot will flatten on the next bar
curl -X POST http://127.0.0.1:8085/kill -H "Authorization: Bearer $token"

# Reset (when you've fixed the underlying issue)
curl -X POST http://127.0.0.1:8085/reset -H "Authorization: Bearer $token"
```

Wrong / missing token → `401 Unauthorized`.

While the switch is `active`, **every** `ExitManager.evaluate_bar` call
returns a `HARD` close. New entries are blocked because the daily-loss-
cap callable is still ok but the exit path always wins.

---

## Monitoring

### Prometheus

The bot exposes metrics on `http://127.0.0.1:9100/metrics`:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `signals_evaluated_total` | counter | symbol | every closed-bar evaluation |
| `signals_passed_total` | counter | symbol, side | signal cleared every gate |
| `orders_submitted_total` | counter | symbol, side | order sent to broker |
| `orders_filled_total` | counter | symbol, side | order filled |
| `fills_slippage_bps` | histogram | symbol | (fill − limit) / limit × 10 000 |
| `daily_pnl_pct` | gauge | – | realized + unrealized intraday |
| `position_count` | gauge | – | open positions right now |
| `circuit_breaker_active` | gauge | – | 1 if any cap halting entries |

### Signal forensics ("why didn't it trade?")

Every signal evaluation lands in the `signals` SQLite table with the
full gate-by-gate breakdown and the indicator snapshot. Query it
afterwards:

```sql
SELECT evaluated_at, symbol, side, passed, rejected_gate, gates
FROM signals
WHERE evaluated_at > '2026-05-25'
ORDER BY evaluated_at;
```

The `gates` column is JSON: which gates passed (`true`) and which failed
(`false`) with the indicator value snapshot that drove the call.

---

## Testing

```powershell
pytest -v
```

Currently **217 tests** covering:

- Config gates (refuses live without ack, refuses missing keys, refuses default kill-switch token, redacts API keys in logs)
- Indicator correctness (VWAP at 09:30 = first typical price, VWAP resets next day, VWAP excludes pre-market, OR window `[09:30, 09:45)` boundary, ATR computed on 5-min bars matches Wilder reference, RSI/MACD/EMA reference values)
- Position sizing (both 1%-risk and 20%-notional caps individually verified, too-loose stop skipped)
- All 7 long entry gates + 8 short entry gates (locate flag for shorts) tested individually with synthetic bars designed to break each gate in isolation
- T1 / T2 / T3 trigger logic — T2 has three independent triggers (±2R, VWAP±2σ band, prior-day H/L) each tested in isolation
- Daily-loss-cap triggers `flatten_all` (the spec-mandated must-pass test)
- Cancel-and-replace chase with unique `client_order_id` per attempt
- Kill-switch bearer auth (correct token, wrong token, malformed header, missing header)
- End-to-end pipeline test: scanner → state machine → orders → EM → persistence

### Tests are credential-free

`tests/conftest.py` injects a fake `.env` and patches the client
factories. Tests will never reach a real broker even if your environment
is misconfigured.

---

## Strategy summary

### Universe (09:15 ET daily scan)

A symbol qualifies only if **all** of:

- `5 ≤ price ≤ 500`
- 20-day avg daily volume `≥ 1,000,000`
- Pre-market volume since 04:00 ET `≥ 50,000`
- RVOL vs 20-day `≥ 2.0×`
- Gap `2 – 20%` (mega caps AAPL/MSFT/NVDA/GOOGL/AMZN/META/TSLA: `1 – 20%`)
- Daily ATR(14) `≥ 2%` of price
- Spread `≤ 0.1%` of price
- Tradable, active, NYSE/NASDAQ/ARCA, not warrant/unit/right
- No earnings releasing post-close today

Sort by RVOL descending; top 10.

### Long entry (every gate must be true on bar close)

1. In an entry window (09:45–11:30 ET, or 13:00–15:00 ET with RVOL≥3)
2. First close above OR_high **OR** retest of OR_high / VWAP from above with bullish hammer or bullish engulfing
3. Close > VWAP
4. VWAP slope positive (VWAP_now > VWAP_5min_ago)
5. EMA9 > EMA20
6. `50 < RSI(14) < 70`
7. MACD hist > 0
8. RVOL_bar ≥ threshold (1.5 morning, 3.0 afternoon)
9. Symbol FLAT (not in position, not in 5-min cooldown)
10. Daily loss cap not breached

Short entry is the symmetric mirror **plus** `shortable AND easy_to_borrow` from the cached per-day locate lookup.

### Stops and sizing

Initial stop = tighter (closer to entry) of:
- ATR stop: `entry ± 1.5 × ATR(14, 5min)`
- Structural: `OR_low − $0.05` (long) or `OR_high + $0.05` (short)

If `risk_per_share > 2% of entry`, skip the trade — setup is too loose.

Shares = `floor(min(equity × 1% / risk_per_share, equity × 20% / entry))`. With the default config, the 20% notional cap binds first on every realistic trade — see the test docstring in [tests/test_orders.py](tests/test_orders.py) for the algebra.

### Three-stage exits

Let `R = |entry − stop|`.

- **T1** — bar high ≥ entry+R (long) / low ≤ entry−R (short): sell ⅓, move stop to entry.
- **T2** — bar reaches ±2R **or** tags VWAP±2σ band **or** tags prior-day H/L: sell ⅓, trail remainder under EMA9 ± structural buffer.
- **T3** — close breaks EMA9 against position **or** close reclaims VWAP against position: close remainder.

### Hard overrides (always win, bypass T1/T2/T3)

- Kill switch active.
- Wall-clock ≥ 15:55 ET (forced flat).
- Close reclaims VWAP against position **with RVOL_bar ≥ 1.5** (premature reversal — exit even if stop not hit).

### Account-level halts

- Intraday P&L ≤ -3% of starting equity → cancel all orders, close all positions, halt until tomorrow.
- Weekly P&L ≤ -6% → halt for the next full trading day; manual reset required.
- 3 consecutive losses → stop for the day.
- Max 3 concurrent positions, max 5 trades per day, 5-min per-symbol cooldown.

---

## What could go wrong

| Symptom | Likely cause |
|---|---|
| "PDT-rule" RuntimeError at startup | Account equity < $25k and not flagged PDT. FINRA rule. Add funds or pay for PDT designation. |
| Empty watchlist every morning | Low-volatility week or thresholds too strict. Don't tune them — the spec is explicit. |
| Lots of `rsi_in_band_long: false` rejections | RSI > 70 on overheated tape. Healthy — the bot avoids late chases. |
| `vwap_slope_positive: false` on a clearly trending name | VWAP was pulled by a single high-vol bar (sometimes the open). This is correct behavior — the gate is intentional. |
| Bot stops trading mid-day | Probably a circuit breaker — check `circuit_breaker_active` metric and the `signals.rejected_gate` column. |
| Backtest is great, paper is worse | Slippage / fill model is optimistic. Re-run backtest with `slippage_bps=3.0` and see if you still hit viability gates. |

---

## Files committed safely (no secrets)

- `.gitignore`, `.env.example`, `config.yaml`, every `.py` and test
- `requirements.txt`

Files NEVER committed:
- `.env` (your credentials)
- `*.sqlite` (your trade log)
- `logs/`
- `backtest/_cache/`, `backtest/reports/`

---

## Credits

Strategy framework adapted from the spec at the top of the project. SDK: `alpaca-py` (the modern, supported one — not `alpaca-trade-api`).

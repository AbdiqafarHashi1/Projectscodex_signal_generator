# Codex Strategy Monitor

This project simulates multi-pair crypto signals with paper trading, diagnostics, and Telegram notifications.

## Configuration

Key environment variables can be set in a `.env` file. An example is provided in `.env.example`.

When the strategy starts it automatically loads a local `.env` file (if present) and exports any keys that are not already set in the process environment. Populate it with:

* `TELEGRAM_TOKEN` – bot token used for notifications.
* `TELEGRAM_CHAT_ID` – chat to receive alerts.

### Runtime flags

The strategy exposes a handful of runtime flags (via `argparse`). Override them with, for example, `python codex_strategy.py --diag_rate_limit_candles 6`:

* `--diag_rate_limit_candles` – number of closed candles to wait between Telegram diagnostics (default `4`).
* `--enable_session_guard` – toggle the quiet-session risk guard (`True`/`False`).
* `--cooldown_after_sl_mult` – multiplier applied to the interval when enforcing the post-stop-loss cool-down (default `3`).
* `--atr_regime_shift` – enable ATR-regime-based threshold and risk adjustments (`True`/`False`).

### Adaptive strictness relief

To reduce the chance of the strategy sitting idle for too long you can enable the relaxed mode tunables (defaults shown):

* `RELAX_STRICTNESS_ENABLED=true` – master switch; set to `false` to disable the feature entirely.
* `RELAX_AFTER_CANDLES=12` – number of consecutive closed candles without a trade before relax mode engages.
* `RELAX_THRESHOLD_OFFSET=1.0` – how many score points to subtract from the long/short thresholds while relaxed.
* `RELAX_DISABLE_PULLBACK=true` – allow trades even if the pullback filter fails once relax mode is active.
* `RELAX_ALLOW_SLOPE_MISMATCH=false` – if `true`, higher-time-frame slope mismatches are bypassed during relaxation.
* `RELAX_RISK_SCALE=0.75` – multiplier applied to risk allocation when relaxed (keep lower than `1` to stay conservative).
* `RELAX_MAX_RELAXED_CANDLES=6` – maximum number of candles to stay relaxed before automatically resetting (use `0` for unlimited).

### CSV rotation

`open_trades`, `closed_trades`, `diagnostics`, and `account_log` rotate daily. Files are suffixed with the current UTC date (e.g. `closed_trades_2024-03-01.csv`).

### Symbol precision

The simulator rounds quantities per symbol. Defaults:

| Symbol   | Decimals |
|----------|----------|
| ETHUSDT  | 3        |
| BTCUSDT  | 3        |
| SOLUSDT  | 2        |

## Development

Install dependencies and run the smoke tests:

```bash
pip install -r requirements.txt
python -m compileall .
python tests/smoke_strategy.py
```

Continuous integration runs the same commands via GitHub Actions.

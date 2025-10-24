import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make repo root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

# If your file is codex_v3.py, use:  import codex_v3 as strat
import codex_strategy as strat


def build_df(rows: int, freq: str) -> pd.DataFrame:
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    # Accept both "15min" and "15Min" etc.
    index = pd.date_range(start=start, periods=rows, freq=freq.replace("H", "h"))
    base = np.linspace(100, 110, rows)
    data = {
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base + np.sin(np.linspace(0, 3.14, rows)) * 0.5,
        "volume": np.linspace(1000, 2000, rows),
    }
    return pd.DataFrame(data, index=index)


def ensure_interval(interval: str):
    original = (strat.INTERVAL, strat.INTERVAL_SECONDS, strat.CYCLE_WAIT_SECONDS)
    strat.INTERVAL = interval
    interval_seconds = strat.interval_seconds(interval)
    strat.INTERVAL_SECONDS = interval_seconds
    strat.CYCLE_WAIT_SECONDS = interval_seconds
    return original


def restore_interval(original):
    strat.INTERVAL, strat.INTERVAL_SECONDS, strat.CYCLE_WAIT_SECONDS = original


def reset_state():
    strat.open_trades.clear()
    strat.closed_trades.clear()
    strat.last_sent_signal_ts.update({pair: None for pair in strat.PAIRS})
    strat.last_sl_time.update({pair: None for pair in strat.PAIRS})
    strat.last_atr_regime.update({pair: None for pair in strat.PAIRS})
    strat.last_volume_ok.update({pair: None for pair in strat.PAIRS})
    strat.diagnostic_cooldown.update({pair: strat.DIAG_RATE_LIMIT_CANDLES for pair in strat.PAIRS})
    # Relax-mode trackers
    strat.quiet_candle_streak.update({pair: 0 for pair in strat.PAIRS})
    strat.relax_active.update({pair: False for pair in strat.PAIRS})
    strat.relax_candles_remaining.update({pair: 0 for pair in strat.PAIRS})


def main():
    # Silence outbound Telegram during tests
    os.environ["TELEGRAM_TOKEN"] = ""
    os.environ["TELEGRAM_CHAT_ID"] = ""
    captured = []
    strat.send_telegram = lambda text: captured.append(text)

    reset_state()

    # 15m interval smoke
    original_interval = ensure_interval("15m")
    df_15m = build_df(200, "15min")
    assert df_15m.index.tzinfo is not None and df_15m.index.tzinfo.utcoffset(df_15m.index[-1]) == timedelta(0)
    sigs_15m, ctx_15m = strat.generate_signals_from_df("ETHUSDT", df_15m)
    strat.diagnose_candle("ETHUSDT", df_15m, ctx_15m)
    restore_interval(original_interval)

    # 1H interval smoke
    original_interval = ensure_interval("1H")
    df_1h = build_df(120, "1h")
    sigs_1h, ctx_1h = strat.generate_signals_from_df("ETHUSDT", df_1h)
    strat.diagnose_candle("ETHUSDT", df_1h, ctx_1h)
    restore_interval(original_interval)

    # Rate limit check (should send at most one DIAG at the threshold)
    reset_state()
    captured.clear()
    original_interval = ensure_interval("15m")
    df = build_df(200, "15min")
    ctx = strat.build_signal_context(df)
    prev_long = strat.LONG_SCORE_THRESHOLD
    prev_short = strat.SHORT_SCORE_THRESHOLD
    strat.LONG_SCORE_THRESHOLD = ctx['long_score']
    strat.SHORT_SCORE_THRESHOLD = ctx['short_score']
    strat.diagnostic_cooldown['ETHUSDT'] = strat.DIAG_RATE_LIMIT_CANDLES
    strat.diagnose_candle("ETHUSDT", df, ctx)
    strat.diagnose_candle("ETHUSDT", df, ctx)
    restore_interval(original_interval)
    strat.LONG_SCORE_THRESHOLD = prev_long
    strat.SHORT_SCORE_THRESHOLD = prev_short
    assert len(captured) <= 1, "Diagnostics rate limit failed"

    # Cooldown after SL blocks signals
    reset_state()
    original_interval = ensure_interval("15m")
    df = build_df(200, "15min")
    ctx = strat.build_signal_context(df)
    strat.last_sl_time['ETHUSDT'] = strat.now_utc()  # emulate SL just hit
    sigs, ctx = strat.generate_signals_from_df("ETHUSDT", df)
    restore_interval(original_interval)
    assert sigs.empty and 'cooldown_active' in ctx.get('skip_reasons', []), "Cooldown did not block signal"

    # Relax strictness kicks in after quiet streak
    reset_state()
    original_interval = ensure_interval("15m")
    df_relax = build_df(200, "15min")
    base_ctx = strat.build_signal_context(df_relax)

    prev_long = strat.LONG_SCORE_THRESHOLD
    prev_short = strat.SHORT_SCORE_THRESHOLD
    prev_relax_config = (
        strat.RELAX_STRICTNESS_ENABLED,
        strat.RELAX_AFTER_CANDLES,
        strat.RELAX_THRESHOLD_OFFSET,
        strat.RELAX_DISABLE_PULLBACK,
        strat.RELAX_ALLOW_SLOPE_MISMATCH,
        strat.RELAX_RISK_SCALE,
        strat.RELAX_MAX_RELAXED_CANDLES,
    )
    prev_volume_min = strat.VOLUME_MIN_MULT

    # Force relax activation quickly and make it permissive
    strat.RELAX_STRICTNESS_ENABLED = True
    strat.RELAX_AFTER_CANDLES = 1
    strat.RELAX_THRESHOLD_OFFSET = 3
    strat.RELAX_DISABLE_PULLBACK = True
    strat.RELAX_ALLOW_SLOPE_MISMATCH = True
    strat.RELAX_RISK_SCALE = 0.5
    strat.RELAX_MAX_RELAXED_CANDLES = 2
    strat.VOLUME_MIN_MULT = 1.0  # avoid volume gating in synthetic data
    strat.LONG_SCORE_THRESHOLD = base_ctx['long_score'] + 2
    strat.SHORT_SCORE_THRESHOLD = base_ctx['short_score'] + 2

    # One quiet candle (no trade) should enable relax mode
    strat.update_relax_tracker('ETHUSDT', False)
    sigs_relax, ctx_relax = strat.generate_signals_from_df("ETHUSDT", df_relax)

    restore_interval(original_interval)
    strat.LONG_SCORE_THRESHOLD = prev_long
    strat.SHORT_SCORE_THRESHOLD = prev_short
    (
        strat.RELAX_STRICTNESS_ENABLED,
        strat.RELAX_AFTER_CANDLES,
        strat.RELAX_THRESHOLD_OFFSET,
        strat.RELAX_DISABLE_PULLBACK,
        strat.RELAX_ALLOW_SLOPE_MISMATCH,
        strat.RELAX_RISK_SCALE,
        strat.RELAX_MAX_RELAXED_CANDLES,
    ) = prev_relax_config
    strat.VOLUME_MIN_MULT = prev_volume_min

    assert ctx_relax.get('relax_active'), "Relax mode did not activate"
    assert not sigs_relax.empty, "Relax mode failed to emit a signal"

    print("Smoke tests passed")


if __name__ == "__main__":
    main()

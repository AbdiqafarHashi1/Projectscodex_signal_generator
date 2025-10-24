import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import codex_strategy as strat


def build_df(rows: int, freq: str) -> pd.DataFrame:
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
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


def main():
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

    # Rate limit check
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

    # Cooldown after SL
    reset_state()
    original_interval = ensure_interval("15m")
    df = build_df(200, "15min")
    ctx = strat.build_signal_context(df)
    strat.last_sl_time['ETHUSDT'] = strat.now_utc()
    sigs, ctx = strat.generate_signals_from_df("ETHUSDT", df)
    restore_interval(original_interval)
    assert sigs.empty and 'cooldown_active' in ctx.get('skip_reasons', []), "Cooldown did not block signal"

    print("Smoke tests passed")


if __name__ == "__main__":
    main()

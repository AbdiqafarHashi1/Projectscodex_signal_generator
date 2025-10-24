import copy
import os
import sys
import tempfile
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
    original = strat.INTERVAL
    strat._set_interval(interval, announce=False)
    return original


def restore_interval(original):
    strat._set_interval(original, announce=False)


def reset_state():
    strat.open_trades.clear()
    strat.closed_trades.clear()
    strat.last_sent_signal_ts.update({pair: None for pair in strat.PAIRS})
    strat.last_sl_time.update({pair: None for pair in strat.PAIRS})
    strat.last_atr_regime.update({pair: None for pair in strat.PAIRS})
    strat.last_volume_ok.update({pair: None for pair in strat.PAIRS})
    strat.diagnostic_cooldown.update({pair: strat.DIAG_RATE_LIMIT_CANDLES for pair in strat.PAIRS})
    strat.quiet_candle_streak.update({pair: 0 for pair in strat.PAIRS})
    strat.relax_active.update({pair: False for pair in strat.PAIRS})
    strat.relax_candles_remaining.update({pair: 0 for pair in strat.PAIRS})
    strat._last_switch_candles = 0


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
    strat.RELAX_STRICTNESS_ENABLED = True
    strat.RELAX_AFTER_CANDLES = 1
    strat.RELAX_THRESHOLD_OFFSET = 3
    strat.RELAX_DISABLE_PULLBACK = True
    strat.RELAX_ALLOW_SLOPE_MISMATCH = True
    strat.RELAX_RISK_SCALE = 0.5
    strat.RELAX_MAX_RELAXED_CANDLES = 2
    strat.VOLUME_MIN_MULT = 1.0
    strat.LONG_SCORE_THRESHOLD = base_ctx['long_score'] + 2
    strat.SHORT_SCORE_THRESHOLD = base_ctx['short_score'] + 2
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

    # Interval tagging ensures CSVs capture interval column
    reset_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        original_files = (
            strat.OPEN_TRADES_CSV,
            strat.CLOSED_TRADES_CSV,
            strat.DIAGNOSTICS_CSV,
            strat.ACCOUNT_LOG_CSV,
        )
        original_bases = {name: cfg['base'] for name, cfg in strat.CSV_DEFINITIONS.items()}
        original_paths = copy.deepcopy(strat.CSV_PATHS)
        original_date = strat.CURRENT_CSV_DATE
        original_balance = strat.account_balance
        saved_interval = ensure_interval("1m")
        try:
            strat.OPEN_TRADES_CSV = os.path.join(tmpdir, "open_trades.csv")
            strat.CLOSED_TRADES_CSV = os.path.join(tmpdir, "closed_trades.csv")
            strat.DIAGNOSTICS_CSV = os.path.join(tmpdir, "diagnostics.csv")
            strat.ACCOUNT_LOG_CSV = os.path.join(tmpdir, "account_log.csv")
            for name, cfg in strat.CSV_DEFINITIONS.items():
                if name == 'open_trades':
                    cfg['base'] = strat.OPEN_TRADES_CSV
                elif name == 'closed_trades':
                    cfg['base'] = strat.CLOSED_TRADES_CSV
                elif name == 'diagnostics':
                    cfg['base'] = strat.DIAGNOSTICS_CSV
                elif name == 'account':
                    cfg['base'] = strat.ACCOUNT_LOG_CSV
            strat.CSV_PATHS = {}
            strat.CURRENT_CSV_DATE = None
            strat.ensure_daily_csvs()
            strat.account_balance = strat.STARTING_BALANCE
            trade = strat.open_simulated_trade("ETHUSDT", "LONG", 100.0, 95.0, 102.0, 105.0, {})
            strat.close_simulated_trade(trade['id'], 'TP2', 105.0)
            open_df = pd.read_csv(strat.CSV_PATHS['open_trades'])
            closed_df = pd.read_csv(strat.CSV_PATHS['closed_trades'])
            assert 'interval' in open_df.columns and open_df['interval'].iloc[-1] == "1m"
            assert 'interval' in closed_df.columns and closed_df['interval'].iloc[-1] == "1m"
        finally:
            strat.account_balance = original_balance
            restore_interval(saved_interval)
            strat.OPEN_TRADES_CSV, strat.CLOSED_TRADES_CSV, strat.DIAGNOSTICS_CSV, strat.ACCOUNT_LOG_CSV = original_files
            for name, cfg in strat.CSV_DEFINITIONS.items():
                cfg['base'] = original_bases[name]
            strat.CSV_PATHS = original_paths
            strat.CURRENT_CSV_DATE = original_date

    # Autoswitch escalation and guard behaviour
    reset_state()
    autoswitch_snapshot = {
        'TF_AUTOSWITCH_ENABLED': strat.TF_AUTOSWITCH_ENABLED,
        'TF_LIST': list(strat.TF_LIST),
        'TF_EVAL_LOOKBACK_TRADES': strat.TF_EVAL_LOOKBACK_TRADES,
        'TF_MIN_TRADES': strat.TF_MIN_TRADES,
        'TF_PNL_THRESHOLD': strat.TF_PNL_THRESHOLD,
        'TF_SWITCH_COOLDOWN_CANDLES': strat.TF_SWITCH_COOLDOWN_CANDLES,
        'TF_ALLOW_STEP_DOWN': strat.TF_ALLOW_STEP_DOWN,
        '_tf_sequence': list(strat._tf_sequence),
        '_current_tf_index': strat._current_tf_index,
        '_last_switch_candles': strat._last_switch_candles,
        'INTERVAL': strat.INTERVAL,
    }
    try:
        strat.TF_AUTOSWITCH_ENABLED = True
        strat.TF_LIST = ["1m", "5m", "15m"]
        strat._tf_sequence = list(strat.TF_LIST)
        strat.TF_EVAL_LOOKBACK_TRADES = 10
        strat.TF_MIN_TRADES = 5
        strat.TF_PNL_THRESHOLD = 0.0
        strat.TF_SWITCH_COOLDOWN_CANDLES = 1
        strat.TF_ALLOW_STEP_DOWN = False
        strat._set_interval("1m", announce=False)
        strat._current_tf_index = strat._tf_sequence.index("1m")
        strat._last_switch_candles = strat.TF_SWITCH_COOLDOWN_CANDLES
        strat.closed_trades.clear()
        for _ in range(10):
            strat.closed_trades.append({'interval': '1m', 'net_pnl': -5.0})
        strat.maybe_switch_timeframe()
        assert strat.INTERVAL == "5m", "Autoswitch did not escalate to 5m"

        strat.TF_ALLOW_STEP_DOWN = True
        strat._set_interval("5m", announce=False)
        strat._current_tf_index = strat._tf_sequence.index("5m")
        strat._last_switch_candles = strat.TF_SWITCH_COOLDOWN_CANDLES
        strat.closed_trades.clear()
        for _ in range(10):
            strat.closed_trades.append({'interval': '5m', 'net_pnl': 3.0})
        strat.maybe_switch_timeframe()
        assert strat.INTERVAL == "1m", "Autoswitch did not step back to 1m"

        strat.TF_AUTOSWITCH_ENABLED = True
        strat._set_interval("1m", announce=False)
        strat._current_tf_index = strat._tf_sequence.index("1m")
        strat._last_switch_candles = strat.TF_SWITCH_COOLDOWN_CANDLES
        strat.closed_trades.clear()
        for _ in range(4):
            strat.closed_trades.append({'interval': '1m', 'net_pnl': -5.0})
        strat.maybe_switch_timeframe()
        assert strat.INTERVAL == "1m", "Switch occurred despite insufficient trades"

        strat.TF_AUTOSWITCH_ENABLED = False
        strat._set_interval("1m", announce=False)
        strat._current_tf_index = strat._tf_sequence.index("1m")
        strat._last_switch_candles = strat.TF_SWITCH_COOLDOWN_CANDLES
        strat.closed_trades.clear()
        for _ in range(10):
            strat.closed_trades.append({'interval': '1m', 'net_pnl': -5.0})
        strat.maybe_switch_timeframe()
        assert strat.INTERVAL == "1m", "Switch occurred while autoswitch disabled"
    finally:
        strat.closed_trades.clear()
        strat.TF_AUTOSWITCH_ENABLED = autoswitch_snapshot['TF_AUTOSWITCH_ENABLED']
        strat.TF_LIST = list(autoswitch_snapshot['TF_LIST'])
        strat.TF_EVAL_LOOKBACK_TRADES = autoswitch_snapshot['TF_EVAL_LOOKBACK_TRADES']
        strat.TF_MIN_TRADES = autoswitch_snapshot['TF_MIN_TRADES']
        strat.TF_PNL_THRESHOLD = autoswitch_snapshot['TF_PNL_THRESHOLD']
        strat.TF_SWITCH_COOLDOWN_CANDLES = autoswitch_snapshot['TF_SWITCH_COOLDOWN_CANDLES']
        strat.TF_ALLOW_STEP_DOWN = autoswitch_snapshot['TF_ALLOW_STEP_DOWN']
        strat._tf_sequence = list(autoswitch_snapshot['_tf_sequence'])
        strat._set_interval(autoswitch_snapshot['INTERVAL'], announce=False)
        strat._current_tf_index = autoswitch_snapshot['_current_tf_index']
        strat._last_switch_candles = autoswitch_snapshot['_last_switch_candles']

    print("Smoke tests passed")


if __name__ == "__main__":
    main()

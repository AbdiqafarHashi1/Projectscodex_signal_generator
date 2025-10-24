# codex_v3.py
# Codex v3.0 — Multi-pair live monitor + diagnostics-to-Telegram + virtual capital + TP/SL monitoring + CSV logbook
# Requirements:
#   pip install python-binance pandas numpy requests pytz

import argparse
import time
import math
import os
from collections import deque
from datetime import datetime, timezone
import pytz
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
try:
    from binance.client import Client
except Exception:
    Client = None

# -----------------------
# CONFIG - tweak these
# -----------------------
# Binance keys (optional for public data; required only if you later enable test/trading)
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

# Telegram (required to receive alerts) — supply via environment
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Pairs to monitor (primary pair first)
PAIRS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

# Timeframe for signals
INTERVAL = os.environ.get("INTERVAL", "15m")

# Timezone (for human timestamps)
TZ = "Africa/Nairobi"
EAT = pytz.timezone(TZ)

# Virtual capital & risk
STARTING_BALANCE = float(os.environ.get("STARTING_BALANCE", 100000.0))  # USDT starting balance (per account)
account_balance = STARTING_BALANCE
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", 0.02))         # fraction of balance risked per trade

# Strategy tuning
REQUIRE_1H_AGREEMENT = False
REQUIRE_4H_AGREEMENT = False
RSI_LONG_MIN = 45
RSI_SHORT_MAX = 55
VOLUME_FACTOR = 1.0

# Advanced confluence tuning
EMA_SLOPE_LOOKBACK = 5
EMA_SLOPE_MIN = 0.0
ATR_PCT_MIN = 0.003          # skip if market too quiet
ATR_PCT_MAX = 0.04           # skip if market too volatile
ATR_PCT_RISK_HIGH = 0.025    # risk cut when ATR % too high
ATR_PCT_RISK_MED = 0.015     # moderate volatility adjustment
VOLUME_MIN_MULT = 1.1        # require 10% volume expansion vs 20-period average
LONG_SCORE_THRESHOLD = 6
SHORT_SCORE_THRESHOLD = 6

# --- Regime-aware tuning & guards ---
ATR_REGIME_SHIFT_ENABLED = True
ATR_REGIME_LOW_SHIFT = 1                 # add 1 point to threshold in low-vol regime
ATR_REGIME_HIGH_RISK_REDUCTION = 0.3     # cut risk by 30% in high-vol regime
ENABLE_PULLBACK_FILTER = True            # require close within ~0.3% of EMA20
ENABLE_HTF_SLOPE_ALIGN = True            # 1H/4H EMA slope must agree with direction
HTF_SLOPE_PERIOD = 20                    # candles for slope calc
ENABLE_SL_COOLDOWN = True
COOLDOWN_AFTER_SL_MULT = float(os.environ.get("COOLDOWN_AFTER_SL_MULT", 3))
ENABLE_SESSION_GUARD = False
SESSION_GUARD_HOURS = (0, 6)             # UTC quiet hours window
SESSION_GUARD_ACTION = os.environ.get("SESSION_GUARD_ACTION", "reduce")
SESSION_GUARD_RISK_SCALE = float(os.environ.get("SESSION_GUARD_RISK_SCALE", 0.5))
DIAG_RATE_LIMIT_CANDLES = int(os.environ.get("DIAG_RATE_LIMIT_CANDLES", 4))

# Stop / TP config
STOP_METHOD = 'atr'            # 'atr' or 'fixed'
FIXED_SL_PCT = 0.005           # 0.5% if using fixed
TP_STYLE = 'two'               # 'one' or 'two'
TP1_SPLIT = 0.5                # proportion closed at TP1 when TP_STYLE=='two'

# Notifications
SEND_DIAGNOSTICS_TO_TELEGRAM = True
SEND_PERIODIC_SUMMARY = True
SUMMARY_INTERVAL_S = 60 * 60   # every hour

# CSV files
OPEN_TRADES_CSV = "open_trades.csv"
CLOSED_TRADES_CSV = "closed_trades.csv"
DIAGNOSTICS_CSV = "diagnostics.csv"
ACCOUNT_LOG_CSV = "account_log.csv"
EVENTS_LOG = "events.log"

# Polling
PRICE_POLL_SECONDS = 60

# Execution flags
ENABLE_TESTNET = False         # leave False — using live data only
MAX_CONCURRENT_TRADES_PER_SYMBOL = 1

# Fees (for simulation)
FEE_RATE = 0.001  # round-trip approx (adjust)

SYMBOL_PRECISION = {
    "ETHUSDT": 3,
    "BTCUSDT": 3,
    "SOLUSDT": 2,
}

# -----------------------
# Setup clients
# -----------------------
client = None
if Client is not None and BINANCE_API_KEY:
    try:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=ENABLE_TESTNET)
    except Exception as e:
        print("Binance client init error:", e)
        client = None
else:
    if Client is None:
        print("python-binance not installed — will use public REST for klines.")

# -----------------------
# Utilities
# -----------------------
def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

def now_local() -> datetime:
    return now_utc().astimezone(EAT)

def ts_to_local_str(ts) -> str:
    t = pd.to_datetime(ts)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    return t.tz_convert(EAT).strftime("%Y-%m-%d %H:%M:%S")

def interval_seconds(s: str) -> int:
    unit = s[-1].lower()
    n = int(s[:-1])
    return n * (60 if unit == 'm' else 3600 if unit == 'h' else 86400 if unit == 'd' else 60)

INTERVAL_SECONDS = interval_seconds(INTERVAL)
CYCLE_WAIT_SECONDS = INTERVAL_SECONDS

def _backoff_delays():
    for delay in (0.5, 1.0, 2.0):
        yield delay

def _with_backoff(func, *args, **kwargs):
    last_err = None
    for delay in _backoff_delays():
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            time.sleep(delay)
    if last_err:
        raise last_err
    raise RuntimeError("backoff failed without attempts")

def send_telegram(text: str) -> dict:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return {"ok": False, "skipped": "no_token"}

    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text}
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        return resp

    try:
        response = _with_backoff(_send)
        try:
            return response.json()
        except ValueError:
            return {"ok": False, "error": "invalid_json"}
    except Exception as e:
        log_event(f"Telegram error: {e}")
        return {"ok": False, "error": str(e)}

def ensure_csv_headers(path: str, columns: List[str]):
    """Guarantee that a CSV file exists with the provided header order."""
    try:
        if os.path.exists(path):
            return
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    except Exception as e:
        print("CSV init error:", e)

def append_csv(path: str, row: Dict[str, Any], columns: Optional[List[str]] = None):
    if not row:
        return
    df_row = pd.DataFrame([row])
    if columns is not None:
        missing = [c for c in columns if c not in df_row.columns]
        for col in missing:
            df_row[col] = np.nan
        df_row = df_row[columns]
    try:
        with open(path, 'a', encoding='utf-8') as f:
            df_row.to_csv(f, header=f.tell() == 0, index=False)
    except Exception as e:
        print("CSV write error:", e)

# ---- Daily-rotated CSVs ----
CSV_DEFINITIONS = {
    'open_trades': {
        'base': OPEN_TRADES_CSV,
        'columns': ['id','pair','signal','entry','sl','tp1','tp2','qty','initial_qty','remaining_qty','notional','est_fees','created_at','tp1_hit','status','sl_to_entry','realized_pnl','risk_frac','pattern','pattern_bias','rsi','score','atr_pct','volume_ratio','htf_confluence','atr_regime']
    },
    'closed_trades': {
        'base': CLOSED_TRADES_CSV,
        'columns': ['id','pair','signal','entry','sl','tp1','tp2','qty','notional','est_fees','created_at','closed_at','result','close_price','gross_pnl','fees','net_pnl','balance_after','remaining_qty','partial','realized_total','risk_frac','pattern','pattern_bias','rsi','score','atr_pct','volume_ratio','htf_confluence','atr_regime']
    },
    'diagnostics': {
        'base': DIAGNOSTICS_CSV,
        'columns': ['timestamp','pair','price','ema20','ema50','ema_slope','rsi','atr','atr_pct','atr_low','atr_high','atr_regime','vol','vol_ma20','volume_ratio','volume_ok','bias_1h','bias_4h','htf_slope_1h','htf_slope_4h','bull_score','bear_score','pattern','pattern_bias','bull_htf','bear_htf','risk_frac','pos_qty','pullback_ok','passed','reasons']
    },
    'account': {
        'base': ACCOUNT_LOG_CSV,
        'columns': ['ts','balance','net_pnl']
    }
}
CURRENT_CSV_DATE = None
CSV_PATHS: Dict[str, str] = {}

def _daily_csv_name(base: str, date_str: str) -> str:
    root, ext = os.path.splitext(base)
    if not ext:
        ext = '.csv'
    return f"{root}_{date_str}{ext}"

def ensure_daily_csvs():
    global CURRENT_CSV_DATE, CSV_PATHS
    today = now_utc().strftime('%Y-%m-%d')
    if CURRENT_CSV_DATE == today:
        return
    CURRENT_CSV_DATE = today
    CSV_PATHS = {}
    for name, cfg in CSV_DEFINITIONS.items():
        path = _daily_csv_name(cfg['base'], today)
        CSV_PATHS[name] = path
        ensure_csv_headers(path, cfg['columns'])

def append_named_csv(name: str, row: Dict[str, Any]):
    ensure_daily_csvs()
    path = CSV_PATHS.get(name)
    if not path:
        return
    append_csv(path, row, CSV_DEFINITIONS[name]['columns'])

def compute_dynamic_risk_frac(atr_pct: float) -> float:
    """Return a risk fraction adjustment based on volatility."""
    if atr_pct is None:
        return 0.0
    try:
        atr_pct_val = float(atr_pct)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(atr_pct_val) or atr_pct_val <= 0:
        return 0.0
    if atr_pct_val < ATR_PCT_MIN:
        return 0.0
    if atr_pct_val > ATR_PCT_MAX:
        return 0.0
    risk = RISK_PER_TRADE
    if atr_pct_val > ATR_PCT_RISK_HIGH:
        risk *= 0.5
    elif atr_pct_val > ATR_PCT_RISK_MED:
        risk *= 0.75
    return risk

def log_event(msg: str):
    ts = now_local().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(EVENTS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# -----------------------
# Indicators / strategy
# -----------------------
def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14):
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def resample_tf(df_small: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Expect uppercase rules like '1H', '4H'
    rule = rule.upper()
    df_small = df_small.sort_index()
    o = df_small['open'].resample(rule).first()
    h = df_small['high'].resample(rule).max()
    l = df_small['low'].resample(rule).min()
    c = df_small['close'].resample(rule).last()
    v = df_small['volume'].resample(rule).sum()
    res = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    return res.dropna()

def detect_candlestick_pattern(prev: pd.Series, current: pd.Series) -> Dict[str, Any]:
    pattern = None
    bias = 'neutral'
    prev_open, prev_close = prev['open'], prev['close']
    cur_open, cur_close = current['open'], current['close']
    prev_high, prev_low = prev['high'], prev['low']
    cur_high, cur_low = current['high'], current['low']

    def body(o, c): return abs(c - o)
    def upper_shadow(o, c, h): return h - max(o, c)
    def lower_shadow(o, c, l): return min(o, c) - l

    prev_body = body(prev_open, prev_close)
    curr_body = body(cur_open, cur_close)
    prev_upper = upper_shadow(prev_open, prev_close, prev_high)
    prev_lower = lower_shadow(prev_open, prev_close, prev_low)
    curr_upper = upper_shadow(cur_open, cur_close, cur_high)
    curr_lower = lower_shadow(cur_open, cur_close, cur_low)

    # Bullish
    if prev_close < prev_open and cur_close > cur_open:
        if cur_close >= prev_open and cur_open <= prev_close and curr_body > prev_body:
            pattern = 'bullish_engulfing'; bias = 'bullish'
    if bias == 'neutral' and cur_close > cur_open:
        if curr_lower >= 2 * curr_body and curr_upper <= curr_body * 0.5:
            pattern = 'bullish_hammer'; bias = 'bullish'

    # Bearish
    if prev_close > prev_open and cur_close < cur_open:
        if cur_open >= prev_close and cur_close <= prev_open and curr_body > prev_body:
            pattern = 'bearish_engulfing'; bias = 'bearish'
    if bias == 'neutral' and cur_close < cur_open:
        if curr_upper >= 2 * curr_body and curr_lower <= curr_body * 0.5:
            pattern = 'bearish_shooting_star'; bias = 'bearish'

    return {'pattern': pattern, 'bias': bias, 'details': {'curr_body': curr_body, 'curr_upper': curr_upper, 'curr_lower': curr_lower}}

def higher_tf_trend(df: pd.DataFrame, rule: str) -> Optional[bool]:
    try:
        res = resample_tf(df, rule)
        if res.empty:
            return None
        res['ema20'] = ema(res['close'], 20)
        res['ema50'] = ema(res['close'], 50)
        latest = res.iloc[-1]
        if pd.isna(latest['ema20']) or pd.isna(latest['ema50']):
            return None
        return bool(latest['ema20'] > latest['ema50'])
    except Exception:
        return None

def build_signal_context(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema20_series = ema(df['close'], 20)
    ema50_series = ema(df['close'], 50)
    ema20 = float(ema20_series.iloc[-1])
    ema50 = float(ema50_series.iloc[-1])
    ema_diff_series = ema20_series - ema50_series
    ema_diff_last = float(ema_diff_series.iloc[-1])
    ema_diff_prev = float(ema_diff_series.iloc[-2])
    ema_slope = float(ema20_series.diff().rolling(EMA_SLOPE_LOOKBACK, min_periods=1).mean().iloc[-1])

    rsi_series = rsi(df['close'], 14)
    rsi_last = float(rsi_series.iloc[-1])

    atr_series = atr(df, 14)
    atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
    atr_pct_series = atr_series / df['close']
    atr_pct = float(atr_pct_series.iloc[-1]) if len(atr_series) else 0.0
    clean_atr = atr_pct_series.dropna()
    atr_low = float(np.nanpercentile(clean_atr, 33)) if len(clean_atr) else float('nan')
    atr_high = float(np.nanpercentile(clean_atr, 66)) if len(clean_atr) else float('nan')
    if len(clean_atr):
        if atr_pct <= atr_low: atr_regime = 'low'
        elif atr_pct >= atr_high: atr_regime = 'high'
        else: atr_regime = 'mid'
    else:
        atr_regime = 'mid'

    vol_ma20 = float(df['volume'].rolling(20, min_periods=1).mean().iloc[-1])
    vol_last = float(last['volume'])
    volume_ratio = (vol_last / vol_ma20) if vol_ma20 else 0.0
    volume_ok = volume_ratio >= VOLUME_MIN_MULT

    pattern_info = detect_candlestick_pattern(df.iloc[-2], df.iloc[-1])

    crossed_up = ema_diff_prev < 0 and ema_diff_last > 0
    crossed_down = ema_diff_prev > 0 and ema_diff_last < 0
    recent_momentum = float(ema_diff_series.rolling(5, min_periods=1).mean().iloc[-1])
    momentum_bull = recent_momentum > 0
    momentum_bear = recent_momentum < 0

    bias_1h = higher_tf_trend(df, '1H')
    bias_4h = higher_tf_trend(df, '4H')
    bull_htf = sum(1 for b in (bias_1h, bias_4h) if b is True)
    bear_htf = sum(1 for b in (bias_1h, bias_4h) if b is False)

    structure_bull = last['close'] > ema20 > ema50
    structure_bear = last['close'] < ema20 < ema50
    atr_in_range = ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX

    long_score = 0
    short_score = 0
    if structure_bull: long_score += 1
    if structure_bear: short_score += 1
    if crossed_up:     long_score += 1
    if crossed_down:   short_score += 1
    if momentum_bull:  long_score += 1
    if momentum_bear:  short_score += 1
    if pattern_info['bias'] == 'bullish': long_score += 1
    if pattern_info['bias'] == 'bearish': short_score += 1
    if rsi_last >= RSI_LONG_MIN: long_score += 1
    if rsi_last <= RSI_SHORT_MAX: short_score += 1
    if volume_ok:
        long_score += 1; short_score += 1
    if atr_in_range:
        long_score += 1; short_score += 1
    if ema_slope > EMA_SLOPE_MIN:  long_score += 1
    if ema_slope < -EMA_SLOPE_MIN: short_score += 1
    long_score += bull_htf
    short_score += bear_htf

    risk_frac = compute_dynamic_risk_frac(atr_pct)

    # HTF slopes + pullback filter
    htf_slopes = {}
    for tf in ['1H', '4H']:
        res = resample_tf(df, tf)
        if len(res) >= HTF_SLOPE_PERIOD + 2:
            slope_series = ema(res['close'], HTF_SLOPE_PERIOD)
            slope_val = float(slope_series.diff().rolling(EMA_SLOPE_LOOKBACK, min_periods=1).mean().iloc[-1])
        else:
            slope_val = float('nan')
        htf_slopes[tf] = slope_val

    pullback_ok = True
    if ENABLE_PULLBACK_FILTER and last['close']:
        pullback = abs((last['close'] - ema20) / last['close'])
        pullback_ok = pullback <= 0.003

    return {
        'ema20': ema20, 'ema50': ema50,
        'ema_diff_last': ema_diff_last, 'ema_diff_prev': ema_diff_prev,
        'ema_slope': ema_slope, 'rsi': rsi_last,
        'atr': atr_val, 'atr_pct': atr_pct,
        'atr_low': atr_low, 'atr_high': atr_high, 'atr_regime': atr_regime,
        'atr_in_range': atr_in_range,
        'volume': vol_last, 'volume_ma20': vol_ma20,
        'volume_ratio': volume_ratio, 'volume_ok': volume_ok,
        'pattern_info': pattern_info,
        'crossed_up': crossed_up, 'crossed_down': crossed_down,
        'momentum_bull': momentum_bull, 'momentum_bear': momentum_bear,
        'bias_1h': bias_1h, 'bias_4h': bias_4h,
        'bull_htf': bull_htf, 'bear_htf': bear_htf,
        'structure_bull': structure_bull, 'structure_bear': structure_bear,
        'long_score': long_score, 'short_score': short_score,
        'risk_frac': risk_frac,
        'htf_slopes': htf_slopes, 'pullback_ok': pullback_ok,
        'last': last, 'prev': prev
    }

# -----------------------
# Live data helpers
# -----------------------
def get_klines_df(symbol: str, interval: str = INTERVAL, limit: int = 500) -> pd.DataFrame:
    try:
        if client is not None:
            def _fetch():
                return client.get_klines(symbol=symbol, interval=interval, limit=limit)
            klines = _with_backoff(_fetch)
        else:
            def _fetch():
                url = "https://api.binance.com/api/v3/klines"
                r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
                r.raise_for_status()
                return r.json()
            klines = _with_backoff(_fetch)
        df = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume','close_time','quote_asset_volume','num_trades','taker_buy_base','taker_buy_quote','ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df = df[['timestamp','open','high','low','close','volume']].set_index('timestamp')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        return df
    except Exception as e:
        log_event(f"get_klines_df error for {symbol}: {e}")
        raise

def get_current_price(symbol: str) -> float:
    try:
        if client is not None:
            def _fetch():
                t = client.get_symbol_ticker(symbol=symbol)
                return float(t['price'])
            return _with_backoff(_fetch)
        else:
            def _fetch():
                r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
                r.raise_for_status()
                return float(r.json()['price'])
            return _with_backoff(_fetch)
    except Exception as e:
        log_event(f"get_current_price error {symbol}: {e}")
        raise

# -----------------------
# Position sizing & account simulation
# -----------------------
open_trades: Dict[str, Dict] = {}
closed_trades: deque = deque(maxlen=200)
last_sent_signal_ts: Dict[str, Optional[str]] = {pair: None for pair in PAIRS}
last_summary_time = time.time()
next_summary_time = time.time() + SUMMARY_INTERVAL_S
last_sl_time: Dict[str, Optional[datetime]] = {pair: None for pair in PAIRS}
last_atr_regime: Dict[str, Optional[str]] = {pair: None for pair in PAIRS}
last_volume_ok: Dict[str, Optional[bool]] = {pair: None for pair in PAIRS}
diagnostic_cooldown: Dict[str, int] = {pair: DIAG_RATE_LIMIT_CANDLES for pair in PAIRS}

def calculate_position_size(symbol: str, entry_price: float, stop_loss: float, balance: float, risk_frac: float) -> float:
    distance = abs(entry_price - stop_loss)
    if distance <= 0:
        return 0.0
    risk_amount = balance * risk_frac
    qty = risk_amount / distance
    precision = SYMBOL_PRECISION.get(symbol, 3)
    factor = 10 ** precision
    return float(math.floor(qty * factor) / factor)

def apply_precision(symbol: str, qty: float) -> float:
    precision = SYMBOL_PRECISION.get(symbol, 3)
    factor = 10 ** precision
    return float(math.floor(qty * factor) / factor)

def open_simulated_trade(pair: str, signal: str, entry: float, sl: float, tp1: float, tp2: float, meta: Optional[Dict[str, Any]] = None):
    global account_balance
    ts_key = f"{pair}_{entry}_{now_utc().isoformat()}"
    # concurrency protection
    count_open = sum(1 for t in open_trades.values() if t['pair'] == pair)
    if count_open >= MAX_CONCURRENT_TRADES_PER_SYMBOL:
        send_telegram(f"⚠️ Skipping open for {pair}: {count_open} open trades already.")
        log_event(f"Skip open {pair}: concurrency")
        return None

    risk_frac = RISK_PER_TRADE
    if meta and meta.get('risk_frac') is not None:
        risk_frac = meta['risk_frac']
    qty = calculate_position_size(pair, entry, sl, account_balance, risk_frac)
    notional = qty * entry
    est_fees = notional * FEE_RATE
    trade = {
        'id': ts_key, 'pair': pair, 'signal': signal,
        'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
        'qty': qty, 'initial_qty': qty, 'remaining_qty': qty,
        'notional': notional, 'est_fees': est_fees,
        'created_at': now_utc().isoformat(), 'tp1_hit': False,
        'status': 'OPEN', 'sl_to_entry': False, 'realized_pnl': 0.0,
        'risk_frac': risk_frac
    }
    if meta: trade.update(meta)
    open_trades[ts_key] = trade
    mark_diag_event(pair)
    append_named_csv('open_trades', trade)
    pattern_note = f" pattern {trade.get('pattern')}" if trade.get('pattern') else ""
    score = trade.get('score')
    risk_pct = trade.get('risk_frac', risk_frac) * 100
    atr_pct = trade.get('atr_pct')
    vol_ratio = trade.get('volume_ratio')
    stats_line = ""
    if (
        atr_pct is not None and isinstance(atr_pct, (int, float)) and not math.isnan(float(atr_pct))
        and vol_ratio is not None and isinstance(vol_ratio, (int, float)) and not math.isnan(float(vol_ratio))
    ):
        score_display = score if score is not None else 'n/a'
        if isinstance(score_display, float): score_display = int(round(score_display))
        htf_conf = trade.get('htf_confluence'); htf_display = htf_conf if htf_conf is not None else 'n/a'
        stats_line = f"\nScore: {score_display} | HTF conf: {htf_display} | Risk: {risk_pct:.2f}% | ATR%: {float(atr_pct):.3f} | VolRatio: {float(vol_ratio):.2f}"
    send_telegram(f"📥 OPEN_SIM {pair} {signal} qty {qty:.6f} @ {entry:.2f} SL {sl:.2f} TP1 {tp1:.2f} TP2 {tp2:.2f}{pattern_note}" + stats_line)
    log_event(f"Opened {pair} {signal} qty {qty:.6f} @ {entry:.2f}")
    return trade

def close_simulated_trade(trade_id: str, result: str, close_price: float):
    global account_balance
    trade = open_trades.pop(trade_id, None)
    if not trade:
        return None
    qty = trade.get('remaining_qty', trade['qty'])
    entry = trade['entry']; signal = trade['signal']
    notional = entry * qty
    if qty <= 0:
        log_event("Close skipped: qty 0")
        return None
    gross = (close_price - entry) * qty if signal == 'LONG' else (entry - close_price) * qty
    fees = notional * FEE_RATE
    net = gross - fees
    account_balance += net
    realized = trade.get('realized_pnl', 0.0) + net
    trade['status'] = 'CLOSED'
    closed = {
        **trade,
        'closed_at': now_utc().isoformat(),
        'result': result,
        'close_price': close_price,
        'gross_pnl': gross,
        'fees': fees,
        'net_pnl': net,
        'realized_total': realized,
        'balance_after': account_balance
    }
    closed_trades.append(closed)
    append_named_csv('closed_trades', closed)
    append_named_csv('account', {'ts': now_utc().isoformat(), 'balance': account_balance, 'net_pnl': net})
    mark_diag_event(trade['pair'])
    send_telegram(f"📤 CLOSE {trade['pair']} {signal} {result} @ {close_price:.2f} net {net:.2f} bal {account_balance:.2f}")
    log_event(f"Closed {trade['pair']} {signal} {result} @ {close_price:.2f} net {net:.2f}")
    return closed

def partial_close_trade(trade_id: str, fraction: float, close_price: float, label: str) -> Optional[Dict[str, Any]]:
    global account_balance
    trade = open_trades.get(trade_id)
    if not trade:
        return None
    fraction = max(0.0, min(1.0, fraction))
    if fraction <= 0:
        return None
    remaining_qty = trade.get('remaining_qty', trade['qty'])
    qty_to_close = remaining_qty * fraction
    qty_to_close = apply_precision(trade['pair'], qty_to_close)
    if qty_to_close <= 0:
        return None
    entry = trade['entry']; signal = trade['signal']
    gross = (close_price - entry) * qty_to_close if signal == 'LONG' else (entry - close_price) * qty_to_close
    fees = entry * qty_to_close * FEE_RATE
    net = gross - fees
    account_balance += net
    trade['remaining_qty'] = max(0.0, remaining_qty - qty_to_close)
    trade['realized_pnl'] = trade.get('realized_pnl', 0.0) + net
    append_named_csv('account', {'ts': now_utc().isoformat(), 'balance': account_balance, 'net_pnl': net})
    proportion = qty_to_close / trade['initial_qty'] if trade['initial_qty'] else 0
    closed = {
        'id': f"{trade_id}_partial",
        'pair': trade['pair'],
        'signal': signal,
        'entry': entry,
        'sl': trade['sl'],
        'tp1': trade['tp1'],
        'tp2': trade['tp2'],
        'qty': qty_to_close,
        'notional': trade['notional'] * proportion,
        'est_fees': trade['est_fees'] * proportion,
        'created_at': trade['created_at'],
        'closed_at': now_utc().isoformat(),
        'result': label,
        'close_price': close_price,
        'gross_pnl': gross,
        'fees': fees,
        'net_pnl': net,
        'balance_after': account_balance,
        'remaining_qty': trade['remaining_qty'],
        'partial': True,
        'realized_total': trade['realized_pnl']
    }
    closed_trades.append(closed)
    append_named_csv('closed_trades', closed)
    send_telegram(f"📉 PARTIAL {trade['pair']} {signal} {label} qty {qty_to_close:.6f} @ {close_price:.2f} net {net:.2f} bal {account_balance:.2f}")
    log_event(f"Partial close {trade['pair']} {signal} {label} @ {close_price:.2f} qty {qty_to_close:.6f} net {net:.2f}")
    if trade['remaining_qty'] <= 0:
        trade['status'] = 'CLOSED'
        open_trades.pop(trade_id, None)
        send_telegram(f"✅ {trade['pair']} position fully closed after {label} scaling out.")
        log_event(f"{trade['pair']} trade fully closed after partial exits.")
    return closed

def check_trades_and_notify(pair: str, current_price: float):
    messages = []
    for tid, trade in list(open_trades.items()):
        if trade['pair'] != pair:
            continue
        signal = trade['signal']; entry = trade['entry']; tp1 = trade['tp1']; tp2 = trade['tp2']; sl = trade['sl']
        if trade['qty'] <= 0:
            continue
        if signal == 'LONG':
            if current_price >= tp2:
                messages.append(f"🎯 TP2 HIT {pair} @ {current_price:.2f}")
                close_simulated_trade(tid, 'TP2', current_price)
            elif current_price >= tp1 and not trade['tp1_hit']:
                trade['tp1_hit'] = True
                messages.append(f"✅ TP1 HIT {pair} @ {current_price:.2f}")
                if TP_STYLE == 'two' and TP1_SPLIT > 0:
                    partial_close_trade(tid, TP1_SPLIT, current_price, 'TP1')
                if not trade.get('sl_to_entry'):
                    trade['sl'] = entry; trade['sl_to_entry'] = True
                    send_telegram(f"🔒 Moved SL to entry for {pair} after TP1. New SL {entry:.2f}")
                    log_event(f"{pair} SL moved to entry after TP1")
            elif current_price <= sl:
                messages.append(f"❌ SL HIT {pair} @ {current_price:.2f}")
                close_simulated_trade(tid, 'SL', current_price)
            else:
                if not trade['tp1_hit'] and (tp1 - current_price) / tp1 <= 0.005:
                    messages.append(f"📈 {pair} price approaching TP1 ({tp1:.2f}) price {current_price:.2f}")
                if (current_price - sl) / sl <= 0.004:
                    messages.append(f"⚠️ {pair} price approaching SL ({sl:.2f}) price {current_price:.2f}")
        else:  # SHORT
            if current_price <= tp2:
                messages.append(f"🎯 TP2 HIT {pair} @ {current_price:.2f}")
                close_simulated_trade(tid, 'TP2', current_price)
            elif current_price <= tp1 and not trade['tp1_hit']:
                trade['tp1_hit'] = True
                messages.append(f"✅ TP1 HIT {pair} @ {current_price:.2f}")
                if TP_STYLE == 'two' and TP1_SPLIT > 0:
                    partial_close_trade(tid, TP1_SPLIT, current_price, 'TP1')
                if not trade.get('sl_to_entry'):
                    trade['sl'] = entry; trade['sl_to_entry'] = True
                    send_telegram(f"🔒 Moved SL to entry for {pair} after TP1. New SL {entry:.2f}")
                    log_event(f"{pair} SL moved to entry after TP1")
            elif current_price >= sl:
                messages.append(f"❌ SL HIT {pair} @ {current_price:.2f}")
                close_simulated_trade(tid, 'SL', current_price)
            else:
                if not trade['tp1_hit'] and (current_price - tp1) / tp1 <= 0.005:
                    messages.append(f"📉 {pair} price approaching TP1 ({tp1:.2f}) price {current_price:.2f}")
                if (sl - current_price) / sl <= 0.004:
                    messages.append(f"⚠️ {pair} price approaching SL ({sl:.2f}) price {current_price:.2f}")
    for m in messages:
        send_telegram(m); log_event(m)
    return messages

# -----------------------
# Diagnostics logic (per closed candle)
# -----------------------
def mark_diag_event(pair: str):
    diagnostic_cooldown[pair] = DIAG_RATE_LIMIT_CANDLES

def should_send_diag(pair: str, ctx: Dict[str, Any], passed: bool, close_score: Optional[int], threshold: int) -> bool:
    atr_regime = ctx.get('atr_regime')
    volume_ok = ctx.get('volume_ok')
    atr_flip = atr_regime != last_atr_regime.get(pair)
    volume_flip = volume_ok != last_volume_ok.get(pair)
    near_threshold = close_score is not None and abs(close_score - threshold) <= 1

    last_atr_regime[pair] = atr_regime
    last_volume_ok[pair] = volume_ok

    diagnostic_cooldown[pair] = diagnostic_cooldown.get(pair, DIAG_RATE_LIMIT_CANDLES) + 1
    if diagnostic_cooldown[pair] < DIAG_RATE_LIMIT_CANDLES:
        return False
    if not (near_threshold or atr_flip or volume_flip):
        return False
    diagnostic_cooldown[pair] = 0
    return True

def session_guard_adjust(ts: pd.Timestamp, risk_frac: float) -> Tuple[float, Optional[str]]:
    if not ENABLE_SESSION_GUARD:
        return risk_frac, None
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    hour = ts.tz_convert(timezone.utc).hour
    if SESSION_GUARD_HOURS[0] <= hour < SESSION_GUARD_HOURS[1]:
        action = SESSION_GUARD_ACTION.lower()
        if action == 'skip':
            return 0.0, 'session_guard_skip'
        return risk_frac * SESSION_GUARD_RISK_SCALE, 'session_guard_reduce'
    return risk_frac, None

def slopes_agree(ctx: Dict[str, Any], direction: str) -> bool:
    if not ENABLE_HTF_SLOPE_ALIGN:
        return True
    desired_sign = 1 if direction == 'LONG' else -1
    slopes = ctx.get('htf_slopes', {})
    for slope in slopes.values():
        if slope is None or (isinstance(slope, float) and math.isnan(slope)):
            continue
        if slope == 0:
            continue
        if slope * desired_sign < 0:
            return False
    return True

def in_sl_cooldown(pair: str, ts: pd.Timestamp) -> bool:
    if not ENABLE_SL_COOLDOWN:
        return False
    last_hit = last_sl_time.get(pair)
    if not last_hit:
        return False
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return (ts.tz_convert(timezone.utc).to_pydatetime() - last_hit).total_seconds() < COOLDOWN_AFTER_SL_MULT * INTERVAL_SECONDS

def diagnose_candle(pair: str, df: pd.DataFrame, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        last_ts = df.index[-1]
        ctx = ctx or build_signal_context(df)
        last_row = ctx['last']

        reasons = []
        if not (ctx['structure_bull'] or ctx['structure_bear']):
            reasons.append("15m structure neutral")
        if ctx['structure_bull'] and ctx['long_score'] < LONG_SCORE_THRESHOLD:
            reasons.append(f"Bull score {ctx['long_score']} < {LONG_SCORE_THRESHOLD}")
        if ctx['structure_bear'] and ctx['short_score'] < SHORT_SCORE_THRESHOLD:
            reasons.append(f"Bear score {ctx['short_score']} < {SHORT_SCORE_THRESHOLD}")
        if ctx['structure_bull'] and ctx['bull_htf'] <= 0:
            reasons.append("Higher TF not bullish")
        if ctx['structure_bear'] and ctx['bear_htf'] <= 0:
            reasons.append("Higher TF not bearish")

        if REQUIRE_1H_AGREEMENT:
            if ctx['structure_bull'] and ctx['bias_1h'] is False: reasons.append("1H not bullish")
            if ctx['structure_bear'] and ctx['bias_1h'] is True:  reasons.append("1H bullish (blocks short)")
        if REQUIRE_4H_AGREEMENT:
            if ctx['structure_bull'] and ctx['bias_4h'] is False: reasons.append("4H not bullish")
            if ctx['structure_bear'] and ctx['bias_4h'] is True:  reasons.append("4H bullish (blocks short)")

        if not ctx['volume_ok']:
            reasons.append(f"Volume ratio {ctx['volume_ratio']:.2f} < {VOLUME_MIN_MULT}")
        if not ctx['atr_in_range']:
            reasons.append("Volatility outside preferred ATR% range")

        risk_frac = ctx['risk_frac']
        atr_component = ctx['atr'] if STOP_METHOD == 'atr' else last_row['close'] * FIXED_SL_PCT
        if ctx['structure_bull']:
            simulated_sl = last_row['close'] - atr_component
        elif ctx['structure_bear']:
            simulated_sl = last_row['close'] + atr_component
        else:
            simulated_sl = 0
        pos_qty = calculate_position_size(pair, last_row['close'], simulated_sl, account_balance, risk_frac) if risk_frac > 0 and simulated_sl and simulated_sl != last_row['close'] else 0.0
        if pos_qty <= 0:
            reasons.append("Position size 0 (vol filter / stop too tight / balance small)")

        passed = (
            (ctx['structure_bull'] and ctx['long_score'] >= LONG_SCORE_THRESHOLD and ctx['bull_htf'] > 0)
            or (ctx['structure_bear'] and ctx['short_score'] >= SHORT_SCORE_THRESHOLD and ctx['bear_htf'] > 0)
        ) and risk_frac > 0

        diag_row = {
            'timestamp': last_ts.isoformat(),
            'pair': pair,
            'price': float(last_row['close']),
            'ema20': ctx['ema20'], 'ema50': ctx['ema50'], 'ema_slope': ctx['ema_slope'],
            'rsi': ctx['rsi'], 'atr': ctx['atr'], 'atr_pct': ctx['atr_pct'],
            'atr_low': ctx['atr_low'], 'atr_high': ctx['atr_high'], 'atr_regime': ctx['atr_regime'],
            'vol': ctx['volume'], 'vol_ma20': ctx['volume_ma20'],
            'volume_ratio': ctx['volume_ratio'], 'volume_ok': ctx['volume_ok'],
            'bias_1h': ctx['bias_1h'], 'bias_4h': ctx['bias_4h'],
            'htf_slope_1h': ctx['htf_slopes'].get('1H'), 'htf_slope_4h': ctx['htf_slopes'].get('4H'),
            'bull_score': ctx['long_score'], 'bear_score': ctx['short_score'],
            'pattern': ctx['pattern_info']['pattern'], 'pattern_bias': ctx['pattern_info']['bias'],
            'bull_htf': ctx['bull_htf'], 'bear_htf': ctx['bear_htf'],
            'risk_frac': risk_frac, 'pos_qty': pos_qty, 'pullback_ok': ctx['pullback_ok'],
            'passed': passed, 'reasons': "; ".join(reasons) if reasons else "OK",
        }
        append_named_csv('diagnostics', diag_row)

        # Verbose console print
        print("\n--- VERBOSE DIAGNOSTICS ---")
        print(f"Time (UTC): {last_ts}  Price: {last_row['close']:.2f}")
        print(f"  - Structure bull: {ctx['structure_bull']}  bear: {ctx['structure_bear']}")
        print(f"  - 1H bias bullish: {ctx['bias_1h']}  4H bias bullish: {ctx['bias_4h']}")
        print(f"  - RSI: {ctx['rsi']:.2f}  ATR: {ctx['atr']:.4f} (pct {ctx['atr_pct']:.4f})")
        print(f"  - Volume OK: {ctx['volume_ok']} (ratio {ctx['volume_ratio']:.2f})")
        print(f"  - Pattern: {ctx['pattern_info']['pattern']} ({ctx['pattern_info']['bias']})")
        print(f"  - Scores -> bull: {ctx['long_score']} bear: {ctx['short_score']}")
        print(f"  - Risk fraction: {risk_frac:.4f}")
        print("  -> Pos qty:", pos_qty)
        print("  -> Passed:", passed)
        print("  -> Reasons:", diag_row['reasons'])
        print("--- end diag ---\n")

        close_score = None
        threshold = LONG_SCORE_THRESHOLD
        if ctx['structure_bull']:
            close_score = ctx['long_score']; threshold = LONG_SCORE_THRESHOLD
        elif ctx['structure_bear']:
            close_score = ctx['short_score']; threshold = SHORT_SCORE_THRESHOLD

        if SEND_DIAGNOSTICS_TO_TELEGRAM and should_send_diag(pair, ctx, passed, close_score, threshold):
            brief = (
                f"🩺 DIAG {pair} {last_ts.strftime('%Y-%m-%d %H:%M')} price {last_row['close']:.2f}"
                f"\nScores bull/bear: {ctx['long_score']}/{ctx['short_score']}"
                f"\nATR% {ctx['atr_pct']:.4f} ({ctx['atr_regime']}) VolRatio {ctx['volume_ratio']:.2f}"
                f"\nResult: {'OK' if passed else 'SKIP'}"
            )
            if diag_row['reasons'] != 'OK':
                brief += f"\nReasons: {diag_row['reasons']}"
            send_telegram(brief)

        return diag_row
    except Exception as e:
        log_event(f"diagnose_candle error: {e}")
        return {}

# === Signal Generation & Diagnostics ===
def check_signal(symbol, df):
    try:
        if df is None or len(df) < 50:
            print(f"[{symbol}] ❌ Not enough data to analyze.")
            return None
        signals_df, _ = generate_signals_from_df(symbol, df)
        if signals_df.empty:
            pattern_info = detect_candlestick_pattern(df.iloc[-2], df.iloc[-1])
            print(f"[{symbol}] 🔍 No actionable setup (pattern {pattern_info['pattern']}, bias {pattern_info['bias']}).")
            return None
        latest_signal = signals_df.iloc[-1]
        reason = (
            f"Score {latest_signal.get('score')} {latest_signal['signal']} with pattern {latest_signal.get('pattern')}"
            f" | RSI {latest_signal.get('rsi', float('nan')):.2f}"
            f" | ATR% {latest_signal.get('atr_pct', float('nan')):.4f}"
            f" | VolRatio {latest_signal.get('volume_ratio', float('nan')):.2f}"
        )
        print(f"[{symbol}] 🔍 {reason}")
        return latest_signal['signal']
    except Exception as e:
        print(f"[{symbol}] ⚠️ Signal error: {e}")
        return None

# -----------------------
# Periodic summary
# -----------------------
def send_periodic_summary():
    global last_summary_time
    try:
        now_ts = now_local().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"📊 Codex Summary {now_ts} — Balance {account_balance:.2f} USDT"]
        lines.append(f"Monitored pairs: {', '.join(PAIRS)}")
        lines.append(f"Open trades: {len(open_trades)}")
        for t in list(open_trades.values())[:10]:
            remaining = t.get('remaining_qty', t.get('qty', 0.0))
            score = t.get('score')
            score_display = score if score is not None else 'n/a'
            risk_pct = t.get('risk_frac', 0.0) * 100
            lines.append(
                f"{t['pair']} {t['signal']} entry {t['entry']:.2f} qty {remaining:.6f} SL {t['sl']:.2f} TP1 {t['tp1']:.2f}"
                f" score {score_display} risk {risk_pct:.2f}%"
            )
        if closed_trades:
            last5 = list(closed_trades)[-5:]
            lines.append("\nRecent closed trades:")
            for c in last5:
                lines.append(f"{c['pair']} {c['signal']} -> {c['result']} net {c['net_pnl']:.2f}")
        send_telegram("\n".join(lines))
        last_summary_time = time.time()
    except Exception as e:
        log_event(f"send_periodic_summary error: {e}")

def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in {"true", "1", "yes", "y", "on"}:
        return True
    if val in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex strategy options")
    parser.add_argument("--diag_rate_limit_candles", type=int, default=DIAG_RATE_LIMIT_CANDLES)
    parser.add_argument("--enable_session_guard", type=parse_bool, default=ENABLE_SESSION_GUARD)
    parser.add_argument("--cooldown_after_sl_mult", type=float, default=COOLDOWN_AFTER_SL_MULT)
    parser.add_argument("--atr_regime_shift", type=parse_bool, default=ATR_REGIME_SHIFT_ENABLED)
    return parser.parse_args()

# -----------------------
# MAIN LOOP
# -----------------------
def generate_signals_from_df(pair: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Generate trade signals and return the evaluation context for further inspection."""
    signals: List[Dict[str, Any]] = []
    if df is None or len(df) < 50:
        return pd.DataFrame(signals), None

    ctx = build_signal_context(df)
    last = ctx['last']

    entry = float(last['close'])
    atr_component = ctx['atr'] if STOP_METHOD == 'atr' else entry * FIXED_SL_PCT
    risk_frac = ctx['risk_frac']

    long_threshold = LONG_SCORE_THRESHOLD
    short_threshold = SHORT_SCORE_THRESHOLD
    adjusted_risk_long = risk_frac
    adjusted_risk_short = risk_frac
    skip_reasons: List[str] = []

    if ATR_REGIME_SHIFT_ENABLED:
        if ctx['atr_regime'] == 'low':
            long_threshold += ATR_REGIME_LOW_SHIFT
            short_threshold += ATR_REGIME_LOW_SHIFT
        elif ctx['atr_regime'] == 'high':
            adjusted_risk_long *= max(0.0, 1 - ATR_REGIME_HIGH_RISK_REDUCTION)
            adjusted_risk_short *= max(0.0, 1 - ATR_REGIME_HIGH_RISK_REDUCTION)

    adjusted_risk_long, session_note_long = session_guard_adjust(last.name, adjusted_risk_long)
    adjusted_risk_short, session_note_short = session_guard_adjust(last.name, adjusted_risk_short)
    if session_note_long:  skip_reasons.append(session_note_long)
    if session_note_short and session_note_short not in skip_reasons:  skip_reasons.append(session_note_short)

    cooldown_block = in_sl_cooldown(pair, last.name)

    long_ok = (
        ctx['structure_bull']
        and ctx['bull_htf'] > 0
        and ctx['long_score'] >= long_threshold
        and ctx['volume_ok']
        and ctx['atr_in_range']
        and ctx['rsi'] >= RSI_LONG_MIN
        and adjusted_risk_long > 0
        and ctx['pullback_ok']
        and slopes_agree(ctx, 'LONG')
        and not cooldown_block
    )

    short_ok = (
        ctx['structure_bear']
        and ctx['bear_htf'] > 0
        and ctx['short_score'] >= short_threshold
        and ctx['volume_ok']
        and ctx['atr_in_range']
        and ctx['rsi'] <= RSI_SHORT_MAX
        and adjusted_risk_short > 0
        and ctx['pullback_ok']
        and slopes_agree(ctx, 'SHORT')
        and not cooldown_block
    )

    if long_ok:
        sl = entry - atr_component
        tp1 = entry + (entry - sl) * 1.5
        tp2 = entry + (entry - sl) * 3
        signals.append({
            'timestamp': last.name,
            'signal': 'LONG',
            'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'pattern': ctx['pattern_info']['pattern'], 'pattern_bias': ctx['pattern_info']['bias'],
            'rsi': ctx['rsi'], 'score': ctx['long_score'],
            'atr_pct': ctx['atr_pct'], 'volume_ratio': ctx['volume_ratio'],
            'htf_confluence': ctx['bull_htf'],
            'risk_frac': adjusted_risk_long, 'atr_regime': ctx['atr_regime'],
        })
    elif short_ok:
        sl = entry + atr_component
        tp1 = entry - (sl - entry) * 1.5
        tp2 = entry - (sl - entry) * 3
        signals.append({
            'timestamp': last.name,
            'signal': 'SHORT',
            'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'pattern': ctx['pattern_info']['pattern'], 'pattern_bias': ctx['pattern_info']['bias'],
            'rsi': ctx['rsi'], 'score': ctx['short_score'],
            'atr_pct': ctx['atr_pct'], 'volume_ratio': ctx['volume_ratio'],
            'htf_confluence': ctx['bear_htf'],
            'risk_frac': adjusted_risk_short, 'atr_regime': ctx['atr_regime'],
        })
    else:
        if cooldown_block: skip_reasons.append('cooldown_active')
        if not ctx['pullback_ok'] and (ctx['structure_bull'] or ctx['structure_bear']): skip_reasons.append('pullback_filter')
        if not slopes_agree(ctx, 'LONG') and ctx['structure_bull']:  skip_reasons.append('htf_slope_mismatch_long')
        if not slopes_agree(ctx, 'SHORT') and ctx['structure_bear']: skip_reasons.append('htf_slope_mismatch_short')
        if adjusted_risk_long <= 0 or adjusted_risk_short <= 0:       skip_reasons.append('risk_zero')
        if ctx['atr_regime'] == 'low':  skip_reasons.append('atr_regime_low')
        if ctx['atr_regime'] == 'high': skip_reasons.append('atr_regime_high')

    ctx['long_threshold'] = long_threshold
    ctx['short_threshold'] = short_threshold
    ctx['adjusted_risk_long'] = adjusted_risk_long
    ctx['adjusted_risk_short'] = adjusted_risk_short
    ctx['skip_reasons'] = skip_reasons

    return pd.DataFrame(signals), ctx

def main_loop():
    global last_summary_time, next_summary_time
    log_event("🚀 Codex v3.0 starting main loop...")

    ensure_daily_csvs()
    account_path = CSV_PATHS.get('account')
    if account_path and (not os.path.exists(account_path) or os.path.getsize(account_path) == 0):
        append_named_csv('account', {'ts': now_utc().isoformat(), 'balance': account_balance, 'net_pnl': 0.0})

    while True:
        try:
            ensure_daily_csvs()
            for pair in PAIRS:
                try:
                    df = get_klines_df(pair, INTERVAL, limit=500)
                except Exception as e:
                    log_event(f"fetch klines failed {pair}: {e}")
                    continue

                latest_ts = df.index[-1]
                current_price = df['close'].iloc[-1]
                candle_close = latest_ts + pd.Timedelta(seconds=interval_seconds(INTERVAL))
                now_ts = pd.Timestamp(now_utc())

                # --- Monitor open trades ---
                check_trades_and_notify(pair, current_price)

                if now_ts < candle_close:
                    log_event(f"[{pair}] Candle open; monitored trades; price {current_price:.2f}")
                    time.sleep(PRICE_POLL_SECONDS)
                    continue

                # --- Candle closed: diagnose + generate signals ---
                signals_df, ctx = generate_signals_from_df(pair, df)
                diag = diagnose_candle(pair, df, ctx)
                if not signals_df.empty:
                    latest_signal = signals_df.iloc[-1].to_dict()
                    ts_key = f"{pair}_{latest_signal['timestamp']}"
                    if diag and not diag.get('passed', False):
                        log_event(f"[{pair}] Signal found but diagnostics blocked: {diag.get('reasons')}")
                    else:
                        if last_sent_signal_ts.get(pair) != ts_key:
                            entry, sl, tp1, tp2 = latest_signal['entry'], latest_signal['sl'], latest_signal['tp1'], latest_signal['tp2']
                            risk_frac = latest_signal.get('risk_frac', RISK_PER_TRADE)
                            qty = calculate_position_size(pair, entry, sl, account_balance, risk_frac)
                            if qty <= 0:
                                send_telegram(f"⚠️ {pair} - qty calculated 0 (stop too tight or balance small).")
                                log_event(f"{pair} qty 0 -> skipped")
                            else:
                                pattern = latest_signal.get('pattern') or 'n/a'
                                pattern_bias = latest_signal.get('pattern_bias') or 'neutral'
                                rsi_val = latest_signal.get('rsi')
                                score = latest_signal.get('score')
                                atr_pct = latest_signal.get('atr_pct')
                                volume_ratio = latest_signal.get('volume_ratio')
                                msg = (
                                    f"🚨 {pair} NEW {latest_signal['signal']}"
                                    f"\nEntry: {entry:.2f}"
                                    f"\nTP1: {tp1:.2f}"
                                    f"\nTP2: {tp2:.2f}"
                                    f"\nSL: {sl:.2f}"
                                    f"\nQty(sim): {qty:.6f}"
                                    f"\nPattern: {pattern} ({pattern_bias})"
                                )
                                if rsi_val is not None and not math.isnan(rsi_val): msg += f"\nRSI: {rsi_val:.2f}"
                                if score is not None and isinstance(score, (int, float)): msg += f"\nScore: {int(round(float(score)))}"
                                if atr_pct is not None and isinstance(atr_pct, (int, float)) and not math.isnan(float(atr_pct)): msg += f"\nATR%: {float(atr_pct):.4f}"
                                if volume_ratio is not None and isinstance(volume_ratio, (int, float)) and not math.isnan(float(volume_ratio)): msg += f"\nVolRatio: {float(volume_ratio):.2f}"
                                htf_conf = latest_signal.get('htf_confluence')
                                if htf_conf is not None: msg += f"\nHTF confluence: {htf_conf}"
                                msg += f"\nRisk: {risk_frac * 100:.2f}% of balance"
                                msg += f"\nTime(candle close): {ts_to_local_str(latest_signal['timestamp'])}"
                                send_telegram(msg)
                                meta = {
                                    'pattern': pattern, 'pattern_bias': pattern_bias,
                                    'rsi': rsi_val, 'score': score, 'atr_pct': atr_pct, 'volume_ratio': volume_ratio,
                                    'htf_confluence': latest_signal.get('htf_confluence'),
                                    'risk_frac': risk_frac, 'atr_regime': latest_signal.get('atr_regime'),
                                }
                                open_simulated_trade(pair, latest_signal['signal'], entry, sl, tp1, tp2, meta)
                                last_sent_signal_ts[pair] = ts_key
                        else:
                            log_event(f"{pair} No new closed-candle signal (already sent for this candle).")
                else:
                    reasons = ctx.get('skip_reasons', []) if ctx else []
                    if reasons:
                        log_event(f"[{pair}] No signals this closed candle. Reasons: {', '.join(sorted(set(reasons)))}")
                    else:
                        log_event(f"[{pair}] No signals this closed candle.")

                time.sleep(2)

            # End-of-cycle: periodic summary
            if SEND_PERIODIC_SUMMARY:
                now_time = time.time()
                if now_time >= next_summary_time:
                    send_periodic_summary()
                    while next_summary_time <= now_time:
                        next_summary_time += SUMMARY_INTERVAL_S

            log_event(f"Cycle complete. Sleeping {CYCLE_WAIT_SECONDS}s. Balance {account_balance:.2f}")
            time.sleep(CYCLE_WAIT_SECONDS)

        except KeyboardInterrupt:
            log_event("Interrupted by user — exiting.")
            break
        except Exception as e:
            log_event(f"Main loop error: {e}")
            try:
                send_telegram(f"⚠️ Codex v3 error: {e}")
            except Exception:
                pass
            time.sleep(15)

if __name__ == "__main__":
    args = parse_args()
    DIAG_RATE_LIMIT_CANDLES = max(1, args.diag_rate_limit_candles)
    diagnostic_cooldown = {pair: DIAG_RATE_LIMIT_CANDLES for pair in PAIRS}
    ENABLE_SESSION_GUARD = bool(args.enable_session_guard)
    COOLDOWN_AFTER_SL_MULT = float(args.cooldown_after_sl_mult)
    ATR_REGIME_SHIFT_ENABLED = bool(args.atr_regime_shift)
    main_loop()

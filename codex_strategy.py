# codex_v3.py
# Codex v3.0 — Multi-pair live monitor + diagnostics-to-Telegram + virtual capital + TP/SL monitoring + CSV logbook
# Requirements:
#   pip install python-binance pandas numpy requests pytz

import time
import math
from datetime import datetime, timedelta, timezone
import pytz
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
try:
    from binance.client import Client
except Exception:
    Client = None

# -----------------------
# CONFIG - tweak these
# -----------------------
# Binance keys (optional for public data; required only if you later enable test/trading)
BINANCE_API_KEY = ""           # put keys if needed (optional)
BINANCE_API_SECRET = ""        # optional

# Telegram (required to receive alerts)
TELEGRAM_TOKEN = "8421566316:AAHEc8RsvjPZTXS9BIYbQ5__n92MLTiut68"
CHAT_ID = "6055044818"

# Pairs to monitor (primary pair first)
PAIRS = ["ETHUSDT" , "BTCUSDT" , "SOLUSDT"]

# Timeframe for signals
INTERVAL = "15m"

# Timezone (for human timestamps)
TZ = "Africa/Nairobi"
EAT = pytz.timezone(TZ)

# Virtual capital & risk
STARTING_BALANCE = 100000.0       # USDT starting balance (per account)
account_balance = STARTING_BALANCE
RISK_PER_TRADE = 0.02          # fraction of balance risked per trade

# Strategy tuning
REQUIRE_1H_AGREEMENT = False
REQUIRE_4H_AGREEMENT = False
RSI_LONG_MIN = 45
RSI_SHORT_MAX = 55
VOLUME_FACTOR = 1.0

# Stop / TP config
STOP_METHOD = 'atr'            # 'atr' or 'fixed'
FIXED_SL_PCT = 0.005           # 0.5% if using fixed
TP_STYLE = 'two'               # 'one' or 'two'
TP1_SPLIT = 0.5                # proportion closed at TP1 when TP_STYLE=='two'

# Notifications
SEND_DIAGNOSTICS_TO_TELEGRAM = True
SEND_PERIODIC_SUMMARY = True
SUMMARY_INTERVAL_S = 60 * 1   # every hour

# CSV files
OPEN_TRADES_CSV = "open_trades.csv"
CLOSED_TRADES_CSV = "closed_trades.csv"
DIAGNOSTICS_CSV = "diagnostics.csv"
ACCOUNT_LOG_CSV = "account_log.csv"
EVENTS_LOG = "events.log"

# Polling
PRICE_POLL_SECONDS = 60
CYCLE_WAIT_SECONDS = 60 * 15   # main loop sleeps until next candle

# Execution flags
ENABLE_TESTNET = False         # leave False — using live data only
MAX_CONCURRENT_TRADES_PER_SYMBOL = 1

# Fees (for simulation)
FEE_RATE = 0.001  # round-trip approx (adjust)

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
def now_local() -> datetime:
    return datetime.now(tz=pytz.utc).astimezone(EAT)

def ts_to_local_str(ts) -> str:
    t = pd.to_datetime(ts)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    return t.tz_convert(EAT).strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(text: str) -> dict:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print("Telegram failed:", r.status_code, r.text)
        try:
            return r.json()
        except Exception:
            return {"ok": False}
    except Exception as e:
        print("Telegram error:", e)
        return {"ok": False, "error": str(e)}

def append_csv(path: str, row: Dict[str, Any], columns: Optional[List[str]] = None):
    df_row = pd.DataFrame([row])
    try:
        if columns is not None:
            df_row = df_row[columns]
    except Exception:
        pass
    try:
        with open(path, 'a', encoding='utf-8') as f:
            df_row.to_csv(f, header=f.tell()==0, index=False)
    except Exception as e:
        print("CSV write error:", e)

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
# Indicators / strategy functions
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
    o = df_small['open'].resample(rule).first()
    h = df_small['high'].resample(rule).max()
    l = df_small['low'].resample(rule).min()
    c = df_small['close'].resample(rule).last()
    v = df_small['volume'].resample(rule).sum()
    res = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    return res.dropna()

# -----------------------
# Live data helpers
# -----------------------
def get_klines_df(symbol: str, interval: str = INTERVAL, limit: int = 500) -> pd.DataFrame:
    # prefer python-binance client if available; otherwise public REST
    try:
        if client is not None:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        else:
            url = "https://api.binance.com/api/v3/klines"
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
            r.raise_for_status()
            klines = r.json()
        df = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume','close_time','quote_asset_volume','num_trades','taker_buy_base','taker_buy_quote','ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        return df[['timestamp','open','high','low','close','volume']].set_index('timestamp')
    except Exception as e:
        log_event(f"get_klines_df error for {symbol}: {e}")
        raise

def get_current_price(symbol: str) -> float:
    try:
        if client is not None:
            t = client.get_symbol_ticker(symbol=symbol)
            return float(t['price'])
        else:
            r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            return float(r.json()['price'])
    except Exception as e:
        log_event(f"get_current_price error {symbol}: {e}")
        raise

# -----------------------
# Position sizing & account simulation
# -----------------------
open_trades: Dict[str, Dict] = {}
closed_trades: List[Dict] = []
last_sent_signal_ts: Dict[str, Optional[str]] = {pair: None for pair in PAIRS}
last_summary_time = 0

def calculate_position_size(entry_price: float, stop_loss: float, balance: float, risk_frac: float) -> float:
    distance = abs(entry_price - stop_loss)
    if distance <= 0:
        return 0.0
    risk_amount = balance * risk_frac
    qty = risk_amount / distance
    # round down to sensible precision (6 decimals for ETH-like)
    return float(math.floor(qty * 1e6) / 1e6)

def open_simulated_trade(pair: str, signal: str, entry: float, sl: float, tp1: float, tp2: float):
    global account_balance
    ts_key = f"{pair}_{entry}_{datetime.utcnow().isoformat()}"
    # concurrency protection
    count_open = sum(1 for t in open_trades.values() if t['pair']==pair)
    if count_open >= MAX_CONCURRENT_TRADES_PER_SYMBOL:
        send_telegram(f"⚠️ Skipping open for {pair}: {count_open} open trades already.")
        log_event(f"Skip open {pair}: concurrency")
        return None

    qty = calculate_position_size(entry, sl, account_balance, RISK_PER_TRADE)
    notional = qty * entry
    est_fees = notional * FEE_RATE
    trade = {
        'id': ts_key,
        'pair': pair,
        'signal': signal,
        'entry': entry,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'qty': qty,
        'notional': notional,
        'est_fees': est_fees,
        'created_at': datetime.now(tz=pytz.utc).isoformat(),
        'tp1_hit': False,
        'status': 'OPEN'
    }
    open_trades[ts_key] = trade
    append_csv(OPEN_TRADES_CSV, trade)
    send_telegram(f"📥 OPEN_SIM {pair} {signal} qty {qty:.6f} @ {entry:.2f} SL {sl:.2f} TP1 {tp1:.2f} TP2 {tp2:.2f}")
    log_event(f"Opened {pair} {signal} qty {qty:.6f} @ {entry:.2f}")
    return trade

def close_simulated_trade(trade_id: str, result: str, close_price: float):
    global account_balance
    trade = open_trades.pop(trade_id, None)
    if not trade:
        return None
    qty = trade['qty']; entry = trade['entry']; signal = trade['signal']; notional = trade['notional']
    if qty <= 0:
        log_event("Close skipped: qty 0")
        return None
    gross = (close_price - entry) * qty if signal == 'LONG' else (entry - close_price) * qty
    fees = notional * FEE_RATE
    net = gross - fees
    account_balance += net
    closed = {**trade, 'closed_at': datetime.now(tz=pytz.utc).isoformat(), 'result': result, 'close_price': close_price, 'gross_pnl': gross, 'fees': fees, 'net_pnl': net, 'balance_after': account_balance}
    closed_trades.append(closed)
    append_csv(CLOSED_TRADES_CSV, closed)
    append_csv(ACCOUNT_LOG_CSV, {'ts': datetime.now(tz=pytz.utc).isoformat(), 'balance': account_balance, 'net_pnl': net})
    send_telegram(f"📤 CLOSE {trade['pair']} {signal} {result} @ {close_price:.2f} net {net:.2f} bal {account_balance:.2f}")
    log_event(f"Closed {trade['pair']} {signal} {result} @ {close_price:.2f} net {net:.2f}")
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
                # optionally move SL to breakeven etc. (not implemented yet)
            elif current_price <= sl:
                messages.append(f"❌ SL HIT {pair} @ {current_price:.2f}")
                close_simulated_trade(tid, 'SL', current_price)
            else:
                # progress notifications
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
            elif current_price >= sl:
                messages.append(f"❌ SL HIT {pair} @ {current_price:.2f}")
                close_simulated_trade(tid, 'SL', current_price)
            else:
                if not trade['tp1_hit'] and (current_price - tp1) / tp1 <= 0.005:
                    messages.append(f"📉 {pair} price approaching TP1 ({tp1:.2f}) price {current_price:.2f}")
                if (sl - current_price) / sl <= 0.004:
                    messages.append(f"⚠️ {pair} price approaching SL ({sl:.2f}) price {current_price:.2f}")
    for m in messages:
        send_telegram(m)
        log_event(m)
    return messages

# -----------------------
# Diagnostics logic (per closed candle)
# -----------------------
def diagnose_candle(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verbose diagnostics: returns same diag_row but also prints the exact checks.
    Use this to understand why signals are blocked.
    """
    try:
        last_ts = df.index[-1]
        last_row = df.iloc[-1]
        ema20 = float(ema(df['close'],20).iloc[-1])
        ema50 = float(ema(df['close'],50).iloc[-1])
        rsi_v = float(rsi(df['close'],14).iloc[-1])
        atr_v = float(atr(df,14).iloc[-1]) if len(df)>0 else 0.0
        vol = float(last_row['volume'])
        vol_ma20 = float(df['volume'].rolling(20, min_periods=1).mean().iloc[-1])
        vol_spike = vol > vol_ma20 * VOLUME_FACTOR

        # higher TF biases
        df1h = resample_tf(df, '1h'); df1h['ema20'] = ema(df1h['close'],20); df1h['ema50'] = ema(df1h['close'],50)
        df4h = resample_tf(df, '4h'); df4h['ema20'] = ema(df4h['close'],20); df4h['ema50'] = ema(df4h['close'],50)
        try:
            bias_1h = bool((df1h['ema20'] > df1h['ema50']).reindex(df.index, method='ffill').loc[last_ts])
            bias_4h = bool((df4h['ema20'] > df4h['ema50']).reindex(df.index, method='ffill').loc[last_ts])
        except Exception:
            bias_1h = False; bias_4h = False

        cond_15m_bull = ema20 > ema50
        cond_15m_bear = ema20 < ema50

        checks = []
        # collect each check (True/False + reason)
        checks.append(("15m EMA bull", cond_15m_bull))
        checks.append(("15m EMA bear", cond_15m_bear))
        checks.append(("1H bias bullish", bias_1h))
        checks.append(("4H bias bullish", bias_4h))
        checks.append((f"RSI >= {RSI_LONG_MIN}", rsi_v >= RSI_LONG_MIN))
        checks.append((f"RSI <= {RSI_SHORT_MAX}", rsi_v <= RSI_SHORT_MAX))
        checks.append((f"Volume spike (vol {vol:.2f} vs ma {vol_ma20:.2f})", vol_spike))

        reasons = []
        # same logic as before but with added detailed conditions
        if not (cond_15m_bull or cond_15m_bear):
            reasons.append("15m EMA neutral")
        else:
            if cond_15m_bull:
                if REQUIRE_1H_AGREEMENT and not bias_1h: reasons.append("1H not bullish")
                if REQUIRE_4H_AGREEMENT and not bias_4h: reasons.append("4H not bullish")
                if rsi_v < RSI_LONG_MIN: reasons.append(f"RSI low {rsi_v:.1f}")
            if cond_15m_bear:
                if REQUIRE_1H_AGREEMENT and bias_1h: reasons.append("1H bullish (blocks short)")
                if REQUIRE_4H_AGREEMENT and bias_4h: reasons.append("4H bullish (blocks short)")
                if rsi_v > RSI_SHORT_MAX: reasons.append(f"RSI high {rsi_v:.1f}")
            if not vol_spike: reasons.append("No volume spike")

        simulated_sl = last_row['close'] - atr_v if STOP_METHOD=='atr' else last_row['close'] - last_row['close']*FIXED_SL_PCT
        pos_qty = calculate_position_size(last_row['close'], simulated_sl, account_balance, RISK_PER_TRADE) if simulated_sl and simulated_sl != last_row['close'] else 0.0
        if pos_qty <= 0:
            reasons.append("Position size 0 (stop too tight / balance small)")

        passed = len(reasons) == 0
        diag_row = {
            'timestamp': last_ts.isoformat(),
            'price': float(last_row['close']),
            'ema20': ema20, 'ema50': ema50,
            'rsi': rsi_v, 'atr': atr_v,
            'vol': vol, 'vol_ma20': vol_ma20, 'vol_spike': vol_spike,
            'bias_1h': bias_1h, 'bias_4h': bias_4h,
            'pos_qty': pos_qty,
            'passed': passed,
            'reasons': "; ".join(reasons) if reasons else "OK"
        }
        append_csv(DIAGNOSTICS_CSV, diag_row)

        # PRINT full verbose diagnostics to console (immediately visible)
        print("\n--- VERBOSE DIAGNOSTICS ---")
        print(f"Time (UTC): {last_ts}  Price: {last_row['close']:.2f}")
        for k,v in checks:
            print(f"  - {k}: {v}")
        print("  -> Pos qty:", pos_qty)
        print("  -> Passed:", passed)
        print("  -> Reasons:", diag_row['reasons'])
        print("--- end diag ---\n")

        # send to telegram (brief) only if configured or if near pass
        if SEND_DIAGNOSTICS_TO_TELEGRAM:
            brief = f"🩺 DIAG {last_ts.strftime('%Y-%m-%d %H:%M')} price {last_row['close']:.2f} -> {'OK' if passed else 'SKIP'}"
            if not passed:
                brief += f"\nReasons: {diag_row['reasons']}"
            send_telegram(brief)

        return diag_row
    except Exception as e:
        log_event(f"diagnose_candle error: {e}")
        return {}

# -----------------------
# Signal generator (multi-TF) -> returns DataFrame of signals
# -----------------------
# === Signal Generation & Diagnostics ===
def check_signal(symbol, df):
    try:
        # Basic safety check
        if df is None or len(df) < 50:
            print(f"[{symbol}] ❌ Not enough data to analyze.")
            return None

        # Calculate indicators
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_diff'] = df['EMA20'] - df['EMA50']

        # Latest candle
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Condition checks
        crossed_up = prev['EMA_diff'] < 0 and last['EMA_diff'] > 0
        crossed_down = prev['EMA_diff'] > 0 and last['EMA_diff'] < 0

        # Diagnostics
        reason = ""
        if crossed_up:
            signal = "BUY"
            reason = "EMA20 crossed above EMA50 (bullish)"
        elif crossed_down:
            signal = "SELL"
            reason = "EMA20 crossed below EMA50 (bearish)"
        else:
            signal = None
            reason = "No EMA cross signal"

        # Log diagnostics
        with open("diagnostics.csv", "a") as f:
            f.write(f"{datetime.now()}, {symbol}, {signal}, {reason}, EMA20={last['EMA20']:.2f}, EMA50={last['EMA50']:.2f}\n")

        print(f"[{symbol}] 🔍 {reason}")

        return signal
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
            lines.append(f"{t['pair']} {t['signal']} entry {t['entry']:.2f} qty {t['qty']:.6f} SL {t['sl']:.2f} TP1 {t['tp1']:.2f}")
        if closed_trades:
            last5 = closed_trades[-5:]
            lines.append("\nRecent closed trades:")
            for c in last5:
                lines.append(f"{c['pair']} {c['signal']} -> {c['result']} net {c['net_pnl']:.2f}")
        send_telegram("\n".join(lines))
        last_summary_time = time.time()
    except Exception as e:
        log_event(f"send_periodic_summary error: {e}")

# -----------------------
# Main loop
# -----------------------
# -----------------------
# MAIN LOOP
# -----------------------
def generate_signals_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trade signals from a closed candle DataFrame.
    Returns DataFrame with timestamp, signal, entry, sl, tp1, tp2.
    """
    signals = []
    if df is None or len(df) < 50:
        return pd.DataFrame(signals)

    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_diff'] = df['EMA20'] - df['EMA50']

    last = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_up = prev['EMA_diff'] < 0 and last['EMA_diff'] > 0
    crossed_down = prev['EMA_diff'] > 0 and last['EMA_diff'] < 0

    if crossed_up:
        signal = 'LONG'
        entry = last['close']
        sl = entry - atr(df,14).iloc[-1] if STOP_METHOD=='atr' else entry - entry*FIXED_SL_PCT
        tp1 = entry + (entry - sl) * 1.5
        tp2 = entry + (entry - sl) * 3
        signals.append({'timestamp': last.name, 'signal': signal, 'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2})
    elif crossed_down:
        signal = 'SHORT'
        entry = last['close']
        sl = entry + atr(df,14).iloc[-1] if STOP_METHOD=='atr' else entry + entry*FIXED_SL_PCT
        tp1 = entry - (sl - entry) * 1.5
        tp2 = entry - (sl - entry) * 3
        signals.append({'timestamp': last.name, 'signal': signal, 'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2})

    return pd.DataFrame(signals)

def main_loop():
    global last_summary_time
    log_event("🚀 Codex v3.0 starting main loop...")
    
    # Ensure CSV headers exist
    append_csv(OPEN_TRADES_CSV, {}, columns=['id','pair','signal','entry','sl','tp1','tp2','qty','notional','est_fees','created_at'])
    append_csv(CLOSED_TRADES_CSV, {}, columns=['id','pair','signal','entry','sl','tp1','tp2','qty','notional','est_fees','created_at','closed_at','result','close_price','gross_pnl','fees','net_pnl','balance_after'])
    append_csv(DIAGNOSTICS_CSV, {}, columns=['timestamp','price','ema20','ema50','rsi','atr','vol','vol_ma20','vol_spike','bias_1h','bias_4h','pos_qty','passed','reasons'])
    append_csv(ACCOUNT_LOG_CSV, {'ts': datetime.now(tz=pytz.utc).isoformat(), 'balance': account_balance}, columns=['ts','balance'])

    while True:
        try:
            for pair in PAIRS:
                try:
                    df = get_klines_df(pair, INTERVAL, limit=500)
                except Exception as e:
                    log_event(f"fetch klines failed {pair}: {e}")
                    continue

                latest_ts = df.index[-1]
                now_utc = datetime.now(tz=timezone.utc)
                candle_close = pd.to_datetime(latest_ts).tz_localize('UTC') + timedelta(minutes=int(INTERVAL[:-1]) if INTERVAL.endswith('m') else 15)
                current_price = df['close'].iloc[-1]

                # --- Monitor open trades ---
                check_trades_and_notify(pair, current_price)  # simulated trades

                if now_utc < candle_close:
                    log_event(f"[{pair}] Candle open; monitored trades; price {current_price:.2f}")
                    if SEND_PERIODIC_SUMMARY and (time.time() - last_summary_time) > SUMMARY_INTERVAL_S:
                        send_periodic_summary()
                    time.sleep(PRICE_POLL_SECONDS)
                    continue

                # --- Candle closed: diagnose + generate signals ---
                diag = diagnose_candle(df)
                
                # Simulated trade signal
                signals_df = generate_signals_from_df(df)
                if not signals_df.empty:
                    latest_signal = signals_df.iloc[-1].to_dict()
                    ts_key = f"{pair}_{latest_signal['timestamp']}"
                    if diag and not diag.get('passed', False):
                        log_event(f"[{pair}] Signal found but diagnostics blocked: {diag.get('reasons')}")
                    else:
                        if last_sent_signal_ts.get(pair) != ts_key:
                            entry, sl, tp1, tp2 = latest_signal['entry'], latest_signal['sl'], latest_signal['tp1'], latest_signal['tp2']
                            qty = calculate_position_size(entry, sl, account_balance, RISK_PER_TRADE)
                            if qty <= 0:
                                send_telegram(f"⚠️ {pair} - qty calculated 0 (stop too tight or balance small).")
                                log_event(f"{pair} qty 0 -> skipped")
                            else:
                                send_telegram(f"🚨 {pair} NEW {latest_signal['signal']}\nEntry: {entry:.2f}\nTP1: {tp1:.2f}\nTP2: {tp2:.2f}\nSL: {sl:.2f}\nQty(sim): {qty:.6f}\nTime(candle close): {ts_to_local_str(latest_signal['timestamp'])}")
                                open_simulated_trade(pair, latest_signal['signal'], entry, sl, tp1, tp2)
                                last_sent_signal_ts[pair] = ts_key
                                log_event(f"{pair} signal sent and simulated open.")
                        else:
                            log_event(f"{pair} No new closed-candle signal (already sent for this candle).")
                else:
                    log_event(f"[{pair}] No signals this closed candle.")

                # Small delay to avoid rate limit
                time.sleep(2)

            # End-of-cycle: periodic summary
            if SEND_PERIODIC_SUMMARY and (time.time() - last_summary_time) > SUMMARY_INTERVAL_S:
                send_periodic_summary()
            
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
    main_loop()


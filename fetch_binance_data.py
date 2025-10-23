import requests
import pandas as pd

symbol = "ETHUSDT"
interval = "15m"
limit = 500  # about 5 days of candles

url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

print("⏳ Fetching data from Binance...")
data = requests.get(url).json()

df = pd.DataFrame(data, columns=[
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
])

df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df.to_csv("ETHUSDT_15m.csv", index=False)

print(f"✅ Saved {len(df)} candles to ETHUSDT_15m.csv")

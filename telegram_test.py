import requests

TOKEN = "8421566316:AAHEc8RsvjPZTXS9BIYbQ5__n92MLTiut68"  # your bot token
CHAT_ID = "6055044818"  # your chat id

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": "🧠 Telegram direct test message"}

r = requests.post(url, data=payload)
print("Status code:", r.status_code)
print("Response:", r.text)

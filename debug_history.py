"""Run on VPS: python3 debug_history.py"""
import hmac, hashlib, time, requests
from pathlib import Path
from dotenv import dotenv_values

env  = dotenv_values(Path(__file__).parent / ".env")
key    = env["BINANCE_API_KEY"]
secret = env["BINANCE_API_SECRET"]
base   = "https://demo-fapi.binance.com/fapi/v1"

def signed_get(path, params):
    p = dict(params)
    p["timestamp"]  = int(time.time() * 1000)
    p["recvWindow"] = 5000
    qs  = "&".join(f"{k}={v}" for k, v in p.items())
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r   = requests.get(f"{base}{path}?{qs}&signature={sig}",
                       headers={"X-MBX-APIKEY": key}, timeout=30)
    data = r.json()
    count = len(data) if isinstance(data, list) else "N/A"
    sample = data[0] if isinstance(data, list) and data else data
    print(f"{path:30s}  status={r.status_code}  count={count}")
    print(f"  {sample}")
    print()

now = int(time.time() * 1000)
d30 = now - 30  * 24 * 3600 * 1000
d89 = now - 89  * 24 * 3600 * 1000

signed_get("/userTrades",  {"symbol": "BTCUSDT", "limit": 5})
signed_get("/allOrders",   {"symbol": "BTCUSDT", "limit": 5})
signed_get("/allOrders",   {"symbol": "BTCUSDT", "limit": 5, "startTime": d30})
signed_get("/allOrders",   {"symbol": "BTCUSDT", "limit": 5, "startTime": d89})

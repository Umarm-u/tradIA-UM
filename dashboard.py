"""
tradIA Dashboard API
====================
Read-only Flask server exposing bot state, trade history, performance metrics,
and live price data for the frontend dashboard.

This file is independent of all trading logic — it only READS output files:
  - live/bot_state.json   → current bot position & daily stats
  - logs/tradia_*.log     → activity & signal logs
  - results/*.csv         → backtest trade history
  - results/*.txt         → backtest performance reports
  - Binance public API    → live candles & mark price (no auth)

DOES NOT modify any trading logic, models, or core backend files.
"""
import csv
import json
import os
import re
import glob
import hmac
import hashlib
import time as _time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="dashboard", static_url_path="")
CORS(app)

PROJECT_ROOT = Path(__file__).parent
STATE_FILE    = PROJECT_ROOT / "live"    / "bot_state.json"
LOG_DIR       = PROJECT_ROOT / "logs"
RESULTS_DIR   = PROJECT_ROOT / "results"
ENV_FILE      = PROJECT_ROOT / ".env"

BINANCE_FUTURES         = "https://fapi.binance.com/fapi/v1"
BINANCE_FUTURES_TESTNET = "https://demo-fapi.binance.com/fapi/v1"

def _futures_base() -> str:
    """Return the correct Futures REST base URL based on the .env BINANCE_TESTNET flag."""
    testnet = _read_env().get("BINANCE_TESTNET", "True").lower() in ("true", "1", "yes")
    return BINANCE_FUTURES_TESTNET if testnet else BINANCE_FUTURES

ALLOWED_SYMBOLS   = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}
ALLOWED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}

HISTORY_CACHE = PROJECT_ROOT / "live" / "binance_history_cache.json"
CACHE_TTL     = 300  # seconds before re-fetching from Binance

# ── Utility helpers ────────────────────────────────────────────────────────────

def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_csv(path: Path) -> list:
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception:
        pass
    return rows


def _parse_report(path: Path) -> dict:
    """Parse key: value pairs from a backtest report text file."""
    metrics = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" not in line:
                    continue
                key, _, raw = line.partition(":")
                key  = key.strip()
                raw  = raw.strip()
                if not key:
                    continue
                try:
                    metrics[key] = float(raw)
                except ValueError:
                    metrics[key] = raw
    except Exception:
        pass
    return metrics


def _read_env() -> dict:
    env = {}
    try:
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def _write_env(updates: dict) -> None:
    """Safely update allowed keys in .env without touching other values."""
    ALLOWED = {
        "RISK_PER_TRADE", "MAX_DAILY_LOSS",
        "BUY_THRESHOLD",  "SELL_THRESHOLD",
        "DRY_RUN",        "LEVERAGE",
    }
    lines = []
    try:
        with open(ENV_FILE, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        pass

    written = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in ALLOWED and k in updates:
            new_lines.append(f"{k}={updates[k]}\n")
            written.add(k)
        else:
            new_lines.append(line)

    for k, v in updates.items():
        if k in ALLOWED and k not in written:
            new_lines.append(f"{k}={v}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


# ── Static files ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


# ── API: Bot status ────────────────────────────────────────────────────────────

@app.route("/api/status")
def get_status():
    state = _read_json(STATE_FILE)
    if state is None:
        return jsonify({
            "bot_running":     False,
            "position":        None,
            "daily_pnl":       0.0,
            "daily_trades":    0,
            "last_updated":    None,
            "last_candle_time": None,
            "trailing_state":  None,
        })
    return jsonify({
        "bot_running":     True,
        "position":        state.get("position"),
        "daily_pnl":       _safe_float(state.get("daily_pnl", 0)),
        "daily_trades":    _safe_int(state.get("daily_trades", 0)),
        "last_updated":    state.get("timestamp_human"),
        "last_candle_time": state.get("last_candle_time"),
        "trailing_state":  state.get("trailing_state"),
    })


# ── API: Performance metrics ───────────────────────────────────────────────────

@app.route("/api/performance")
def get_performance():
    buy_r  = _parse_report(RESULTS_DIR / "backtest_report_buy_enhanced.txt")
    sell_r = _parse_report(RESULTS_DIR / "backtest_report_sell_enhanced.txt")

    # Derive combined stats from report data (authoritative).
    # The CSV files contain all raw signal candidates, not just the final
    # backtest trades, so using them produces wrong totals/win-rates.
    buy_n   = int(_safe_float(buy_r.get("total_trades", 0)))
    sell_n  = int(_safe_float(sell_r.get("total_trades", 0)))
    total   = buy_n + sell_n

    buy_wr  = _safe_float(buy_r.get("win_rate", 0))
    sell_wr = _safe_float(sell_r.get("win_rate", 0))
    # Weighted-average win rate across both strategies
    combined_wr = (buy_n * buy_wr + sell_n * sell_wr) / total if total else 0

    buy_ret  = _safe_float(buy_r.get("total_return_pct", 0))
    sell_ret = _safe_float(sell_r.get("total_return_pct", 0))
    # Average total return across both strategies
    combined_return = (buy_ret + sell_ret) / 2 if total else 0

    return jsonify({
        "buy":  buy_r,
        "sell": sell_r,
        "combined": {
            "total_trades":     total,
            "win_rate":         combined_wr,
            "total_return_pct": combined_return,
        },
    })


# ── API: Trade history ─────────────────────────────────────────────────────────

@app.route("/api/trades")
def get_trades():
    buy_t  = _read_csv(RESULTS_DIR / "trades_buy_enhanced.csv")
    sell_t = _read_csv(RESULTS_DIR / "trades_sell_enhanced.csv")

    trades = []
    for t in buy_t + sell_t:
        trades.append({
            "entry_time":   t.get("entry_time",  ""),
            "exit_time":    t.get("exit_time",   ""),
            "direction":    t.get("direction",   ""),
            "entry_price":  _safe_float(t.get("entry_price",  0)),
            "exit_price":   _safe_float(t.get("exit_price",   0)),
            "sl_price":     _safe_float(t.get("sl_price",     0)),
            "tp_price":     _safe_float(t.get("tp_price",     0)),
            "exit_reason":  t.get("exit_reason",  ""),
            "net_pnl_pct":  _safe_float(t.get("net_pnl_pct",  0)),
            "gross_pnl_pct": _safe_float(t.get("gross_pnl_pct", 0)),
            "confidence":   _safe_float(t.get("confidence",  0)),
            "holding_bars": _safe_int(t.get("holding_bars",  0)),
            "trail_activated": t.get("trail_activated", "False").lower() == "true",
        })

    trades.sort(key=lambda x: x["entry_time"])
    return jsonify(trades)


# ── API: Live trades (parsed from log files) ──────────────────────────────────

@app.route("/api/live_trades")
def get_live_trades():
    """
    Parse completed live trades from bot log files.

    Looks for these log patterns written by order_manager.py & run_live_bot.py:
      - 'OPENING (LONG|SHORT) TRADE'     → start of opening block
      - '  SL:  $X'  / '  TP:  $X'      → SL/TP from the opening block
      - '  Quantity: X'                  → position size from opening block
      - '(LONG|SHORT) trade opened @ $X' → confirmed entry (single line)
      - 'POSITION CLOSED: (LONG|SHORT) | PnL: $X' → closed trade
      - 'Trail active: True/False'        → whether trailing stop fired
      - 'Signal: (LONG|SHORT) | BUY=X | SELL=Y' → confidence scores
    """
    if not LOG_DIR.exists():
        return jsonify([])

    # ── Patterns ──────────────────────────────────────────────────────────────
    TS_RE      = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
    OPEN_RE    = re.compile(r'(LONG|SHORT) trade opened @ \$(\d+\.?\d*)', re.I)
    CLOSE_RE   = re.compile(r'POSITION CLOSED:\s*(LONG|SHORT)\s*\|\s*PnL:\s*\$(-?\d+\.?\d*)', re.I)
    DIR_RE     = re.compile(r'OPENING\s+(LONG|SHORT)\s+TRADE', re.I)
    SL_RE      = re.compile(r'\bSL:\s*\$(\d+\.?\d*)')
    TP_RE      = re.compile(r'\bTP:\s*\$(\d+\.?\d*)')
    QTY_RE     = re.compile(r'Quantity:\s*(\d+\.?\d*)')
    TRAIL_RE   = re.compile(r'Trail active:\s*(True|False)', re.I)
    SIGNAL_RE  = re.compile(r'Signal:\s*(LONG|SHORT|NO_SIGNAL)\s*\|\s*BUY=(\d+\.?\d*)\s*\|\s*SELL=(\d+\.?\d*)', re.I)

    completed   = []
    current     = None   # trade being built
    in_block    = False  # inside a multi-line OPENING block
    last_signal = None   # most-recent signal details
    current_ts  = None   # timestamp of current log line

    log_files = sorted(LOG_DIR.glob("tradia_*.log"))

    for log_file in log_files:
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue

                    # Grab timestamp and reset block flag on new log record
                    ts_m = TS_RE.match(line)
                    if ts_m:
                        current_ts = ts_m.group(1)
                        in_block   = False  # new record = out of multi-line block

                    # ── Signal confidence ───────────────────────────────────
                    m = SIGNAL_RE.search(line)
                    if m:
                        last_signal = {
                            "direction":   m.group(1).upper(),
                            "buy_proba":   _safe_float(m.group(2)),
                            "sell_proba":  _safe_float(m.group(3)),
                            "ts":          current_ts,
                        }

                    # ── Opening block start ─────────────────────────────────
                    m = DIR_RE.search(line)
                    if m:
                        in_block  = True
                        direction = m.group(1).upper()
                        # Attach confidence if last signal matches
                        conf = None
                        if last_signal and last_signal["direction"] == direction:
                            conf = max(last_signal["buy_proba"], last_signal["sell_proba"])
                        current = {
                            "direction":   direction,
                            "entry_time":  current_ts or "",
                            "entry_price": None,
                            "sl_price":    None,
                            "tp_price":    None,
                            "quantity":    None,
                            "confidence":  conf,
                            "source":      "live",
                        }

                    # ── Harvest SL/TP/Qty from the multi-line opening block ─
                    if in_block and current is not None:
                        m = SL_RE.search(line)
                        if m and current["sl_price"] is None:
                            current["sl_price"] = _safe_float(m.group(1))

                        m = TP_RE.search(line)
                        if m and current["tp_price"] is None:
                            current["tp_price"] = _safe_float(m.group(1))

                        m = QTY_RE.search(line)
                        if m and current["quantity"] is None:
                            current["quantity"] = _safe_float(m.group(1))

                    # ── Trade confirmed open (single-line, has timestamp) ───
                    m = OPEN_RE.search(line)
                    if m:
                        direction    = m.group(1).upper()
                        entry_price  = _safe_float(m.group(2))
                        if current is None or current["direction"] != direction:
                            # Create trade object if we missed the opening block
                            conf = None
                            if last_signal and last_signal["direction"] == direction:
                                conf = max(last_signal["buy_proba"], last_signal["sell_proba"])
                            current = {
                                "direction":   direction,
                                "entry_time":  current_ts or "",
                                "entry_price": None,
                                "sl_price":    None,
                                "tp_price":    None,
                                "quantity":    None,
                                "confidence":  conf,
                                "source":      "live",
                            }
                        current["entry_price"] = entry_price
                        if current_ts:
                            current["entry_time"] = current_ts
                        in_block = False

                    # ── Position closed ─────────────────────────────────────
                    m = CLOSE_RE.search(line)
                    if m:
                        direction = m.group(1).upper()
                        pnl_usdt  = _safe_float(m.group(2))

                        if current and current["direction"] == direction:
                            current["exit_time"]      = current_ts or ""
                            current["pnl_usdt"]       = pnl_usdt
                            current["trail_activated"] = False  # will be updated by TRAIL_RE
                            completed.append(current)
                        else:
                            # Orphan close (bot restarted mid-trade)
                            completed.append({
                                "direction":   direction,
                                "entry_time":  "",
                                "exit_time":   current_ts or "",
                                "entry_price": None,
                                "sl_price":    None,
                                "tp_price":    None,
                                "quantity":    None,
                                "pnl_usdt":    pnl_usdt,
                                "confidence":  None,
                                "trail_activated": False,
                                "source":      "live",
                            })
                        current  = None
                        in_block = False

                    # ── Trail active annotation for the last completed trade ─
                    m = TRAIL_RE.search(line)
                    if m and completed:
                        completed[-1]["trail_activated"] = (m.group(1).lower() == "true")

        except Exception:
            pass

    # Sort completed trades by entry_time ascending
    completed.sort(key=lambda t: t.get("entry_time") or "")
    return jsonify(completed)


# ── API: Equity curve ─────────────────────────────────────────────────────────

@app.route("/api/equity")
def get_equity():
    buy_t  = _read_csv(RESULTS_DIR / "trades_buy_enhanced.csv")
    sell_t = _read_csv(RESULTS_DIR / "trades_sell_enhanced.csv")

    all_t = []
    for t in buy_t + sell_t:
        try:
            all_t.append({
                "exit_time":   t.get("exit_time", ""),
                "entry_time":  t.get("entry_time", ""),
                "net_pnl_pct": _safe_float(t.get("net_pnl_pct", 0)),
                "direction":   t.get("direction", ""),
            })
        except Exception:
            pass

    if not all_t:
        return jsonify([])

    all_t.sort(key=lambda x: x["entry_time"])

    equity = 100.0
    curve  = [{"time": all_t[0]["entry_time"], "value": round(equity, 4)}]
    for t in all_t:
        equity *= 1 + t["net_pnl_pct"] / 100
        curve.append({"time": t["exit_time"], "value": round(equity, 4)})

    return jsonify(curve)


# ── API: Logs ─────────────────────────────────────────────────────────────────

@app.route("/api/logs")
def get_logs():
    logs = []

    log_files = sorted(LOG_DIR.glob("tradia_*.log"), reverse=True) if LOG_DIR.exists() else []

    pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+\[(\w+)\s*\]\s+\[.*?\]\s+(.*)"
    )

    for log_file in log_files[:3]:  # last 3 days max
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = pattern.match(line)
                    if m:
                        logs.append({
                            "timestamp": m.group(1),
                            "level":     m.group(2).strip(),
                            "message":   m.group(3),
                        })
        except Exception:
            pass

    # Limit to 300 most-recent entries, newest first
    logs = logs[-300:]
    logs.reverse()
    return jsonify(logs)


# ── API: Live candles (Binance public proxy) ───────────────────────────────────

@app.route("/api/candles")
def get_candles():
    symbol   = request.args.get("symbol",   "BTCUSDT")
    interval = request.args.get("interval", "15m")
    limit    = request.args.get("limit",    "200")

    if symbol   not in ALLOWED_SYMBOLS:   symbol   = "BTCUSDT"
    if interval not in ALLOWED_INTERVALS: interval = "15m"
    try:
        limit = min(int(limit), 500)
    except Exception:
        limit = 200

    try:
        resp = requests.get(
            f"{_futures_base()}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        candles = [
            {
                "time":   k[0] // 1000,
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            }
            for k in raw
        ]
        return jsonify(candles)
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ── API: Live price (Binance public proxy) ────────────────────────────────────

@app.route("/api/price")
def get_price():
    symbol = request.args.get("symbol", "BTCUSDT")
    if symbol not in ALLOWED_SYMBOLS:
        symbol = "BTCUSDT"
    try:
        # /markPrice is unavailable on demo-fapi testnet; /premiumIndex
        # returns the same markPrice field and works on both environments.
        resp = requests.get(
            f"{_futures_base()}/premiumIndex",
            params={"symbol": symbol},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return jsonify({
            "symbol":    data.get("symbol"),
            "price":     _safe_float(data.get("markPrice", 0)),
            "timestamp": data.get("time"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ── API: 24h ticker (Binance public proxy) ────────────────────────────────────

@app.route("/api/ticker24h")
def get_ticker24h():
    symbol = request.args.get("symbol", "BTCUSDT")
    if symbol not in ALLOWED_SYMBOLS:
        symbol = "BTCUSDT"
    try:
        resp = requests.get(
            f"{_futures_base()}/ticker/24hr",
            params={"symbol": symbol},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return jsonify({
            "priceChange":    _safe_float(data.get("priceChange")),
            "priceChangePct": _safe_float(data.get("priceChangePercent")),
            "highPrice":      _safe_float(data.get("highPrice")),
            "lowPrice":       _safe_float(data.get("lowPrice")),
            "volume":         _safe_float(data.get("volume")),
            "quoteVolume":    _safe_float(data.get("quoteVolume")),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ── API: Config read/write ────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    env = _read_env()
    return jsonify({
        "risk_per_trade":  _safe_float(env.get("RISK_PER_TRADE", "0.02")),
        "max_daily_loss":  _safe_float(env.get("MAX_DAILY_LOSS",  "0.05")),
        "buy_threshold":   _safe_float(env.get("BUY_THRESHOLD",  "0.600")),
        "sell_threshold":  _safe_float(env.get("SELL_THRESHOLD", "0.775")),
        "dry_run":         env.get("DRY_RUN", "True").lower() in ("true", "1", "yes"),
        "leverage":        _safe_int(env.get("LEVERAGE", "5")),
        "symbol":          env.get("SYMBOL", "BTCUSDT"),
    })


@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    VALIDATORS = {
        "risk_per_trade": ("RISK_PER_TRADE",  lambda v: 0 < _safe_float(v) <= 0.10),
        "max_daily_loss": ("MAX_DAILY_LOSS",   lambda v: 0 < _safe_float(v) <= 0.25),
        "buy_threshold":  ("BUY_THRESHOLD",    lambda v: 0.4 <= _safe_float(v) < 1.0),
        "sell_threshold": ("SELL_THRESHOLD",   lambda v: 0.4 <= _safe_float(v) < 1.0),
        "dry_run":        ("DRY_RUN",          lambda v: str(v).lower() in ("true", "false")),
        "leverage":       ("LEVERAGE",         lambda v: 1 <= _safe_int(v) <= 125),
    }

    updates = {}
    for json_key, (env_key, validator) in VALIDATORS.items():
        if json_key not in data:
            continue
        val = data[json_key]
        try:
            if not validator(val):
                return jsonify({"error": f"Invalid value for {json_key}"}), 400
        except Exception as exc:
            return jsonify({"error": f"Invalid value for {json_key}: {exc}"}), 400
        updates[env_key] = str(val)

    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400

    if not ENV_FILE.exists():
        return jsonify({"error": ".env file not found — cannot persist changes"}), 404

    _write_env(updates)
    return jsonify({"success": True, "updated": list(updates.keys())})


# ── Authenticated Binance helpers ─────────────────────────────────────────────

def _binance_signed_get(path: str, params: dict) -> any:
    """Authenticated GET to Binance Futures; respects BINANCE_TESTNET flag."""
    env        = _read_env()
    api_key    = env.get("BINANCE_API_KEY", "")
    api_secret = env.get("BINANCE_API_SECRET", "")
    testnet    = env.get("BINANCE_TESTNET", "True").lower() in ("true", "1", "yes")
    base       = ("https://demo-fapi.binance.com/fapi/v1" if testnet
                  else "https://fapi.binance.com/fapi/v1")

    p = dict(params)
    p["timestamp"]  = int(_time.time() * 1000)
    p["recvWindow"] = 5000

    query_string = urlencode(p)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    resp = requests.get(
        f"{base}{path}?{query_string}&signature={signature}",
        headers={"X-MBX-APIKEY": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _sync_history() -> dict:
    """Return cached Binance account history, refreshing if older than CACHE_TTL."""
    cache = _read_json(HISTORY_CACHE) or {}
    now   = _time.time()

    if cache.get("synced_at") and (now - cache["synced_at"]) < CACHE_TTL:
        return cache

    symbol = _read_env().get("SYMBOL", "BTCUSDT")

    def _fetch(path, params):
        try:
            return _binance_signed_get(path, params), None
        except Exception as exc:
            return None, str(exc)

    trades,   err1 = _fetch("/userTrades", {"symbol": symbol, "limit": 1000})
    inc_pnl,  err2 = _fetch("/income", {"symbol": symbol, "incomeType": "REALIZED_PNL",  "limit": 1000})
    inc_comm, err3 = _fetch("/income", {"symbol": symbol, "incomeType": "COMMISSION",    "limit": 1000})
    inc_fund, err4 = _fetch("/income", {"symbol": symbol, "incomeType": "FUNDING_FEE",   "limit": 1000})

    # If the fill endpoint returns nothing (demo testnet wipes fill history but
    # retains order history), fall back to /allOrders for the last 89 days.
    if isinstance(trades, list) and len(trades) == 0:
        start_ms = int((_time.time() - 89 * 24 * 3600) * 1000)
        orders, _ = _fetch("/allOrders", {"symbol": symbol, "limit": 1000, "startTime": start_ms})
        if isinstance(orders, list):
            filled = [o for o in orders
                      if o.get("status") == "FILLED" and o.get("type") != "LIQUIDATION"]
            trades = [{
                "id":              o.get("orderId"),
                "orderId":         o.get("orderId"),
                "side":            o.get("side", ""),
                "positionSide":    o.get("positionSide", "BOTH"),
                "price":           _safe_float(o.get("avgPrice", 0)),
                "qty":             _safe_float(o.get("executedQty", 0)),
                "quoteQty":        _safe_float(o.get("cumQuote", 0)),
                "realizedPnl":     _safe_float(o.get("realizedPnl", 0)),
                "commission":      0.0,
                "commissionAsset": "USDT",
                "maker":           False,
                "time":            o.get("updateTime", o.get("time", 0)),
                "buyer":           o.get("side") == "BUY",
            } for o in filled]
            # Derive income_pnl from orders when the income endpoint is also empty
            if isinstance(inc_pnl, list) and len(inc_pnl) == 0:
                inc_pnl = [
                    {
                        "symbol":     symbol,
                        "incomeType": "REALIZED_PNL",
                        "income":     str(t["realizedPnl"]),
                        "asset":      "USDT",
                        "time":       t["time"],
                        "tradeId":    str(t["orderId"]),
                    }
                    for t in trades
                    if t["realizedPnl"] != 0.0
                ]

    # Only surface errors from critical endpoints (trades + realized PnL).
    # Commission and funding-fee endpoints return 400 on demo-fapi testnet
    # because those income types are unsupported there — treat as non-fatal.
    errors = [e for e in (err1, err2) if e]

    # Merge live results with existing cache so history is never lost.
    # New entries from the API are added; cached entries not returned by the
    # API (e.g. after a demo-testnet wipe) are kept.  Dedup by "id" / "tradeId".
    def _merge(live, key, id_field):
        cached   = cache.get(key, [])
        if not isinstance(live, list):
            return cached
        existing = {str(e.get(id_field)) for e in cached if e.get(id_field) is not None}
        merged   = list(cached)
        for entry in live:
            if str(entry.get(id_field)) not in existing:
                merged.append(entry)
        return sorted(merged, key=lambda e: e.get("time", 0))

    new_cache = {
        "synced_at":    now,
        "synced_human": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "trades":       _merge(trades,   "trades",      "id"),
        "income_pnl":   _merge(inc_pnl,  "income_pnl",  "tradeId"),
        "income_comm":  _merge(inc_comm, "income_comm", "tradeId"),
        "income_fund":  _merge(inc_fund, "income_fund", "tradeId"),
        "error":        errors[0] if errors else None,
    }

    tmp = HISTORY_CACHE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(new_cache, f)
        tmp.replace(HISTORY_CACHE)
    except Exception:
        pass

    return new_cache


def _compute_account_summary(cache: dict) -> dict:
    """Derive all stats and chart series from cached Binance data."""
    pnl_entries  = cache.get("income_pnl",  [])
    comm_entries = cache.get("income_comm", [])
    fund_entries = cache.get("income_fund", [])

    pnl_vals   = [_safe_float(e.get("income", 0)) for e in pnl_entries]
    wins       = [v for v in pnl_vals if v > 0]
    losses     = [v for v in pnl_vals if v < 0]
    closed     = len(wins) + len(losses)

    gross_pnl  = sum(wins)
    gross_loss = sum(losses)
    total_pnl  = sum(pnl_vals)
    total_comm = sum(_safe_float(e.get("income", 0)) for e in comm_entries)
    total_fund = sum(_safe_float(e.get("income", 0)) for e in fund_entries)
    net_profit = total_pnl + total_comm + total_fund

    win_rate   = len(wins) / closed      if closed          else 0.0
    avg_win    = gross_pnl  / len(wins)  if wins            else 0.0
    avg_loss   = gross_loss / len(losses) if losses         else 0.0
    rr         = abs(avg_win / avg_loss) if avg_loss != 0   else 0.0

    # Cumulative PnL curve (one point per closed position)
    sorted_pnl = sorted(pnl_entries, key=lambda e: e.get("time", 0))
    cum, curve = 0.0, []
    for e in sorted_pnl:
        cum += _safe_float(e.get("income", 0))
        curve.append({"time": e["time"] // 1000, "value": round(cum, 4)})

    # Monthly breakdown
    monthly = {}
    for e in pnl_entries:
        dt  = datetime.fromtimestamp(e.get("time", 0) / 1000, tz=timezone.utc)
        key = dt.strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0.0) + _safe_float(e.get("income", 0))
    monthly_list = [{"month": k, "pnl": round(v, 4)} for k, v in sorted(monthly.items())]

    return {
        "total_closed_trades": closed,
        "total_fills":         len(cache.get("trades", [])),
        "wins":                len(wins),
        "losses":              len(losses),
        "win_rate":            round(win_rate,   4),
        "gross_pnl":           round(gross_pnl,  4),
        "gross_loss":          round(gross_loss, 4),
        "total_realized_pnl":  round(total_pnl,  4),
        "total_commission":    round(total_comm, 4),
        "total_funding":       round(total_fund, 4),
        "net_profit":          round(net_profit, 4),
        "avg_win":             round(avg_win,    4),
        "avg_loss":            round(avg_loss,   4),
        "rr_ratio":            round(rr,         4),
        "curve":               curve,
        "monthly":             monthly_list,
        "synced_at":           cache.get("synced_human", "—"),
        "error":               cache.get("error"),
    }


# ── Static: history page ───────────────────────────────────────────────────────

@app.route("/history")
def history_page():
    return send_from_directory("dashboard", "history.html")


# ── API: Account summary (stats + chart series) ────────────────────────────────

@app.route("/api/account/summary")
def get_account_summary():
    cache = _sync_history()
    return jsonify(_compute_account_summary(cache))


# ── API: All trade fills ───────────────────────────────────────────────────────

@app.route("/api/account/trades")
def get_account_trades():
    cache  = _sync_history()
    trades = cache.get("trades", [])
    result = []
    for t in trades:
        result.append({
            "time":            t.get("time", 0),
            "id":              t.get("id"),
            "orderId":         t.get("orderId"),
            "side":            t.get("side", ""),
            "positionSide":    t.get("positionSide", "BOTH"),
            "price":           _safe_float(t.get("price", 0)),
            "qty":             _safe_float(t.get("qty", 0)),
            "quoteQty":        _safe_float(t.get("quoteQty", 0)),
            "realizedPnl":     _safe_float(t.get("realizedPnl", 0)),
            "commission":      _safe_float(t.get("commission", 0)),
            "commissionAsset": t.get("commissionAsset", "USDT"),
            "maker":           t.get("maker", False),
        })
    result.sort(key=lambda x: x["time"])
    return jsonify(result)


# ── API: Income history (PnL + commissions + funding) ─────────────────────────

@app.route("/api/account/income")
def get_account_income():
    cache  = _sync_history()
    income = (
        cache.get("income_pnl",  []) +
        cache.get("income_comm", []) +
        cache.get("income_fund", [])
    )
    income.sort(key=lambda x: x.get("time", 0))
    return jsonify(income)


# ── API: Force-refresh account history cache ──────────────────────────────────

@app.route("/api/account/sync", methods=["POST"])
def force_sync_account():
    cache = _read_json(HISTORY_CACHE) or {}
    cache["synced_at"] = 0          # invalidate TTL
    tmp = HISTORY_CACHE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        tmp.replace(HISTORY_CACHE)
    except Exception:
        pass

    cache   = _sync_history()
    summary = _compute_account_summary(cache)
    return jsonify({
        "success":   True,
        "synced_at": summary["synced_at"],
        "error":     summary["error"],
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  tradIA Dashboard API")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)

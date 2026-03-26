"""
tradIA Live Trading Configuration
Loads settings from environment variables (.env file).
All sensitive credentials are kept out of source code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ─────────────────────────────────────────────
# Binance API
# ─────────────────────────────────────────────
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ─────────────────────────────────────────────
# Trading Pair & Timeframe
# ─────────────────────────────────────────────
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TIMEFRAME = "15m"                          # must match training data
TIMEFRAME_MINUTES = 15
LEVERAGE = int(os.getenv("LEVERAGE", "5"))

# ─────────────────────────────────────────────
# Mode
# ─────────────────────────────────────────────
DRY_RUN = os.getenv("DRY_RUN", "True").lower() in ("true", "1", "yes")

# ─────────────────────────────────────────────
# Signal Thresholds (from training)
# ─────────────────────────────────────────────
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.600"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.775"))

# Meta-model filter
META_FILTER_ENABLED = os.getenv("META_FILTER_ENABLED", "True").lower() in ("true", "1", "yes")
META_THRESHOLD = float(os.getenv("META_THRESHOLD", "0.30"))

# ─────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))   # 2% of account
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.05"))    # 5% max daily loss
MIN_TRADE_USDT = 100.0    # Binance Futures minimum notional requirement
MIN_QUANTITY = 0.001      # minimum BTC quantity on Binance Futures

# ─────────────────────────────────────────────
# SL / TP (from backtester config)
# ─────────────────────────────────────────────
# These match the values used in training / backtesting
from src.config import (
    BACKTEST_SL_ATR_MULT,
    BACKTEST_TP_ATR_MULT,
    SELL_TP_RATIO,
    BUY_TP_RATIO,
    SELL_TRAILING_STOP, SELL_TRAILING_ACTIVATION, SELL_TRAILING_DISTANCE,
    BUY_TRAILING_STOP, BUY_TRAILING_ACTIVATION, BUY_TRAILING_DISTANCE,
    BACKTEST_COMMISSION, BACKTEST_SLIPPAGE,
)

# ─────────────────────────────────────────────
# Data Pipeline
# ─────────────────────────────────────────────
CANDLE_BUFFER_SIZE = 500   # historical candles to maintain (warmup for rolling features)
CANDLE_FETCH_DELAY = 5     # seconds to wait after candle close before fetching

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_FILE = PROJECT_ROOT / "live" / "bot_state.json"

LOG_DIR.mkdir(exist_ok=True)
(PROJECT_ROOT / "live").mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Retry / Resilience
# ─────────────────────────────────────────────
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2        # seconds (doubles each retry)
API_TIMEOUT = 30            # seconds

# ─────────────────────────────────────────────
# Session Filter (UTC hours — same as training)
# ─────────────────────────────────────────────
SESSION_FILTER_ENABLED = True
SESSION_HOURS = [(7, 11), (13, 17)]   # London + NY kill zones

"""
tradIA Configuration
Central configuration for all pipeline parameters.
"""
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "FINAL (1).csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
DATETIME_COL = "open_time"
OHLC_COLS = ["open", "high", "low", "close"]
TIMEFRAME_MINUTES = 15

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
SWING_PERIOD = 5               # candles on each side to confirm swing
ATR_PERIODS = [14, 50]         # ATR lookback windows
RETURN_PERIODS = [1, 4, 16, 48]  # return lookback windows
VOLATILITY_PERIODS = [20, 50]  # rolling std windows
FVG_LOOKBACK = 100             # max candles to track active FVGs
LIQUIDITY_CLUSTER_THRESHOLD = 0.001  # price difference threshold for equal highs/lows (0.1%)
LIQUIDITY_LOOKBACK = 96        # candles to scan for liquidity clusters (24 hours)

# Multi-timeframe aggregation
HTF_PERIODS = {
    "1h": 4,    # 4 × 15m = 1 hour
    "4h": 16,   # 16 × 15m = 4 hours
}

# ─────────────────────────────────────────────
# Label Generation (Triple Barrier)
# ─────────────────────────────────────────────
TP_MULTIPLIER = 2.0       # take profit = TP_MULT × ATR
SL_MULTIPLIER = 2.0       # stop loss = SL_MULT × ATR
MAX_HOLD_CANDLES = 32     # maximum holding period (8 hours on 15m)
MIN_DISPLACEMENT = 1.5    # minimum displacement_ratio to consider a setup

# -----------------------------------------------
# SMC-Aligned Labeling
# -----------------------------------------------
SMC_SETUP_WINDOW = 40          # candles to scan for setup completion
SMC_MIN_GAP = 8                # minimum candles between signals (2 hours)
SMC_VALIDATION_TP = 1.5        # ATR multiplier for outcome validation TP
SMC_VALIDATION_SL = 1.0        # ATR multiplier for outcome validation SL
SMC_VALIDATION_HOLD = 32       # max hold for outcome validation
SMC_CISD_THRESHOLD = 2.0       # range_expansion threshold for CISD

# -----------------------------------------------
# SELL Strategy (Buyside Liquidity Sweep)
# -----------------------------------------------
SELL_SWING_PERIOD = 5                # candles each side for swing high confirmation
SELL_ROLLING_HIGH_PERIODS = [48, 96, 192]  # rolling N-period highs (12h, 24h, 48h)
SELL_DISPLACEMENT_WINDOW = 10        # max candles after sweep to find strong green candle
SELL_BODY_RATIO_MIN = 0.6           # min body/range ratio for displacement candle
SELL_UPPER_WICK_RATIO_MAX = 0.3     # max upper_wick/body ratio (70:30 rule)
SELL_PULLBACK_MIN = 1               # min bearish pullback candles
SELL_PULLBACK_MAX = 3               # max bearish pullback candles
SELL_TP_RATIO = 1.5                 # TP distance = 1.5 × SL distance (favorable R:R)
SELL_MIN_GAP = 8                    # minimum candles between SELL signals
SELL_SESSION_HIGH_HOURS = 8         # hours for session high calculation
SELL_VALIDATE_OUTCOME = True        # verify TP hit before SL for label quality

# Session filter: only trade during high-volatility kill zones (UTC hours)
# London: 07:00–11:00, New York: 13:00–17:00
SELL_SESSION_HOURS = [(7, 11), (13, 17)]  # (start_hour, end_hour) tuples in UTC
SELL_SESSION_FILTER = True           # enable/disable session filter

# Confluence
SELL_MIN_LIQUIDITY_LEVELS = 2       # min number of liquidity levels to confirm zone
SELL_MIN_DISPLACEMENT_ATR = 1.0     # displacement candle range >= 1.0 × ATR_14

# Trailing stop
SELL_TRAILING_STOP = True           # enable trailing stop
SELL_TRAILING_ACTIVATION = 0.5     # activate trail after price moves 50% toward TP
SELL_TRAILING_DISTANCE = 0.3       # trail distance = 30% of SL distance

# -----------------------------------------------
# BUY Strategy (Sellside Liquidity Sweep)
# -----------------------------------------------
BUY_SWING_PERIOD = 5                 # candles each side for swing low confirmation
BUY_ROLLING_LOW_PERIODS = [48, 96, 192]  # rolling N-period lows (12h, 24h, 48h)
BUY_DISPLACEMENT_WINDOW = 10        # max candles after sweep to find strong red candle
BUY_BODY_RATIO_MIN = 0.6            # min body/range ratio for displacement candle
BUY_LOWER_WICK_RATIO_MAX = 0.3      # max lower_wick/body ratio (70:30 rule)
BUY_PULLBACK_MIN = 1                # min bullish pullback candles
BUY_PULLBACK_MAX = 3                # max bullish pullback candles
BUY_TP_RATIO = 1.5                  # TP distance = 1.5 × SL distance (favorable R:R)
BUY_MIN_GAP = 8                     # minimum candles between BUY signals
BUY_SESSION_LOW_HOURS = 8           # hours for session low calculation
BUY_VALIDATE_OUTCOME = True         # verify TP hit before SL for label quality

# Session filter (same kill zones as SELL)
BUY_SESSION_HOURS = [(7, 11), (13, 17)]
BUY_SESSION_FILTER = True

# Confluence
BUY_MIN_LIQUIDITY_LEVELS = 2
BUY_MIN_DISPLACEMENT_ATR = 1.0

# Trailing stop
BUY_TRAILING_STOP = True
BUY_TRAILING_ACTIVATION = 0.5
BUY_TRAILING_DISTANCE = 0.3

# -----------------------------------------------
# Meta-Labeling
# -----------------------------------------------
META_THRESHOLD = 0.55           # minimum meta-model confidence
META_PROFIT_ATR_MULT = 0.5     # ATR multiplier to define "profitable" meta-label
META_PROFIT_HORIZON = 16       # candles to check for profitability
META_FEATURES = [
    "primary_proba",
    "volatility_20", "volatility_50",
    "displacement_ratio",
    "liquidity_swept_buy", "liquidity_swept_sell",
    "structure_bias_16", "structure_bias_48",
    "nearest_bull_fvg_dist", "nearest_bear_fvg_dist",
    "htf_alignment",
    "atr_14",
    "buying_pressure_8",
    "body_momentum_8",
    "range_expansion",
]

# -----------------------------------------------
# Signal Filtering
# -----------------------------------------------
SIGNAL_MODEL_PROB_THRESHOLD = 0.70
REQUIRE_STRUCTURE_ALIGNMENT = True
REQUIRE_DISPLACEMENT_OR_SWEEP = True
DISPLACEMENT_LOOKBACK = 4      # candles to check for recent displacement/sweep

# -----------------------------------------------
# Backtesting
# -----------------------------------------------
BACKTEST_COMMISSION = 0.0004   # taker fee per side (0.04%)
BACKTEST_SLIPPAGE = 0.0001     # estimated slippage (0.01%)
BACKTEST_TP_ATR_MULT = 2.0     # take profit in ATR multiples
BACKTEST_SL_ATR_MULT = 1.0     # stop loss in ATR multiples
BACKTEST_MAX_HOLD = 32         # maximum holding period (candles)
INITIAL_CAPITAL = 10000.0      # starting capital for equity curve

# -----------------------------------------------
# Model Training
# -----------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

WALK_FORWARD_FOLDS = 5       # number of walk-forward splits
SIGNAL_THRESHOLD = 0.60      # minimum probability to emit a signal

# LightGBM defaults
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
}

# LSTM defaults
LSTM_SEQUENCE_LENGTH = 64     # input sequence length
LSTM_HIDDEN_SIZE = 128        # hidden units per LSTM layer
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 256
LSTM_PATIENCE = 10            # early stopping patience

# Optuna
OPTUNA_N_TRIALS = 50

# ─────────────────────────────────────────────
# Random Seed
# ─────────────────────────────────────────────
RANDOM_SEED = 42

"""
tradIA Feature Engineering — ICT / Smart Money Concepts
Transforms raw OHLC data into features representing how a professional
SMC/ICT trader reads the market.
"""
import pandas as pd
import numpy as np
from typing import Optional

from src.config import (
    SWING_PERIOD, ATR_PERIODS, RETURN_PERIODS, VOLATILITY_PERIODS,
    FVG_LOOKBACK, LIQUIDITY_CLUSTER_THRESHOLD, LIQUIDITY_LOOKBACK,
    HTF_PERIODS
)


# ═══════════════════════════════════════════════════════════════
# 1. CANDLESTICK STRUCTURE
# ═══════════════════════════════════════════════════════════════

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Body, wick, and shape features from each candle."""
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]

    candle_range = h - l
    candle_range_safe = candle_range.replace(0, np.nan)  # avoid division by zero

    df["body_size"] = (c - o).abs() / c
    df["upper_wick"] = (h - np.maximum(o, c)) / c
    df["lower_wick"] = (np.minimum(o, c) - l) / c
    df["body_to_range"] = (c - o).abs() / candle_range_safe
    df["candle_range"] = candle_range / c
    df["is_bullish"] = (c > o).astype(int)

    # Doji detection (body < 10% of range)
    df["is_doji"] = (df["body_to_range"] < 0.10).astype(int)

    # Hammer / shooting star patterns
    df["is_hammer"] = (
        (df["lower_wick"] > df["body_size"] * 2) &
        (df["upper_wick"] < df["body_size"] * 0.5)
    ).astype(int)

    df["is_shooting_star"] = (
        (df["upper_wick"] > df["body_size"] * 2) &
        (df["lower_wick"] < df["body_size"] * 0.5)
    ).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
# 2. VOLATILITY & PRICE DISPLACEMENT
# ═══════════════════════════════════════════════════════════════

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """ATR, returns, volatility, and displacement detection."""
    h, l, c = df["high"], df["low"], df["close"]

    # True Range
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))

    # ATR at multiple periods
    for period in ATR_PERIODS:
        df[f"atr_{period}"] = tr.rolling(window=period, min_periods=period).mean()

    # Displacement ratio (current range vs ATR)
    df["displacement_ratio"] = df["candle_range"] / (df["atr_14"] / c)
    df["displacement_ratio"] = df["displacement_ratio"].replace([np.inf, -np.inf], np.nan)
    df["is_displacement"] = (df["displacement_ratio"] > 2.0).astype(int)

    # Returns at multiple lookback windows
    for period in RETURN_PERIODS:
        df[f"return_{period}"] = c.pct_change(periods=period)

    # Rolling volatility
    for period in VOLATILITY_PERIODS:
        df[f"volatility_{period}"] = c.pct_change().rolling(window=period).std()

    # Price momentum
    df["momentum_short"] = c / c.shift(4) - 1   # 1-hour momentum
    df["momentum_medium"] = c / c.shift(16) - 1  # 4-hour momentum
    df["momentum_long"] = c / c.shift(96) - 1    # 24-hour momentum

    return df


# ═══════════════════════════════════════════════════════════════
# 3. SWING HIGH / LOW DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_swing_points(df: pd.DataFrame, period: int = SWING_PERIOD) -> pd.DataFrame:
    """
    Detect swing highs and swing lows.
    A swing high has the highest 'high' within ±period candles.
    A swing low has the lowest 'low' within ±period candles.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = np.zeros(n, dtype=int)
    swing_low = np.zeros(n, dtype=int)

    for i in range(period, n - period):
        # Swing high: high[i] is the max of the window
        window_highs = highs[i - period: i + period + 1]
        if highs[i] == np.max(window_highs):
            swing_high[i] = 1

        # Swing low: low[i] is the min of the window
        window_lows = lows[i - period: i + period + 1]
        if lows[i] == np.min(window_lows):
            swing_low[i] = 1

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low

    # Track the last swing high/low price and distance
    last_sh_price = np.full(n, np.nan)
    last_sl_price = np.full(n, np.nan)
    bars_since_sh = np.full(n, np.nan)
    bars_since_sl = np.full(n, np.nan)

    last_sh = np.nan
    last_sl = np.nan
    count_sh = 0
    count_sl = 0

    for i in range(n):
        if swing_high[i] == 1:
            last_sh = highs[i]
            count_sh = 0
        if swing_low[i] == 1:
            last_sl = lows[i]
            count_sl = 0

        last_sh_price[i] = last_sh
        last_sl_price[i] = last_sl
        bars_since_sh[i] = count_sh
        bars_since_sl[i] = count_sl
        count_sh += 1
        count_sl += 1

    close_arr = df["close"].values
    df["last_swing_high_price"] = last_sh_price
    df["last_swing_low_price"] = last_sl_price
    df["dist_to_swing_high"] = (close_arr - last_sh_price) / close_arr
    df["dist_to_swing_low"] = (close_arr - last_sl_price) / close_arr
    df["bars_since_swing_high"] = bars_since_sh
    df["bars_since_swing_low"] = bars_since_sl

    return df


# ═══════════════════════════════════════════════════════════════
# 4. MARKET STRUCTURE / BREAK OF STRUCTURE (BOS)
# ═══════════════════════════════════════════════════════════════

def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS), Market Structure Shifts (MSS),
    and track the current market structure state.
    """
    n = len(df)
    close = df["close"].values
    last_sh = df["last_swing_high_price"].values
    last_sl = df["last_swing_low_price"].values

    bos_bullish = np.zeros(n, dtype=int)
    bos_bearish = np.zeros(n, dtype=int)
    market_structure = np.zeros(n, dtype=int)  # 1=bullish, -1=bearish, 0=ranging

    current_structure = 0
    prev_sh_broken = np.nan
    prev_sl_broken = np.nan

    for i in range(1, n):
        # Bullish BOS: close breaks above last swing high
        if not np.isnan(last_sh[i]) and close[i] > last_sh[i] and close[i - 1] <= last_sh[i]:
            bos_bullish[i] = 1
            current_structure = 1

        # Bearish BOS: close breaks below last swing low
        if not np.isnan(last_sl[i]) and close[i] < last_sl[i] and close[i - 1] >= last_sl[i]:
            bos_bearish[i] = 1
            current_structure = -1

        market_structure[i] = current_structure

    df["bos_bullish"] = bos_bullish
    df["bos_bearish"] = bos_bearish
    df["market_structure"] = market_structure

    # Market Structure Shift (change in structure)
    df["structure_shift"] = (df["market_structure"].diff().abs() > 0).astype(int)

    # Rolling structure bias (last N candles)
    df["structure_bias_16"] = df["market_structure"].rolling(16).mean()
    df["structure_bias_48"] = df["market_structure"].rolling(48).mean()

    return df


# ═══════════════════════════════════════════════════════════════
# 5. FAIR VALUE GAP (FVG) DETECTION
# ═══════════════════════════════════════════════════════════════

def add_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (imbalance between supply and demand).
    Bullish FVG: candle[i-2].high < candle[i].low
    Bearish FVG: candle[i-2].low > candle[i].high
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    close = df["close"].values

    bull_fvg = np.zeros(n, dtype=int)
    bear_fvg = np.zeros(n, dtype=int)

    # Detect FVGs
    for i in range(2, n):
        if highs[i - 2] < lows[i]:
            bull_fvg[i] = 1
        if lows[i - 2] > highs[i]:
            bear_fvg[i] = 1

    df["bull_fvg"] = bull_fvg
    df["bear_fvg"] = bear_fvg

    # Track active (unfilled) FVGs and compute distances
    active_bull_fvg_count = np.zeros(n)
    active_bear_fvg_count = np.zeros(n)
    nearest_bull_fvg_dist = np.full(n, np.nan)
    nearest_bear_fvg_dist = np.full(n, np.nan)
    in_fvg = np.zeros(n, dtype=int)

    # Store active FVGs as lists of (midpoint, upper, lower)
    bull_fvgs_active = []
    bear_fvgs_active = []

    for i in range(2, n):
        # Add new FVGs
        if bull_fvg[i]:
            fvg_lower = highs[i - 2]
            fvg_upper = lows[i]
            fvg_mid = (fvg_lower + fvg_upper) / 2
            bull_fvgs_active.append((fvg_mid, fvg_upper, fvg_lower))

        if bear_fvg[i]:
            fvg_upper = lows[i - 2]
            fvg_lower = highs[i]
            fvg_mid = (fvg_upper + fvg_lower) / 2
            bear_fvgs_active.append((fvg_mid, fvg_upper, fvg_lower))

        # Remove filled FVGs (price has traded through them)
        bull_fvgs_active = [
            (m, u, l) for m, u, l in bull_fvgs_active
            if lows[i] > l  # not yet filled
        ]
        bear_fvgs_active = [
            (m, u, l) for m, u, l in bear_fvgs_active
            if highs[i] < u  # not yet filled
        ]

        # Keep only recent FVGs
        bull_fvgs_active = bull_fvgs_active[-FVG_LOOKBACK:]
        bear_fvgs_active = bear_fvgs_active[-FVG_LOOKBACK:]

        active_bull_fvg_count[i] = len(bull_fvgs_active)
        active_bear_fvg_count[i] = len(bear_fvgs_active)

        # Distance to nearest FVG
        if bull_fvgs_active:
            dists = [(close[i] - m) / close[i] for m, u, l in bull_fvgs_active]
            nearest_bull_fvg_dist[i] = min(dists, key=abs)
            # Check if price is in any bullish FVG
            for m, u, l_val in bull_fvgs_active:
                if l_val <= close[i] <= u:
                    in_fvg[i] = 1
                    break

        if bear_fvgs_active:
            dists = [(close[i] - m) / close[i] for m, u, l in bear_fvgs_active]
            nearest_bear_fvg_dist[i] = min(dists, key=abs)
            if in_fvg[i] == 0:
                for m, u, l_val in bear_fvgs_active:
                    if l_val <= close[i] <= u:
                        in_fvg[i] = 1
                        break

    df["active_bull_fvg_count"] = active_bull_fvg_count
    df["active_bear_fvg_count"] = active_bear_fvg_count
    df["nearest_bull_fvg_dist"] = nearest_bull_fvg_dist
    df["nearest_bear_fvg_dist"] = nearest_bear_fvg_dist
    df["in_fvg"] = in_fvg

    return df


# ═══════════════════════════════════════════════════════════════
# 6. LIQUIDITY ZONE IDENTIFICATION
# ═══════════════════════════════════════════════════════════════

def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify liquidity pools (clusters of equal highs/lows)
    and detect liquidity sweeps.
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    close = df["close"].values

    buy_liq_dist = np.full(n, np.nan)
    sell_liq_dist = np.full(n, np.nan)
    liq_swept_buy = np.zeros(n, dtype=int)
    liq_swept_sell = np.zeros(n, dtype=int)

    lookback = LIQUIDITY_LOOKBACK
    threshold = LIQUIDITY_CLUSTER_THRESHOLD

    for i in range(lookback, n):
        window_highs = highs[i - lookback: i]
        window_lows = lows[i - lookback: i]

        # Find sell-side liquidity (clusters of similar highs)
        sell_liq_levels = _find_clusters(window_highs, threshold)
        if sell_liq_levels:
            nearest_sell = min(sell_liq_levels, key=lambda x: abs(x - close[i]))
            sell_liq_dist[i] = (nearest_sell - close[i]) / close[i]

            # Sweep detection: price went above the level and came back
            if highs[i] > nearest_sell and close[i] < nearest_sell:
                liq_swept_sell[i] = 1

        # Find buy-side liquidity (clusters of similar lows)
        buy_liq_levels = _find_clusters(window_lows, threshold)
        if buy_liq_levels:
            nearest_buy = min(buy_liq_levels, key=lambda x: abs(x - close[i]))
            buy_liq_dist[i] = (close[i] - nearest_buy) / close[i]

            # Sweep detection: price went below the level and came back
            if lows[i] < nearest_buy and close[i] > nearest_buy:
                liq_swept_buy[i] = 1

    df["buy_liquidity_dist"] = buy_liq_dist
    df["sell_liquidity_dist"] = sell_liq_dist
    df["liquidity_swept_buy"] = liq_swept_buy
    df["liquidity_swept_sell"] = liq_swept_sell

    return df


def _find_clusters(prices: np.ndarray, threshold: float, min_touches: int = 3) -> list:
    """Find price levels that have been touched multiple times (liquidity pools)."""
    if len(prices) == 0:
        return []

    sorted_prices = np.sort(prices)
    clusters = []
    cluster_start = 0

    for i in range(1, len(sorted_prices)):
        if (sorted_prices[i] - sorted_prices[cluster_start]) / sorted_prices[cluster_start] > threshold:
            if i - cluster_start >= min_touches:
                clusters.append(np.mean(sorted_prices[cluster_start:i]))
            cluster_start = i

    # Check last cluster
    if len(sorted_prices) - cluster_start >= min_touches:
        clusters.append(np.mean(sorted_prices[cluster_start:]))

    return clusters


# ═══════════════════════════════════════════════════════════════
# 7. CHANGE IN STATE OF DELIVERY (CISD)
# ═══════════════════════════════════════════════════════════════

def add_cisd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Change in State of Delivery — transition from
    ranging/consolidation to impulsive/trending delivery.
    """
    # Ratio of current candle range to rolling average range
    avg_range = df["candle_range"].rolling(20).mean()
    df["range_expansion"] = df["candle_range"] / avg_range
    df["range_expansion"] = df["range_expansion"].replace([np.inf, -np.inf], np.nan)

    # CISD: significant expansion after contraction
    rolling_min_range = df["candle_range"].rolling(10).min()
    df["cisd_signal"] = (
        (df["range_expansion"] > 2.5) &
        (rolling_min_range.shift(1) < avg_range.shift(1) * 0.5)
    ).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
# 8. ORDER FLOW IMBALANCE (price-based proxy)
# ═══════════════════════════════════════════════════════════════

def add_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy order flow using consecutive candle analysis and pressure ratios."""
    is_bull = df["is_bullish"].values
    n = len(is_bull)

    # Consecutive bullish/bearish candles
    consec_bull = np.zeros(n, dtype=int)
    consec_bear = np.zeros(n, dtype=int)

    for i in range(1, n):
        if is_bull[i] == 1:
            consec_bull[i] = consec_bull[i - 1] + 1
            consec_bear[i] = 0
        else:
            consec_bear[i] = consec_bear[i - 1] + 1
            consec_bull[i] = 0

    df["consecutive_bullish"] = consec_bull
    df["consecutive_bearish"] = consec_bear

    # Buying/selling pressure (rolling ratio of bullish candles)
    for window in [8, 16, 32]:
        df[f"buying_pressure_{window}"] = df["is_bullish"].rolling(window).mean()
        df[f"selling_pressure_{window}"] = 1 - df[f"buying_pressure_{window}"]

    # Body size momentum (running avg of signed body size)
    signed_body = np.where(is_bull, df["body_size"], -df["body_size"])
    df["body_momentum_8"] = pd.Series(signed_body, index=df.index).rolling(8).mean()
    df["body_momentum_16"] = pd.Series(signed_body, index=df.index).rolling(16).mean()

    return df


# ═══════════════════════════════════════════════════════════════
# 9. MULTI-TIMEFRAME FEATURES
# ═══════════════════════════════════════════════════════════════

def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15m candles into higher timeframes (1H, 4H)
    and compute structure features on those timeframes.
    """
    for htf_name, multiplier in HTF_PERIODS.items():
        # Aggregate OHLC
        htf_open = df["open"].shift(multiplier - 1).rolling(multiplier).apply(lambda x: x.iloc[0], raw=False)
        htf_high = df["high"].rolling(multiplier).max()
        htf_low = df["low"].rolling(multiplier).min()
        htf_close = df["close"]  # current close is HTF close

        # HTF candle features
        htf_range = (htf_high - htf_low) / htf_close
        htf_body = (htf_close - htf_open) / htf_close
        df[f"htf_{htf_name}_range"] = htf_range
        df[f"htf_{htf_name}_body"] = htf_body
        df[f"htf_{htf_name}_is_bullish"] = (htf_body > 0).astype(int)

        # HTF returns
        df[f"htf_{htf_name}_return"] = df["close"].pct_change(multiplier)

        # HTF volatility
        df[f"htf_{htf_name}_volatility"] = df["close"].pct_change().rolling(multiplier * 4).std()

    # Multi-TF alignment: all timeframes agree on direction
    df["htf_alignment"] = 0
    bull_align = (
        (df["is_bullish"] == 1) &
        (df["htf_1h_is_bullish"] == 1) &
        (df["htf_4h_is_bullish"] == 1)
    )
    bear_align = (
        (df["is_bullish"] == 0) &
        (df["htf_1h_is_bullish"] == 0) &
        (df["htf_4h_is_bullish"] == 0)
    )
    df.loc[bull_align, "htf_alignment"] = 1
    df.loc[bear_align, "htf_alignment"] = -1

    return df


# ═══════════════════════════════════════════════════════════════
# 10. DISTANCE & CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Session-based, temporal, and distance features."""
    # Time features
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute

    # Session classification (UTC-based)
    hour = df.index.hour
    df["session_asian"] = ((hour >= 0) & (hour < 8)).astype(int)
    df["session_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    df["session_newyork"] = ((hour >= 13) & (hour < 22)).astype(int)

    # Daily high/low distance
    daily_high = df["high"].rolling(96).max()   # 96 candles = 24 hours
    daily_low = df["low"].rolling(96).min()
    df["dist_to_daily_high"] = (df["close"] - daily_high) / df["close"]
    df["dist_to_daily_low"] = (df["close"] - daily_low) / df["close"]

    # Position within daily range
    daily_range = daily_high - daily_low
    daily_range_safe = daily_range.replace(0, np.nan)
    df["position_in_daily_range"] = (df["close"] - daily_low) / daily_range_safe

    return df


# ═══════════════════════════════════════════════════════════════
# 11. SELL STRATEGY FEATURES
# ═══════════════════════════════════════════════════════════════

def add_sell_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features specifically designed for the SELL strategy:
    buyside liquidity sweep → displacement → pullback pattern.
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values

    # --- Distance to previous day high ---
    if hasattr(df.index, 'date'):
        dates = df.index.date
        unique_dates = np.unique(dates)
        daily_highs = {}
        for d in unique_dates:
            mask = dates == d
            daily_highs[d] = np.max(highs[mask])

        sorted_dates = sorted(daily_highs.keys())
        prev_day_h = np.full(n, np.nan)
        for i_d, d in enumerate(sorted_dates):
            if i_d == 0:
                continue
            mask = dates == d
            prev_day_h[mask] = daily_highs[sorted_dates[i_d - 1]]

        df["dist_to_prev_day_high"] = (closes - prev_day_h) / closes
    else:
        df["dist_to_prev_day_high"] = 0.0

    # --- Distance to rolling highs ---
    for period in [48, 96, 192]:
        rolling_h = pd.Series(highs).rolling(period, min_periods=period).max().values
        df[f"dist_to_rolling_high_{period}"] = (closes - rolling_h) / closes

    # --- Recent sweep count ---
    sweep_events = np.zeros(n, dtype=int)
    for period in [48, 96]:
        rolling_h = pd.Series(highs).rolling(period, min_periods=period).max().shift(1).values
        for i in range(period, n):
            if not np.isnan(rolling_h[i]) and highs[i] > rolling_h[i]:
                sweep_events[i] = 1
    df["recent_sweep_count_24"] = pd.Series(sweep_events).rolling(24, min_periods=1).sum().values

    # --- Displacement candle metrics ---
    body = closes - opens
    candle_range = highs - lows
    candle_range_safe = np.where(candle_range > 0, candle_range, np.nan)

    body_ratio = np.abs(body) / candle_range_safe
    is_bull = body > 0

    bull_body_ratio = np.where(is_bull, body_ratio, 0)
    df["max_bull_body_ratio_10"] = pd.Series(bull_body_ratio).rolling(10, min_periods=1).max().values

    upper_wick = highs - np.maximum(opens, closes)
    body_safe = np.where(np.abs(body) > 0, np.abs(body), np.nan)
    wick_ratio = upper_wick / body_safe
    bull_wick_ratio = np.where(is_bull, wick_ratio, np.nan)
    df["min_bull_wick_ratio_10"] = pd.Series(bull_wick_ratio).rolling(10, min_periods=1).min().values

    # --- Pullback depth ---
    recent_high = pd.Series(highs).rolling(10, min_periods=1).max().values
    df["pullback_depth_10"] = (recent_high - closes) / recent_high

    # Consecutive bearish candles
    consec_bear = np.zeros(n, dtype=int)
    for i in range(1, n):
        if closes[i] < opens[i]:
            consec_bear[i] = consec_bear[i - 1] + 1
        else:
            consec_bear[i] = 0
    df["recent_bearish_streak"] = consec_bear

    # --- Distance to nearest liquidity above ---
    nearest_above = np.full(n, np.nan)
    for period in [48, 96]:
        rh = pd.Series(highs).rolling(period, min_periods=period).max().values
        diff = rh - closes
        for i in range(period, n):
            if not np.isnan(rh[i]) and diff[i] >= 0:
                if np.isnan(nearest_above[i]) or diff[i] < nearest_above[i]:
                    nearest_above[i] = diff[i]
    df["dist_to_nearest_liquidity_above"] = nearest_above / closes

    return df


# ═══════════════════════════════════════════════════════════════
# 12. BUY STRATEGY FEATURES
# ═══════════════════════════════════════════════════════════════

def add_buy_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features specifically designed for the BUY strategy:
    sellside liquidity sweep → displacement → pullback pattern.
    Mirror of add_sell_strategy_features().
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values

    # --- Distance to previous day low ---
    if hasattr(df.index, 'date'):
        dates = df.index.date
        unique_dates = np.unique(dates)
        daily_lows = {}
        for d in unique_dates:
            mask = dates == d
            daily_lows[d] = np.min(lows[mask])

        sorted_dates = sorted(daily_lows.keys())
        prev_day_l = np.full(n, np.nan)
        for i_d, d in enumerate(sorted_dates):
            if i_d == 0:
                continue
            mask = dates == d
            prev_day_l[mask] = daily_lows[sorted_dates[i_d - 1]]

        df["dist_to_prev_day_low"] = (closes - prev_day_l) / closes
    else:
        df["dist_to_prev_day_low"] = 0.0

    # --- Distance to rolling lows ---
    for period in [48, 96, 192]:
        rolling_l = pd.Series(lows).rolling(period, min_periods=period).min().values
        df[f"dist_to_rolling_low_{period}"] = (closes - rolling_l) / closes

    # --- Recent sellside sweep count ---
    sweep_events = np.zeros(n, dtype=int)
    for period in [48, 96]:
        rolling_l = pd.Series(lows).rolling(period, min_periods=period).min().shift(1).values
        for i in range(period, n):
            if not np.isnan(rolling_l[i]) and lows[i] < rolling_l[i]:
                sweep_events[i] = 1
    df["recent_sell_sweep_count_24"] = pd.Series(sweep_events).rolling(24, min_periods=1).sum().values

    # --- Bearish displacement candle metrics ---
    body = opens - closes  # positive for bearish
    candle_range = highs - lows
    candle_range_safe = np.where(candle_range > 0, candle_range, np.nan)

    body_ratio = np.abs(body) / candle_range_safe
    is_bear = closes < opens

    bear_body_ratio = np.where(is_bear, body_ratio, 0)
    df["max_bear_body_ratio_10"] = pd.Series(bear_body_ratio).rolling(10, min_periods=1).max().values

    lower_wick = np.minimum(opens, closes) - lows
    body_safe = np.where(np.abs(body) > 0, np.abs(body), np.nan)
    wick_ratio = lower_wick / body_safe
    bear_wick_ratio = np.where(is_bear, wick_ratio, np.nan)
    df["min_bear_wick_ratio_10"] = pd.Series(bear_wick_ratio).rolling(10, min_periods=1).min().values

    # --- Bullish pullback depth ---
    recent_low = pd.Series(lows).rolling(10, min_periods=1).min().values
    df["pullback_depth_from_low_10"] = (closes - recent_low) / closes

    # Consecutive bullish candles
    consec_bull = np.zeros(n, dtype=int)
    for i in range(1, n):
        if closes[i] > opens[i]:
            consec_bull[i] = consec_bull[i - 1] + 1
        else:
            consec_bull[i] = 0
    df["recent_bullish_streak"] = consec_bull

    # --- Distance to nearest liquidity below ---
    nearest_below = np.full(n, np.nan)
    for period in [48, 96]:
        rl = pd.Series(lows).rolling(period, min_periods=period).min().values
        diff = closes - rl
        for i in range(period, n):
            if not np.isnan(rl[i]) and diff[i] >= 0:
                if np.isnan(nearest_below[i]) or diff[i] < nearest_below[i]:
                    nearest_below[i] = diff[i]
    df["dist_to_nearest_liquidity_below"] = nearest_below / closes

    return df


# ═══════════════════════════════════════════════════════════════
# MAIN FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    Returns DataFrame with all engineered features added.
    """
    print("[Features] Adding candlestick structure features...")
    df = add_candle_features(df)

    print("[Features] Adding volatility & displacement features...")
    df = add_volatility_features(df)

    print("[Features] Detecting swing highs and lows...")
    df = detect_swing_points(df)

    print("[Features] Adding market structure (BOS/MSS)...")
    df = add_market_structure(df)

    print("[Features] Detecting Fair Value Gaps...")
    df = add_fvg_features(df)

    print("[Features] Identifying liquidity zones...")
    df = add_liquidity_features(df)

    print("[Features] Adding CISD features...")
    df = add_cisd_features(df)

    print("[Features] Adding order flow proxy features...")
    df = add_order_flow_features(df)

    print("[Features] Adding multi-timeframe features...")
    df = add_multi_timeframe_features(df)

    print("[Features] Adding context & distance features...")
    df = add_context_features(df)

    print("[Features] Adding SELL strategy features...")
    df = add_sell_strategy_features(df)

    print("[Features] Adding BUY strategy features...")
    df = add_buy_strategy_features(df)

    # Drop rows with insufficient data (from rolling calculations)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    print(f"[Features] Dropped {dropped} rows with NaN (warmup period)")
    print(f"[Features] Final feature set: {len(df)} rows × {len(df.columns)} columns")

    return df



def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes OHLC and label columns)."""
    exclude = [
        "open", "high", "low", "close",
        "last_swing_high_price", "last_swing_low_price",
        "label", "meta_label", "primary_proba", "trade_confidence",
    ]
    return [c for c in df.columns if c not in exclude]


"""
tradIA BUY-Only Label Generator (Mirror of SELL strategy)
Detects bullish setups: sellside liquidity sweep → strong red displacement candle
→ weak bullish pullback → LONG ENTRY at displacement candle open.

Strategy sequence (all conditions must be satisfied):
  1. Detect sellside liquidity (previous day low, session low, swing lows, rolling lows)
  2. Liquidity sweep (price breaks below liquidity level)
  3. Strong bearish displacement candle within 10 bars of sweep
     - close < open, body_ratio ≥ 0.6, lower_wick/body ≤ 0.3
     - candle range >= MIN_DISPLACEMENT_ATR × ATR_14
  4. Weak bullish pullback (1–3 candles: close > open, low ≥ displacement low)
  5. Entry = displacement candle open, SL = displacement low, TP = 1.5 × SL distance
  6. Session filter: only during London/NY kill zones
  7. Confluence: multiple liquidity levels must confirm the zone
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from src.config import (
    BUY_SWING_PERIOD,
    BUY_ROLLING_LOW_PERIODS,
    BUY_DISPLACEMENT_WINDOW,
    BUY_BODY_RATIO_MIN,
    BUY_LOWER_WICK_RATIO_MAX,
    BUY_PULLBACK_MIN,
    BUY_PULLBACK_MAX,
    BUY_TP_RATIO,
    BUY_MIN_GAP,
    BUY_SESSION_LOW_HOURS,
    BUY_VALIDATE_OUTCOME,
    TIMEFRAME_MINUTES,
    BUY_SESSION_HOURS,
    BUY_SESSION_FILTER,
    BUY_MIN_LIQUIDITY_LEVELS,
    BUY_MIN_DISPLACEMENT_ATR,
)


# ─────────────────────────────────────────────────────────────
# 1. SELLSIDE LIQUIDITY DETECTION
# ─────────────────────────────────────────────────────────────

def _detect_swing_lows(lows: np.ndarray, period: int) -> np.ndarray:
    """
    Detect swing lows: low[i] is lower than all candles within ±period.
    Returns boolean array where True = swing low at index i.
    """
    n = len(lows)
    is_swing = np.zeros(n, dtype=bool)
    for i in range(period, n - period):
        window = lows[i - period: i + period + 1]
        if lows[i] == np.min(window) and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            is_swing[i] = True
    return is_swing


def _compute_previous_day_low(
    timestamps: pd.DatetimeIndex, lows: np.ndarray
) -> np.ndarray:
    """
    Compute the previous calendar day's low for each candle.
    """
    n = len(lows)
    prev_day_low = np.full(n, np.nan)

    dates = timestamps.date
    unique_dates = np.unique(dates)

    daily_lows = {}
    for d in unique_dates:
        mask = dates == d
        daily_lows[d] = np.min(lows[mask])

    sorted_dates = sorted(daily_lows.keys())
    for i, d in enumerate(sorted_dates):
        if i == 0:
            continue
        prev_d = sorted_dates[i - 1]
        mask = dates == d
        prev_day_low[mask] = daily_lows[prev_d]

    return prev_day_low


def _compute_session_low(
    lows: np.ndarray, session_hours: int, tf_minutes: int
) -> np.ndarray:
    """
    Compute the previous session's low (rolling window of session_hours).
    """
    candles_per_session = (session_hours * 60) // tf_minutes
    n = len(lows)
    session_low = np.full(n, np.nan)
    for i in range(candles_per_session, n):
        start = max(0, i - 2 * candles_per_session)
        end = i - candles_per_session
        if end > start:
            session_low[i] = np.min(lows[start:end])
    return session_low


def _compute_rolling_lows(
    lows: np.ndarray, periods: List[int]
) -> Dict[int, np.ndarray]:
    """Compute rolling N-period lows for multiple periods."""
    result = {}
    for period in periods:
        rolling_low = np.full(len(lows), np.nan)
        for i in range(period, len(lows)):
            rolling_low[i] = np.min(lows[i - period: i])
        result[period] = rolling_low
    return result


def collect_liquidity_levels(
    idx: int,
    lows: np.ndarray,
    swing_lows: np.ndarray,
    prev_day_low: np.ndarray,
    session_low: np.ndarray,
    rolling_lows: Dict[int, np.ndarray],
    swing_period: int,
) -> List[float]:
    """
    Collect all active sellside liquidity levels at candle index `idx`.
    Returns a list of price levels where sell-side liquidity likely sits.
    """
    levels = set()

    # Previous day low
    if not np.isnan(prev_day_low[idx]):
        levels.add(prev_day_low[idx])

    # Session low
    if not np.isnan(session_low[idx]):
        levels.add(session_low[idx])

    # Recent swing lows (look back up to 200 candles)
    lookback = min(idx, 200)
    for j in range(idx - lookback, idx):
        if swing_lows[j]:
            levels.add(lows[j])

    # Rolling period lows
    for period, rl in rolling_lows.items():
        if not np.isnan(rl[idx]):
            levels.add(rl[idx])

    return list(levels)


# ─────────────────────────────────────────────────────────────
# 2. LIQUIDITY SWEEP DETECTION
# ─────────────────────────────────────────────────────────────

def _check_sweep(
    low: float, liquidity_levels: List[float]
) -> Tuple[bool, float, int]:
    """
    Check if the current candle's low sweeps any sellside liquidity level.
    Returns (is_sweep, swept_level, count_swept).
    """
    swept_levels = [lvl for lvl in liquidity_levels if low < lvl]
    if swept_levels:
        return True, min(swept_levels), len(swept_levels)
    return False, 0.0, 0


# ─────────────────────────────────────────────────────────────
# 3. STRONG RED CANDLE (DISPLACEMENT)
# ─────────────────────────────────────────────────────────────

def _find_displacement_candle(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    start: int,
    window: int,
    body_ratio_min: float,
    wick_ratio_max: float,
    min_atr_mult: float,
) -> int:
    """
    Find the first strong bearish displacement candle in [start, start+window).

    Conditions:
      - Bearish: close < open
      - Strong body: abs(open - close) / (high - low) >= body_ratio_min
      - Lower wick restriction: (close - low) / body <= wick_ratio_max
      - Size filter: candle range (high - low) >= min_atr_mult × ATR_14
    Returns index of displacement candle, or -1 if not found.
    """
    end = min(start + window, len(opens))
    for i in range(start, end):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # Must be bearish
        if c >= o:
            continue

        candle_range = h - l
        if candle_range <= 0:
            continue

        body = o - c
        body_ratio = body / candle_range

        if body_ratio < body_ratio_min:
            continue

        # Lower wick restriction
        lower_wick = c - l
        if body > 0 and (lower_wick / body) > wick_ratio_max:
            continue

        # Size filter: displacement must be significant relative to ATR
        if not np.isnan(atr[i]) and atr[i] > 0:
            if candle_range < min_atr_mult * atr[i]:
                continue

        return i

    return -1


# ─────────────────────────────────────────────────────────────
# 4. PULLBACK DETECTION
# ─────────────────────────────────────────────────────────────

def _find_pullback(
    opens: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    disp_idx: int,
    disp_low: float,
    pullback_min: int,
    pullback_max: int,
) -> int:
    """
    After the displacement candle at disp_idx, find a valid bullish pullback.

    Conditions for pullback candles (1 to pullback_max consecutive):
      - Each candle is bullish: close > open
      - Each candle's low ≥ displacement candle's low

    Returns the index of the LAST pullback candle (entry signal), or -1.
    """
    n = len(opens)
    bullish_count = 0

    for i in range(disp_idx + 1, min(disp_idx + 1 + pullback_max, n)):
        # Must be bullish
        if closes[i] <= opens[i]:
            break

        # Low must not go below displacement low
        if lows[i] < disp_low:
            break

        bullish_count += 1

    if pullback_min <= bullish_count <= pullback_max:
        return disp_idx + bullish_count

    return -1


# ─────────────────────────────────────────────────────────────
# 5. SESSION FILTER
# ─────────────────────────────────────────────────────────────

def _in_session(timestamp, session_hours: List[Tuple[int, int]]) -> bool:
    """Check if a timestamp falls within any of the kill zone sessions."""
    hour = timestamp.hour
    for start_h, end_h in session_hours:
        if start_h <= hour < end_h:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 6. OUTCOME VALIDATION
# ─────────────────────────────────────────────────────────────

def _validate_buy_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    entry_idx: int,
    max_hold: int = 96,
) -> bool:
    """
    Verify that a BUY trade entered at entry_price would hit TP before SL.

    SL = displacement candle low (price going DOWN hits SL)
    TP = entry_price + tp_distance (price going UP hits TP)
    """
    n = len(highs)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
        # Check SL first (worst case — price drops to SL)
        if lows[j] <= sl_price:
            return False
        # Check TP (price rises to TP)
        if highs[j] >= tp_price:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# MAIN LABEL GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_buy_labels(
    df: pd.DataFrame,
    swing_period: int = BUY_SWING_PERIOD,
    rolling_periods: List[int] = None,
    displacement_window: int = BUY_DISPLACEMENT_WINDOW,
    body_ratio_min: float = BUY_BODY_RATIO_MIN,
    wick_ratio_max: float = BUY_LOWER_WICK_RATIO_MAX,
    pullback_min: int = BUY_PULLBACK_MIN,
    pullback_max: int = BUY_PULLBACK_MAX,
    tp_ratio: float = BUY_TP_RATIO,
    min_gap: int = BUY_MIN_GAP,
    session_hours: int = BUY_SESSION_LOW_HOURS,
    validate_outcome: bool = BUY_VALIDATE_OUTCOME,
    session_filter: bool = BUY_SESSION_FILTER,
    session_windows: List[Tuple[int, int]] = None,
    min_liquidity_levels: int = BUY_MIN_LIQUIDITY_LEVELS,
    min_displacement_atr: float = BUY_MIN_DISPLACEMENT_ATR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate BUY labels by scanning the entire dataset for the
    sellside liquidity sweep → displacement → pullback pattern.

    Returns:
        df: Original DataFrame with 'label' column added (1=BUY setup, NaN=no setup)
        setups_df: DataFrame with detailed setup information for each detected trade
    """
    if rolling_periods is None:
        rolling_periods = BUY_ROLLING_LOW_PERIODS
    if session_windows is None:
        session_windows = BUY_SESSION_HOURS

    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    # Get ATR for displacement size filter
    atr = df["atr_14"].values if "atr_14" in df.columns else np.full(n, np.nan)

    print("[BUY Labels] Step 1: Detecting sellside liquidity levels...")

    # Pre-compute liquidity sources
    swing_lows = _detect_swing_lows(lows, swing_period)
    prev_day_low = _compute_previous_day_low(timestamps, lows)
    session_low = _compute_session_low(
        lows, session_hours, TIMEFRAME_MINUTES
    )
    rolling_lows_dict = _compute_rolling_lows(lows, rolling_periods)

    sl_count = int(swing_lows.sum())
    print(f"  Swing lows detected: {sl_count:,}")
    print(f"  Rolling low periods: {rolling_periods}")
    print(f"  Session filter: {'ON (' + str(session_windows) + ')' if session_filter else 'OFF'}")
    print(f"  Min liquidity levels: {min_liquidity_levels}")
    print(f"  Min displacement ATR: {min_displacement_atr}")
    print(f"  TP ratio: {tp_ratio}")

    # Storage for labels and setup details
    labels = np.full(n, np.nan)
    setups = []
    last_label_idx = -min_gap - 1

    # Debug counters
    total_sweeps = 0
    session_filtered = 0
    confluence_filtered = 0
    disp_filtered = 0
    pullback_filtered = 0
    outcome_filtered = 0

    print("[BUY Labels] Step 2-5: Scanning for complete setups...")

    for i in range(swing_period + max(rolling_periods), n - displacement_window - pullback_max - 2):
        # Enforce minimum gap between signals
        if i - last_label_idx < min_gap:
            continue

        # Step 1: Collect liquidity levels at this candle
        liq_levels = collect_liquidity_levels(
            i, lows, swing_lows, prev_day_low, session_low,
            rolling_lows_dict, swing_period,
        )

        if not liq_levels:
            continue

        # Step 2: Check for liquidity sweep (low breaks below liquidity)
        is_sweep, swept_level, n_swept = _check_sweep(lows[i], liq_levels)
        if not is_sweep:
            continue

        total_sweeps += 1

        # CONFLUENCE FILTER: require sweeping multiple liquidity levels
        if n_swept < min_liquidity_levels:
            confluence_filtered += 1
            continue

        # SESSION FILTER: only trade during kill zones
        if session_filter and not _in_session(timestamps[i], session_windows):
            session_filtered += 1
            continue

        # Step 3: Find strong red candle within next displacement_window candles
        disp_idx = _find_displacement_candle(
            opens, highs, lows, closes, atr,
            start=i,
            window=displacement_window,
            body_ratio_min=body_ratio_min,
            wick_ratio_max=wick_ratio_max,
            min_atr_mult=min_displacement_atr,
        )
        if disp_idx < 0:
            disp_filtered += 1
            continue

        disp_low = lows[disp_idx]
        disp_open = opens[disp_idx]
        disp_close = closes[disp_idx]

        # Step 4: Find bullish pullback after displacement
        pullback_end_idx = _find_pullback(
            opens, lows, closes,
            disp_idx, disp_low,
            pullback_min, pullback_max,
        )
        if pullback_end_idx < 0:
            pullback_filtered += 1
            continue

        # Step 5: Define entry, SL, TP
        entry_price = disp_open  # LONG entry at displacement candle open
        sl_price = disp_low      # SL at displacement candle low
        sl_distance = entry_price - sl_price

        if sl_distance <= 0:
            continue

        tp_distance = tp_ratio * sl_distance
        tp_price = entry_price + tp_distance  # TP above entry

        # Step 6: Outcome validation (optional)
        if validate_outcome:
            is_winner = _validate_buy_outcome(
                highs, lows, entry_price, sl_price, tp_price,
                pullback_end_idx,
            )
            if not is_winner:
                outcome_filtered += 1
                continue

        # Label at the pullback completion candle
        label_idx = pullback_end_idx
        labels[label_idx] = 1  # 1 = BUY setup present
        last_label_idx = label_idx

        # Record setup details
        setup = {
            "timestamp": timestamps[label_idx],
            "sweep_idx": i,
            "sweep_candle_time": timestamps[i],
            "swept_level": swept_level,
            "n_liquidity_levels_swept": n_swept,
            "disp_idx": disp_idx,
            "disp_candle_time": timestamps[disp_idx],
            "disp_body_ratio": (disp_open - disp_close) / (highs[disp_idx] - disp_low),
            "disp_low": disp_low,
            "disp_open": disp_open,
            "disp_range_atr": (highs[disp_idx] - disp_low) / atr[disp_idx] if not np.isnan(atr[disp_idx]) and atr[disp_idx] > 0 else 0,
            "pullback_candles": pullback_end_idx - disp_idx,
            "pullback_end_time": timestamps[pullback_end_idx],
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_distance_pct": sl_distance / entry_price * 100,
            "tp_distance_pct": tp_distance / entry_price * 100,
            "risk_reward": tp_distance / sl_distance if sl_distance > 0 else 0,
            "session_hour": timestamps[label_idx].hour,
        }
        setups.append(setup)

    df = df.copy()
    df["label"] = labels

    setups_df = pd.DataFrame(setups)
    buy_count = len(setups)

    print(f"\n[BUY Labels] Filter Statistics:")
    print(f"  Total sweeps detected:    {total_sweeps:,}")
    print(f"  Confluence filtered:      {confluence_filtered:,}")
    print(f"  Session filtered:         {session_filtered:,}")
    print(f"  No displacement found:    {disp_filtered:,}")
    print(f"  No valid pullback:        {pullback_filtered:,}")
    print(f"  Outcome filtered (SL):    {outcome_filtered:,}")

    print(f"\n[BUY Labels] Results:")
    print(f"  Total BUY setups detected: {buy_count:,}")
    if buy_count > 0:
        print(f"  Date range: {setups_df['timestamp'].min()} -> {setups_df['timestamp'].max()}")
        print(f"  Avg SL distance: {setups_df['sl_distance_pct'].mean():.3f}%")
        print(f"  Avg TP distance: {setups_df['tp_distance_pct'].mean():.3f}%")
        print(f"  Avg risk/reward: {setups_df['risk_reward'].mean():.2f}")
        print(f"  Avg pullback candles: {setups_df['pullback_candles'].mean():.1f}")
        if 'session_hour' in setups_df.columns:
            print(f"  Session distribution:")
            for (s, e) in BUY_SESSION_HOURS:
                count = ((setups_df['session_hour'] >= s) & (setups_df['session_hour'] < e)).sum()
                print(f"    {s:02d}:00-{e:02d}:00 UTC: {count} setups")

    return df, setups_df

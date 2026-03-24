"""
tradIA SELL-Only Label Generator (Enhanced)
Detects bearish setups: buyside liquidity sweep → strong green displacement candle
→ weak bearish pullback → SHORT ENTRY at displacement candle open.

Enhanced with:
  - Session filter (London/NY kill zones only)
  - Confluence filter (minimum 2 liquidity levels, displacement >= 1 ATR)
  - Improved TP ratio (1.5x SL for favorable R:R)

Strategy sequence (all conditions must be satisfied):
  1. Detect buyside liquidity (previous day high, session high, swing highs, rolling highs)
  2. Liquidity sweep (price breaks above liquidity level)
  3. Strong bullish displacement candle within 10 bars of sweep
     - close > open, body_ratio ≥ 0.6, upper_wick/body ≤ 0.3
     - candle range >= MIN_DISPLACEMENT_ATR × ATR_14
  4. Weak bearish pullback (1–3 candles: close < open, high ≤ displacement high)
  5. Entry = displacement candle open, SL = displacement high, TP = 1.5 × SL distance
  6. [Optional] Session filter: only during London/NY kill zones
  7. [Optional] Confluence: multiple liquidity levels must confirm the zone
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from src.config import (
    SELL_SWING_PERIOD,
    SELL_ROLLING_HIGH_PERIODS,
    SELL_DISPLACEMENT_WINDOW,
    SELL_BODY_RATIO_MIN,
    SELL_UPPER_WICK_RATIO_MAX,
    SELL_PULLBACK_MIN,
    SELL_PULLBACK_MAX,
    SELL_TP_RATIO,
    SELL_MIN_GAP,
    SELL_SESSION_HIGH_HOURS,
    SELL_VALIDATE_OUTCOME,
    TIMEFRAME_MINUTES,
    SELL_SESSION_HOURS,
    SELL_SESSION_FILTER,
    SELL_MIN_LIQUIDITY_LEVELS,
    SELL_MIN_DISPLACEMENT_ATR,
)


# ─────────────────────────────────────────────────────────────
# 1. BUYSIDE LIQUIDITY DETECTION
# ─────────────────────────────────────────────────────────────

def _detect_swing_highs(highs: np.ndarray, period: int) -> np.ndarray:
    """
    Detect swing highs: high[i] is higher than all candles within ±period.
    Returns boolean array where True = swing high at index i.
    """
    n = len(highs)
    is_swing = np.zeros(n, dtype=bool)
    for i in range(period, n - period):
        window = highs[i - period: i + period + 1]
        if highs[i] == np.max(window) and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            is_swing[i] = True
    return is_swing


def _compute_previous_day_high(
    timestamps: pd.DatetimeIndex, highs: np.ndarray
) -> np.ndarray:
    """
    Compute the previous calendar day's high for each candle.
    Returns array of previous day high values.
    """
    n = len(highs)
    prev_day_high = np.full(n, np.nan)

    # Group by date
    dates = timestamps.date
    unique_dates = np.unique(dates)

    daily_highs = {}
    for d in unique_dates:
        mask = dates == d
        daily_highs[d] = np.max(highs[mask])

    sorted_dates = sorted(daily_highs.keys())
    for i, d in enumerate(sorted_dates):
        if i == 0:
            continue
        prev_d = sorted_dates[i - 1]
        mask = dates == d
        prev_day_high[mask] = daily_highs[prev_d]

    return prev_day_high


def _compute_session_high(
    highs: np.ndarray, session_hours: int, tf_minutes: int
) -> np.ndarray:
    """
    Compute the previous session's high (rolling window of session_hours).
    """
    candles_per_session = (session_hours * 60) // tf_minutes
    n = len(highs)
    session_high = np.full(n, np.nan)
    for i in range(candles_per_session, n):
        # Previous session = the session window ending one candle ago
        start = max(0, i - 2 * candles_per_session)
        end = i - candles_per_session
        if end > start:
            session_high[i] = np.max(highs[start:end])
    return session_high


def _compute_rolling_highs(
    highs: np.ndarray, periods: List[int]
) -> Dict[int, np.ndarray]:
    """Compute rolling N-period highs for multiple periods."""
    result = {}
    for period in periods:
        rolling_high = np.full(len(highs), np.nan)
        for i in range(period, len(highs)):
            rolling_high[i] = np.max(highs[i - period: i])
        result[period] = rolling_high
    return result


def collect_liquidity_levels(
    idx: int,
    highs: np.ndarray,
    swing_highs: np.ndarray,
    prev_day_high: np.ndarray,
    session_high: np.ndarray,
    rolling_highs: Dict[int, np.ndarray],
    swing_period: int,
) -> List[float]:
    """
    Collect all active buyside liquidity levels at candle index `idx`.
    Returns a list of price levels where buy-side liquidity likely sits.
    """
    levels = set()

    # Previous day high
    if not np.isnan(prev_day_high[idx]):
        levels.add(prev_day_high[idx])

    # Session high
    if not np.isnan(session_high[idx]):
        levels.add(session_high[idx])

    # Recent swing highs (look back up to 200 candles)
    lookback = min(idx, 200)
    for j in range(idx - lookback, idx):
        if swing_highs[j]:
            levels.add(highs[j])

    # Rolling period highs
    for period, rh in rolling_highs.items():
        if not np.isnan(rh[idx]):
            levels.add(rh[idx])

    return list(levels)


# ─────────────────────────────────────────────────────────────
# 2. LIQUIDITY SWEEP DETECTION
# ─────────────────────────────────────────────────────────────

def _check_sweep(
    high: float, liquidity_levels: List[float]
) -> Tuple[bool, float, int]:
    """
    Check if the current candle's high sweeps any liquidity level.
    Returns (is_sweep, swept_level, count_swept).
    """
    swept_levels = [lvl for lvl in liquidity_levels if high > lvl]
    if swept_levels:
        return True, max(swept_levels), len(swept_levels)
    return False, 0.0, 0


# ─────────────────────────────────────────────────────────────
# 3. STRONG GREEN CANDLE (DISPLACEMENT)
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
    Find the first strong bullish displacement candle in [start, start+window).

    Conditions:
      - Bullish: close > open
      - Strong body: abs(close - open) / (high - low) >= body_ratio_min
      - Upper wick restriction: (high - close) / body <= wick_ratio_max
      - Size filter: candle range (high - low) >= min_atr_mult × ATR_14
    Returns index of displacement candle, or -1 if not found.
    """
    end = min(start + window, len(opens))
    for i in range(start, end):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # Must be bullish
        if c <= o:
            continue

        candle_range = h - l
        if candle_range <= 0:
            continue

        body = c - o
        body_ratio = body / candle_range

        if body_ratio < body_ratio_min:
            continue

        # Upper wick restriction
        upper_wick = h - c
        if body > 0 and (upper_wick / body) > wick_ratio_max:
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
    highs: np.ndarray,
    closes: np.ndarray,
    disp_idx: int,
    disp_high: float,
    pullback_min: int,
    pullback_max: int,
) -> int:
    """
    After the displacement candle at disp_idx, find a valid pullback.

    Conditions for pullback candles (1 to pullback_max consecutive):
      - Each candle is bearish: close < open
      - Each candle's high ≤ displacement candle's high

    Returns the index of the LAST pullback candle (entry signal), or -1.
    """
    n = len(opens)
    bearish_count = 0

    for i in range(disp_idx + 1, min(disp_idx + 1 + pullback_max, n)):
        # Must be bearish
        if closes[i] >= opens[i]:
            break  # Non-bearish candle ends pullback check

        # High must not exceed displacement high
        if highs[i] > disp_high:
            break

        bearish_count += 1

    if pullback_min <= bearish_count <= pullback_max:
        return disp_idx + bearish_count  # Index of last pullback candle

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

def _validate_sell_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    entry_idx: int,
    max_hold: int = 96,  # 24 hours max hold
) -> bool:
    """
    Verify that a SELL trade entered at entry_price would hit TP before SL.

    SL = displacement candle high (price going UP hits SL)
    TP = entry_price - tp_distance (price going DOWN hits TP)
    """
    n = len(highs)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
        # Check SL first (worst case — price goes up to SL)
        if highs[j] >= sl_price:
            return False
        # Check TP (price drops to TP)
        if lows[j] <= tp_price:
            return True
    return False  # Neither hit within max_hold


# ─────────────────────────────────────────────────────────────
# MAIN LABEL GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_sell_labels(
    df: pd.DataFrame,
    swing_period: int = SELL_SWING_PERIOD,
    rolling_periods: List[int] = None,
    displacement_window: int = SELL_DISPLACEMENT_WINDOW,
    body_ratio_min: float = SELL_BODY_RATIO_MIN,
    wick_ratio_max: float = SELL_UPPER_WICK_RATIO_MAX,
    pullback_min: int = SELL_PULLBACK_MIN,
    pullback_max: int = SELL_PULLBACK_MAX,
    tp_ratio: float = SELL_TP_RATIO,
    min_gap: int = SELL_MIN_GAP,
    session_hours: int = SELL_SESSION_HIGH_HOURS,
    validate_outcome: bool = SELL_VALIDATE_OUTCOME,
    session_filter: bool = SELL_SESSION_FILTER,
    session_windows: List[Tuple[int, int]] = None,
    min_liquidity_levels: int = SELL_MIN_LIQUIDITY_LEVELS,
    min_displacement_atr: float = SELL_MIN_DISPLACEMENT_ATR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate SELL labels by scanning the entire dataset for the
    buyside liquidity sweep → displacement → pullback pattern.

    Returns:
        df: Original DataFrame with 'label' column added (1=SELL setup, NaN=no setup)
        setups_df: DataFrame with detailed setup information for each detected trade
    """
    if rolling_periods is None:
        rolling_periods = SELL_ROLLING_HIGH_PERIODS
    if session_windows is None:
        session_windows = SELL_SESSION_HOURS

    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    # Get ATR for displacement size filter
    atr = df["atr_14"].values if "atr_14" in df.columns else np.full(n, np.nan)

    print("[SELL Labels] Step 1: Detecting buyside liquidity levels...")

    # Pre-compute liquidity sources
    swing_highs = _detect_swing_highs(highs, swing_period)
    prev_day_high = _compute_previous_day_high(timestamps, highs)
    session_high = _compute_session_high(
        highs, session_hours, TIMEFRAME_MINUTES
    )
    rolling_highs_dict = _compute_rolling_highs(highs, rolling_periods)

    sh_count = int(swing_highs.sum())
    print(f"  Swing highs detected: {sh_count:,}")
    print(f"  Rolling high periods: {rolling_periods}")
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

    print("[SELL Labels] Step 2-5: Scanning for complete setups...")

    for i in range(swing_period + max(rolling_periods), n - displacement_window - pullback_max - 2):
        # Enforce minimum gap between signals
        if i - last_label_idx < min_gap:
            continue

        # Step 1: Collect liquidity levels at this candle
        liq_levels = collect_liquidity_levels(
            i, highs, swing_highs, prev_day_high, session_high,
            rolling_highs_dict, swing_period,
        )

        if not liq_levels:
            continue

        # Step 2: Check for liquidity sweep
        is_sweep, swept_level, n_swept = _check_sweep(highs[i], liq_levels)
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

        # Step 3: Find strong green candle within next displacement_window candles
        disp_idx = _find_displacement_candle(
            opens, highs, lows, closes, atr,
            start=i,  # can be the sweep candle itself or after
            window=displacement_window,
            body_ratio_min=body_ratio_min,
            wick_ratio_max=wick_ratio_max,
            min_atr_mult=min_displacement_atr,
        )
        if disp_idx < 0:
            disp_filtered += 1
            continue

        disp_high = highs[disp_idx]
        disp_open = opens[disp_idx]
        disp_close = closes[disp_idx]

        # Step 4: Find pullback after displacement
        pullback_end_idx = _find_pullback(
            opens, highs, closes,
            disp_idx, disp_high,
            pullback_min, pullback_max,
        )
        if pullback_end_idx < 0:
            pullback_filtered += 1
            continue

        # Step 5: Define entry, SL, TP
        entry_price = disp_open
        sl_price = disp_high
        sl_distance = sl_price - entry_price

        if sl_distance <= 0:
            continue

        tp_distance = tp_ratio * sl_distance
        tp_price = entry_price - tp_distance

        # Step 6: Outcome validation (optional)
        if validate_outcome:
            is_winner = _validate_sell_outcome(
                highs, lows, entry_price, sl_price, tp_price,
                pullback_end_idx,
            )
            if not is_winner:
                outcome_filtered += 1
                continue

        # Label at the pullback completion candle
        label_idx = pullback_end_idx
        labels[label_idx] = 1  # 1 = SELL setup present
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
            "disp_body_ratio": (disp_close - disp_open) / (disp_high - lows[disp_idx]),
            "disp_high": disp_high,
            "disp_open": disp_open,
            "disp_range_atr": (disp_high - lows[disp_idx]) / atr[disp_idx] if not np.isnan(atr[disp_idx]) and atr[disp_idx] > 0 else 0,
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
    sell_count = len(setups)

    print(f"\n[SELL Labels] Filter Statistics:")
    print(f"  Total sweeps detected:    {total_sweeps:,}")
    print(f"  Confluence filtered:      {confluence_filtered:,}")
    print(f"  Session filtered:         {session_filtered:,}")
    print(f"  No displacement found:    {disp_filtered:,}")
    print(f"  No valid pullback:        {pullback_filtered:,}")
    print(f"  Outcome filtered (SL):    {outcome_filtered:,}")

    print(f"\n[SELL Labels] Results:")
    print(f"  Total SELL setups detected: {sell_count:,}")
    if sell_count > 0:
        print(f"  Date range: {setups_df['timestamp'].min()} -> {setups_df['timestamp'].max()}")
        print(f"  Avg SL distance: {setups_df['sl_distance_pct'].mean():.3f}%")
        print(f"  Avg TP distance: {setups_df['tp_distance_pct'].mean():.3f}%")
        print(f"  Avg risk/reward: {setups_df['risk_reward'].mean():.2f}")
        print(f"  Avg pullback candles: {setups_df['pullback_candles'].mean():.1f}")
        if 'session_hour' in setups_df.columns:
            print(f"  Session distribution:")
            for (s, e) in SELL_SESSION_HOURS:
                count = ((setups_df['session_hour'] >= s) & (setups_df['session_hour'] < e)).sum()
                print(f"    {s:02d}:00–{e:02d}:00 UTC: {count} setups")

    return df, setups_df

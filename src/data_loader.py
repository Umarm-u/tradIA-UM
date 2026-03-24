"""
tradIA Data Loader & Validator
Loads raw OHLC CSV data, validates integrity, and preprocesses for feature engineering.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

from src.config import (
    DATA_PATH, DATETIME_COL, OHLC_COLS, TIMEFRAME_MINUTES
)


def load_raw_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV data and parse datetime index."""
    df = pd.read_csv(filepath, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    df.set_index(DATETIME_COL, inplace=True)
    # Ensure numeric types
    for col in OHLC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Run comprehensive data quality checks.
    Returns a report dict with findings.
    """
    report = {}

    # 1. Shape
    report["total_rows"] = len(df)
    report["columns"] = list(df.columns)
    report["date_range"] = (str(df.index.min()), str(df.index.max()))

    # 2. Missing values
    missing = df[OHLC_COLS].isnull().sum().to_dict()
    report["missing_values"] = missing
    report["total_missing"] = sum(missing.values())

    # 3. Duplicate timestamps
    dup_count = df.index.duplicated().sum()
    report["duplicate_timestamps"] = int(dup_count)

    # 4. Gaps in time series (missing candles)
    expected_freq = pd.Timedelta(minutes=TIMEFRAME_MINUTES)
    time_diffs = df.index.to_series().diff().dropna()
    gaps = time_diffs[time_diffs > expected_freq]
    report["time_gaps_count"] = len(gaps)
    if len(gaps) > 0:
        report["largest_gap"] = str(gaps.max())
        report["gap_details"] = [
            {"at": str(idx), "gap": str(gap)}
            for idx, gap in gaps.head(10).items()
        ]

    # 5. OHLC integrity: low <= open, close <= high
    ohlc_violations = (
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"])
    )
    report["ohlc_integrity_violations"] = int(ohlc_violations.sum())

    # 6. Abnormal price jumps (>10% in a single candle)
    pct_change = df["close"].pct_change().abs()
    abnormal_jumps = pct_change[pct_change > 0.10]
    report["abnormal_jumps_count"] = len(abnormal_jumps)
    if len(abnormal_jumps) > 0:
        report["abnormal_jumps_top5"] = [
            {"at": str(idx), "pct_change": f"{val:.2%}"}
            for idx, val in abnormal_jumps.nlargest(5).items()
        ]

    # 7. Zero-range candles (high == low)
    zero_range = (df["high"] == df["low"]).sum()
    report["zero_range_candles"] = int(zero_range)

    return report


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the data after validation.
    - Drop duplicate timestamps
    - Forward-fill small gaps (max 2 consecutive)
    - Drop rows with remaining NaN
    - Sort chronologically
    """
    df = df.copy()

    # Drop duplicate timestamps (keep first)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    # Sort by time
    df = df.sort_index()

    # Resample to ensure regular frequency, forward-fill up to 2 periods
    df = df.resample(f"{TIMEFRAME_MINUTES}min").first()
    df = df.ffill(limit=2)

    # Drop remaining NaN rows (big gaps we can't fill)
    rows_before = len(df)
    df = df.dropna(subset=OHLC_COLS)
    rows_dropped = rows_before - len(df)

    if rows_dropped > 0:
        print(f"[preprocess] Dropped {rows_dropped} rows with unfillable NaN values")

    return df


def print_report(report: Dict) -> None:
    """Pretty-print the data validation report."""
    print("=" * 60)
    print("  tradIA Data Validation Report")
    print("=" * 60)
    print(f"  Total rows       : {report['total_rows']:,}")
    print(f"  Date range       : {report['date_range'][0]} → {report['date_range'][1]}")
    print(f"  Columns          : {report['columns']}")
    print("-" * 60)
    print(f"  Missing values   : {report['total_missing']}")
    for col, cnt in report["missing_values"].items():
        if cnt > 0:
            print(f"    {col}: {cnt}")
    print(f"  Duplicate times  : {report['duplicate_timestamps']}")
    print(f"  Time gaps        : {report['time_gaps_count']}")
    if report["time_gaps_count"] > 0:
        print(f"    Largest gap    : {report['largest_gap']}")
    print(f"  OHLC violations  : {report['ohlc_integrity_violations']}")
    print(f"  Abnormal jumps   : {report['abnormal_jumps_count']}")
    print(f"  Zero-range bars  : {report['zero_range_candles']}")
    print("=" * 60)


def load_and_prepare_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Full pipeline: load → validate → preprocess.
    Returns cleaned DataFrame ready for feature engineering.
    """
    print("[1/3] Loading raw data...")
    df = load_raw_data(filepath)

    print("[2/3] Validating data integrity...")
    report = validate_data(df)
    print_report(report)

    print("[3/3] Preprocessing data...")
    df = preprocess_data(df)
    print(f"  Final dataset: {len(df):,} rows")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")

    return df

"""
tradIA Live Trading — Real-Time Data Pipeline
Fetches candles from Binance and runs the EXACT same feature engineering
pipeline used during training. Maintains a rolling buffer of candles.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

from live.binance_client import BinanceClient
from live.config_live import CANDLE_BUFFER_SIZE
from live.logger import log

# Import the EXACT same feature engineering used during training
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import engineer_features, get_feature_columns


class LiveDataPipeline:
    """
    Maintains a rolling buffer of OHLC candles and produces
    feature-engineered data identical to the training pipeline.
    """

    def __init__(self, client: BinanceClient, buffer_size: int = CANDLE_BUFFER_SIZE):
        self.client = client
        self.buffer_size = buffer_size
        self.raw_buffer: Optional[pd.DataFrame] = None   # raw OHLC candles
        self.feat_buffer: Optional[pd.DataFrame] = None   # feature-engineered
        self.feature_cols: List[str] = []
        self._last_candle_time: Optional[pd.Timestamp] = None

    def initialize(self) -> pd.DataFrame:
        """
        Fetch historical candles to warm up all rolling features.
        Must be called once on startup before the main loop.
        """
        log.info(f"Initializing data pipeline — fetching {self.buffer_size} candles...")

        self.raw_buffer = self.client.fetch_klines(limit=self.buffer_size)
        log.info(
            f"Fetched {len(self.raw_buffer)} candles: "
            f"{self.raw_buffer.index[0]} → {self.raw_buffer.index[-1]}"
        )

        # Run feature engineering on the full buffer
        self.feat_buffer = engineer_features(self.raw_buffer.copy())
        self.feature_cols = get_feature_columns(self.feat_buffer)

        self._last_candle_time = self.raw_buffer.index[-1]

        log.info(
            f"Feature engineering complete: {len(self.feat_buffer)} rows × "
            f"{len(self.feature_cols)} features"
        )

        return self.feat_buffer

    def update(self) -> Tuple[Optional[pd.Series], bool]:
        """
        Fetch the latest candle(s) and update features.

        Returns:
            (latest_features, is_new_candle)
            - latest_features: Series with all features for the newest closed candle
            - is_new_candle: True if a new candle was added (vs. duplicate)
        """
        # Fetch recent candles (small batch to catch any gaps)
        recent = self.client.fetch_klines(limit=5)

        if recent.empty:
            log.warning("No candles fetched in update()")
            return None, False

        latest_time = recent.index[-1]

        # Check if we have a new candle
        if self._last_candle_time is not None and latest_time <= self._last_candle_time:
            log.debug(f"No new candle yet. Latest: {latest_time}")
            return self._get_latest_features(), False

        # Append new candles to buffer
        new_candles = recent[recent.index > self._last_candle_time] if self._last_candle_time else recent

        if new_candles.empty:
            return self._get_latest_features(), False

        log.info(f"New candle(s): {len(new_candles)} — latest: {latest_time}")

        # Append to raw buffer
        self.raw_buffer = pd.concat([self.raw_buffer, new_candles])

        # Remove duplicates (keep last)
        self.raw_buffer = self.raw_buffer[~self.raw_buffer.index.duplicated(keep="last")]

        # Trim to buffer size
        if len(self.raw_buffer) > self.buffer_size:
            self.raw_buffer = self.raw_buffer.iloc[-self.buffer_size:]

        # Re-run feature engineering on full buffer
        # This ensures all rolling features are correct
        self.feat_buffer = engineer_features(self.raw_buffer.copy())
        self.feature_cols = get_feature_columns(self.feat_buffer)

        self._last_candle_time = latest_time

        latest_features = self._get_latest_features()

        if latest_features is not None:
            log.debug(
                f"Updated features — latest candle: {self.feat_buffer.index[-1]}, "
                f"close={latest_features.get('close', 'N/A')}, "
                f"atr_14={latest_features.get('atr_14', 'N/A'):.4f}"
            )

        return latest_features, True

    def _get_latest_features(self) -> Optional[pd.Series]:
        """Return the most recent feature row."""
        if self.feat_buffer is None or self.feat_buffer.empty:
            return None
        return self.feat_buffer.iloc[-1]

    def get_latest_atr(self) -> float:
        """Get the latest ATR(14) value for SL/TP calculation."""
        features = self._get_latest_features()
        if features is not None and "atr_14" in features.index:
            return float(features["atr_14"])
        return 0.0

    def get_latest_close(self) -> float:
        """Get the latest close price."""
        features = self._get_latest_features()
        if features is not None and "close" in features.index:
            return float(features["close"])
        return 0.0

    def get_feature_vector(self, feature_names: List[str]) -> Optional[np.ndarray]:
        """
        Extract a feature vector from the latest row, matching the given
        feature names (from model metadata).

        Args:
            feature_names: list of feature column names the model expects

        Returns:
            1D numpy array of feature values, or None if data unavailable
        """
        features = self._get_latest_features()
        if features is None:
            return None

        # Build vector matching model's expected features
        vector = []
        for name in feature_names:
            if name in features.index:
                val = features[name]
                vector.append(0.0 if pd.isna(val) else float(val))
            else:
                log.warning(f"Feature '{name}' not found in live data — using 0.0")
                vector.append(0.0)

        return np.array(vector).reshape(1, -1)

    @property
    def last_candle_time(self) -> Optional[pd.Timestamp]:
        return self._last_candle_time

    @property
    def buffer_length(self) -> int:
        return len(self.raw_buffer) if self.raw_buffer is not None else 0

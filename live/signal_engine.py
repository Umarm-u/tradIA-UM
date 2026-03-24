"""
tradIA Live Trading — Signal Engine
Loads the trained BUY and SELL models (primary + meta)
and generates trading signals from live feature data.
"""
import os
import json
import numpy as np
import lightgbm as lgb
import joblib
from typing import Optional, Dict, Tuple
from enum import Enum

from live.config_live import (
    MODELS_DIR, BUY_THRESHOLD, SELL_THRESHOLD,
    META_FILTER_ENABLED, META_THRESHOLD,
    SESSION_FILTER_ENABLED, SESSION_HOURS,
)
from live.logger import log


class Signal(Enum):
    """Trading signal types."""
    NO_SIGNAL = "NO_SIGNAL"
    LONG = "LONG"       # BUY signal → open long
    SHORT = "SHORT"     # SELL signal → open short


class SignalEngine:
    """
    Loads trained BUY and SELL models and generates trading signals.

    Uses:
      - buy_lightgbm_model.txt   (primary BUY model)
      - sell_lightgbm_model.txt  (primary SELL model)
      - buy_feature_names.json   (ordered feature names for BUY model)
      - sell_feature_names.json  (ordered feature names for SELL model)
      - buy_meta_model.txt       (meta-labeler for BUY)
      - sell_meta_model.txt      (meta-labeler for SELL)
    """

    def __init__(self):
        self.buy_model: Optional[lgb.Booster] = None
        self.sell_model: Optional[lgb.Booster] = None
        self.buy_meta_model: Optional[lgb.Booster] = None
        self.sell_meta_model: Optional[lgb.Booster] = None

        self.buy_feature_names: list = []
        self.sell_feature_names: list = []
        self.buy_meta_feature_names: list = []
        self.sell_meta_feature_names: list = []

    def load_models(self):
        """Load all trained models from disk."""
        models_dir = str(MODELS_DIR)

        # ── BUY primary model ──
        buy_path = os.path.join(models_dir, "buy_lightgbm_model.txt")
        buy_features_path = os.path.join(models_dir, "buy_feature_names.json")
        if os.path.exists(buy_path):
            self.buy_model = lgb.Booster(model_file=buy_path)
            # Load feature names from JSON (created from SHAP importance CSV)
            if os.path.exists(buy_features_path):
                with open(buy_features_path, "r") as f:
                    self.buy_feature_names = json.load(f)
            else:
                log.warning(f"BUY feature names not found at {buy_features_path}")
                self.buy_feature_names = []
            log.info(f"BUY model loaded: {len(self.buy_feature_names)} features")
        else:
            log.warning(f"BUY model not found at {buy_path}")

        # ── SELL primary model ──
        sell_path = os.path.join(models_dir, "sell_lightgbm_model.txt")
        sell_features_path = os.path.join(models_dir, "sell_feature_names.json")
        if os.path.exists(sell_path):
            self.sell_model = lgb.Booster(model_file=sell_path)
            # Load feature names from JSON
            if os.path.exists(sell_features_path):
                with open(sell_features_path, "r") as f:
                    self.sell_feature_names = json.load(f)
            else:
                log.warning(f"SELL feature names not found at {sell_features_path}")
                self.sell_feature_names = []
            log.info(f"SELL model loaded: {len(self.sell_feature_names)} features")
        else:
            log.warning(f"SELL model not found at {sell_path}")

        # ── BUY meta model ──
        buy_meta_model_path = os.path.join(models_dir, "buy_meta_model.txt")
        buy_meta_model_meta_path = buy_meta_model_path.replace(".txt", "_meta.pkl")
        if META_FILTER_ENABLED and os.path.exists(buy_meta_model_path):
            self.buy_meta_model = lgb.Booster(model_file=buy_meta_model_path)
            meta = joblib.load(buy_meta_model_meta_path)
            self.buy_meta_feature_names = meta.get("feature_names") or []
            log.info(f"BUY meta model loaded: {len(self.buy_meta_feature_names)} features")
        else:
            log.info("BUY meta model not loaded (disabled or not found)")

        # ── SELL meta model ──
        sell_meta_model_path = os.path.join(models_dir, "sell_meta_model.txt")
        sell_meta_model_meta_path = sell_meta_model_path.replace(".txt", "_meta.pkl")
        if META_FILTER_ENABLED and os.path.exists(sell_meta_model_path):
            self.sell_meta_model = lgb.Booster(model_file=sell_meta_model_path)
            meta = joblib.load(sell_meta_model_meta_path)
            self.sell_meta_feature_names = meta.get("feature_names") or []
            log.info(f"SELL meta model loaded: {len(self.sell_meta_feature_names)} features")
        else:
            log.info("SELL meta model not loaded (disabled or not found)")

    def generate_signal(self, features_row, candle_time=None) -> Tuple[Signal, Dict]:
        """
        Generate a trading signal from the latest feature data.

        Args:
            features_row: pandas Series with all engineered features
            candle_time: timestamp of the candle (for session filtering)

        Returns:
            (signal, details) where details contains probabilities and reasoning
        """
        details = {
            "buy_proba": 0.0,
            "sell_proba": 0.0,
            "buy_meta_proba": None,
            "sell_meta_proba": None,
            "buy_signal": False,
            "sell_signal": False,
            "reason": "",
        }

        # ── Session filter ──
        if SESSION_FILTER_ENABLED and candle_time is not None:
            hour = candle_time.hour
            in_session = any(start <= hour < end for start, end in SESSION_HOURS)
            if not in_session:
                details["reason"] = f"Outside kill zone (hour={hour})"
                return Signal.NO_SIGNAL, details

        # ── BUY model prediction ──
        buy_signal = False
        if self.buy_model is not None:
            buy_vector = self._extract_features(features_row, self.buy_feature_names)
            if buy_vector is not None:
                buy_proba = float(self.buy_model.predict(buy_vector)[0])
                details["buy_proba"] = buy_proba

                if buy_proba >= BUY_THRESHOLD:
                    buy_signal = True

                    # Apply meta filter
                    if self.buy_meta_model is not None:
                        buy_meta_vector = self._extract_meta_features(
                            features_row, buy_proba, self.buy_meta_feature_names
                        )
                        if buy_meta_vector is not None:
                            buy_meta_proba = float(self.buy_meta_model.predict(buy_meta_vector)[0])
                            details["buy_meta_proba"] = buy_meta_proba
                            if buy_meta_proba < META_THRESHOLD:
                                buy_signal = False
                                log.debug(
                                    f"BUY meta filter rejected: meta_proba={buy_meta_proba:.3f} "
                                    f"< {META_THRESHOLD}"
                                )

        # ── SELL model prediction ──
        sell_signal = False
        if self.sell_model is not None:
            sell_vector = self._extract_features(features_row, self.sell_feature_names)
            if sell_vector is not None:
                sell_proba = float(self.sell_model.predict(sell_vector)[0])
                details["sell_proba"] = sell_proba

                if sell_proba >= SELL_THRESHOLD:
                    sell_signal = True

                    # Apply meta filter
                    if self.sell_meta_model is not None:
                        sell_meta_vector = self._extract_meta_features(
                            features_row, sell_proba, self.sell_meta_feature_names
                        )
                        if sell_meta_vector is not None:
                            sell_meta_proba = float(self.sell_meta_model.predict(sell_meta_vector)[0])
                            details["sell_meta_proba"] = sell_meta_proba
                            if sell_meta_proba < META_THRESHOLD:
                                sell_signal = False
                                log.debug(
                                    f"SELL meta filter rejected: meta_proba={sell_meta_proba:.3f} "
                                    f"< {META_THRESHOLD}"
                                )

        details["buy_signal"] = buy_signal
        details["sell_signal"] = sell_signal

        # ── Signal logic ──
        if buy_signal and not sell_signal:
            details["reason"] = (
                f"BUY signal (proba={details['buy_proba']:.3f} >= {BUY_THRESHOLD})"
            )
            return Signal.LONG, details

        elif sell_signal and not buy_signal:
            details["reason"] = (
                f"SELL signal (proba={details['sell_proba']:.3f} >= {SELL_THRESHOLD})"
            )
            return Signal.SHORT, details

        elif buy_signal and sell_signal:
            details["reason"] = "Both BUY and SELL signals — conflict, skipping"
            return Signal.NO_SIGNAL, details

        else:
            details["reason"] = (
                f"No signal (buy={details['buy_proba']:.3f}, sell={details['sell_proba']:.3f})"
            )
            return Signal.NO_SIGNAL, details

    def _extract_features(self, row, feature_names: list) -> Optional[np.ndarray]:
        """Extract feature vector matching model's expected features."""
        try:
            vector = []
            for name in feature_names:
                if name in row.index:
                    val = row[name]
                    vector.append(0.0 if np.isnan(val) else float(val))
                else:
                    vector.append(0.0)
            return np.array(vector).reshape(1, -1)
        except Exception as e:
            log.error(f"Feature extraction error: {e}")
            return None

    def _extract_meta_features(
        self, row, primary_proba: float, meta_feature_names: list
    ) -> Optional[np.ndarray]:
        """
        Extract meta-model features. The meta model uses a mix of
        primary probability and market features.
        """
        try:
            vector = []
            for name in meta_feature_names:
                if name == "primary_proba":
                    vector.append(primary_proba)
                elif name in row.index:
                    val = row[name]
                    vector.append(0.0 if np.isnan(val) else float(val))
                else:
                    vector.append(0.0)
            return np.array(vector).reshape(1, -1)
        except Exception as e:
            log.error(f"Meta feature extraction error: {e}")
            return None

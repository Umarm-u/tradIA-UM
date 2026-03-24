"""
tradIA Meta-Labeling System
Two-stage prediction: primary model predicts direction,
meta model predicts whether the trade will be profitable.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import joblib
from typing import Tuple, Dict, List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report

from src.config import (
    META_THRESHOLD, META_PROFIT_ATR_MULT, META_PROFIT_HORIZON,
    META_FEATURES, WALK_FORWARD_FOLDS, RANDOM_SEED, MODELS_DIR,
)


class MetaLabeler:
    """
    Meta-labeling pipeline.

    1. Takes primary model predictions (direction probabilities).
    2. Creates binary meta-labels (1=profitable, 0=not profitable).
    3. Trains a second LightGBM to predict profitability.
    4. Outputs a trade confidence score = primary_proba * meta_proba.
    """

    def __init__(self, threshold: float = META_THRESHOLD):
        self.threshold = threshold
        self.model = None
        self.meta_feature_names = None

    # ---------------------------------------------------------
    # Meta-label creation
    # ---------------------------------------------------------

    @staticmethod
    def create_meta_labels(
        df: pd.DataFrame,
        primary_labels: np.ndarray,
        profit_atr_mult: float = META_PROFIT_ATR_MULT,
        horizon: int = META_PROFIT_HORIZON,
    ) -> np.ndarray:
        """
        Create binary meta-labels: 1 if the primary signal direction is
        correct AND price moves >= profit_atr_mult * ATR in that direction
        within *horizon* candles.
        """
        labeled = df.dropna(subset=["label"])
        n = len(labeled)
        min_len = min(n, len(primary_labels))

        close = labeled["close"].values[:min_len]
        high = labeled["high"].values[:min_len]
        low = labeled["low"].values[:min_len]
        atr = labeled["atr_14"].values[:min_len]

        meta_labels = np.zeros(min_len, dtype=int)

        for i in range(min_len - horizon):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue

            direction = int(primary_labels[i])  # 1=BUY, 0=SELL
            target_move = profit_atr_mult * atr[i]

            if direction == 1:  # BUY - check if price went up enough
                max_price = np.max(high[i + 1: i + horizon + 1])
                if max_price - close[i] >= target_move:
                    meta_labels[i] = 1
            else:  # SELL - check if price went down enough
                min_price = np.min(low[i + 1: i + horizon + 1])
                if close[i] - min_price >= target_move:
                    meta_labels[i] = 1

        return meta_labels

    # ---------------------------------------------------------
    # Feature extraction for meta model
    # ---------------------------------------------------------

    @staticmethod
    def extract_meta_features(
        df: pd.DataFrame,
        primary_proba: np.ndarray,
        feature_list: List[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build the meta-model feature matrix from the DataFrame and
        primary model probabilities.
        """
        feature_list = feature_list or META_FEATURES

        labeled = df.dropna(subset=["label"])
        min_len = min(len(labeled), len(primary_proba))
        labeled = labeled.iloc[:min_len].copy()

        # Inject primary probability as a column
        labeled = labeled.copy()
        labeled["primary_proba"] = primary_proba[:min_len]

        # Select available features
        available = [f for f in feature_list if f in labeled.columns]
        X_meta = labeled[available].values

        # Fill remaining NaNs with 0
        X_meta = np.nan_to_num(X_meta, nan=0.0)

        return X_meta, available

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(
        self,
        X_meta: np.ndarray,
        meta_labels: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """Train the meta-model on extracted features and meta-labels."""
        self.meta_feature_names = feature_names

        n_pos = meta_labels.sum()
        n_neg = len(meta_labels) - n_pos
        scale = n_neg / max(n_pos, 1)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 500,
            "scale_pos_weight": scale,
        }

        # Simple 80/20 time split for meta model
        split = int(len(X_meta) * 0.8)
        X_tr, X_val = X_meta[:split], X_meta[split:]
        y_tr, y_val = meta_labels[:split], meta_labels[split:]

        train_ds = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        self.model = lgb.train(
            params, train_ds,
            valid_sets=[val_ds],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Evaluate
        val_proba = self.model.predict(X_val)
        try:
            auc = roc_auc_score(y_val, val_proba)
        except ValueError:
            auc = 0.5

        print(f"[Meta] Meta-model validation AUC: {auc:.4f}")
        print(f"[Meta] Meta-label distribution: "
              f"profitable={int(meta_labels.sum()):,}, "
              f"unprofitable={int(len(meta_labels) - meta_labels.sum()):,}")

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict meta-probabilities (probability of profitability)."""
        return self.model.predict(X_meta)

    def get_trade_confidence(
        self,
        primary_proba: np.ndarray,
        meta_proba: np.ndarray,
    ) -> np.ndarray:
        """
        Compute composite trade confidence.
        confidence = primary_confidence * meta_probability
        where primary_confidence = |primary_proba - 0.5| * 2  (normalized to 0..1)
        """
        primary_confidence = np.abs(primary_proba - 0.5) * 2
        return primary_confidence * meta_proba

    # ---------------------------------------------------------
    # Walk-forward meta evaluation
    # ---------------------------------------------------------

    def walk_forward_evaluate(
        self,
        df: pd.DataFrame,
        X_features: np.ndarray,
        y_labels: np.ndarray,
        feature_cols: List[str],
        primary_pipeline,
        n_splits: int = 3,
    ) -> Dict:
        """
        Full walk-forward meta-labeling evaluation.
        For each fold:
          1. Train primary model on training set
          2. Get out-of-fold primary predictions
          3. Create meta-labels from the training portion
          4. Train meta model
          5. Predict on test portion
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_confidence = []
        all_meta_proba = []
        all_primary_proba = []
        all_y_true = []

        labeled_df = df.dropna(subset=["label"]).copy()
        min_len = min(len(labeled_df), len(X_features))
        labeled_df = labeled_df.iloc[:min_len]

        print(f"\n[Meta] Walk-Forward Meta-Labeling ({n_splits} folds)")
        print("=" * 50)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_features)):
            X_tr, X_te = X_features[train_idx], X_features[test_idx]
            y_tr, y_te = y_labels[train_idx], y_labels[test_idx]

            # 1. Train primary on this fold
            val_sz = int(len(X_tr) * 0.2)
            primary_pipeline.train(
                X_tr[:-val_sz], y_tr[:-val_sz],
                X_tr[-val_sz:], y_tr[-val_sz:]
            )

            # 2. Get primary predictions for train set (in-sample for meta)
            _, train_proba = primary_pipeline.predict(X_tr)

            # 3. Create meta-labels on training set
            train_df = labeled_df.iloc[train_idx]
            train_primary_labels = (train_proba >= 0.5).astype(int)
            meta_y_train = self.create_meta_labels(
                train_df, train_primary_labels
            )

            # 4. Extract meta features and train meta model
            X_meta_tr, meta_feat_names = self.extract_meta_features(
                train_df, train_proba
            )
            min_meta = min(len(X_meta_tr), len(meta_y_train))
            self.train(
                X_meta_tr[:min_meta],
                meta_y_train[:min_meta],
                meta_feat_names
            )

            # 5. Predict on test set
            _, test_proba = primary_pipeline.predict(X_te)
            test_df = labeled_df.iloc[test_idx]
            X_meta_te, _ = self.extract_meta_features(
                test_df, test_proba, meta_feat_names
            )
            meta_proba = self.predict(X_meta_te)

            confidence = self.get_trade_confidence(test_proba, meta_proba)

            all_confidence.extend(confidence)
            all_meta_proba.extend(meta_proba)
            all_primary_proba.extend(test_proba)
            all_y_true.extend(y_te)

            high_conf = (confidence > self.threshold).sum()
            print(f"  Fold {fold+1}: {high_conf}/{len(confidence)} "
                  f"signals above threshold {self.threshold}")

        return {
            "confidence": np.array(all_confidence),
            "meta_proba": np.array(all_meta_proba),
            "primary_proba": np.array(all_primary_proba),
            "y_true": np.array(all_y_true),
        }

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "meta_model.txt")
        self.model.save_model(filepath)
        meta = {"feature_names": self.meta_feature_names,
                "threshold": self.threshold}
        joblib.dump(meta, filepath.replace(".txt", "_meta.pkl"))
        print(f"[Meta] Model saved to {filepath}")

    def load(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "meta_model.txt")
        self.model = lgb.Booster(model_file=filepath)
        meta = joblib.load(filepath.replace(".txt", "_meta.pkl"))
        self.meta_feature_names = meta["feature_names"]
        self.threshold = meta["threshold"]
        print(f"[Meta] Model loaded from {filepath}")

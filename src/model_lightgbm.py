"""
tradIA LightGBM Model Pipeline
Primary model: Gradient-boosted trees with walk-forward validation
and Optuna hyperparameter optimization.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
import os
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from src.config import (
    LGBM_PARAMS, WALK_FORWARD_FOLDS, SIGNAL_THRESHOLD,
    OPTUNA_N_TRIALS, RANDOM_SEED, MODELS_DIR
)


class LightGBMPipeline:
    """Complete LightGBM training, validation, and prediction pipeline."""

    def __init__(self, params: dict = None):
        self.params = params or LGBM_PARAMS.copy()
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.cv_results = []

    def prepare_data(
        self, df: pd.DataFrame, feature_cols: list
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Extract features and labels from DataFrame, dropping NaN labels."""
        labeled = df.dropna(subset=["label"]).copy()
        X = labeled[feature_cols].values
        y = labeled["label"].values.astype(int)
        timestamps = labeled.index
        self.feature_names = feature_cols
        return X, y, timestamps

    def walk_forward_split(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = WALK_FORWARD_FOLDS
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create walk-forward (expanding window) train/test splits.
        Each fold uses all previous data for training and the next chunk for testing.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        for train_idx, test_idx in tscv.split(X):
            splits.append((train_idx, test_idx))
        return splits

    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, n_trials: int = OPTUNA_N_TRIALS
    ) -> dict:
        """Use Optuna to find optimal hyperparameters via walk-forward CV."""
        splits = self.walk_forward_split(X, y, n_splits=3)  # fewer folds for speed

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "verbose": -1,
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "n_estimators": 1000,
            }

            scores = []
            for train_idx, test_idx in splits:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Compute class weights
                n_pos = y_train.sum()
                n_neg = len(y_train) - n_pos
                scale_pos_weight = n_neg / max(n_pos, 1)
                params["scale_pos_weight"] = scale_pos_weight

                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

                y_pred_proba = model.predict(X_test)
                try:
                    score = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    score = 0.5
                scores.append(score)

            return np.mean(scores)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_params.update({
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_estimators": 1000,
        })

        print(f"[LightGBM] Best Optuna score: {study.best_value:.4f}")
        print(f"[LightGBM] Best params: {self.best_params}")

        return self.best_params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        params: dict = None,
    ) -> lgb.Booster:
        """Train the LightGBM model."""
        params = params or self.best_params or self.params.copy()

        # Class weighting
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=self.feature_names,
        )

        callbacks = [
            lgb.log_evaluation(period=100),
        ]

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            callbacks.append(lgb.early_stopping(50, verbose=True))
            self.model = lgb.train(
                params, train_data,
                valid_sets=[valid_data],
                callbacks=callbacks,
            )
        else:
            n_est = params.pop("n_estimators", 1000)
            self.model = lgb.train(
                params, train_data,
                num_boost_round=n_est,
                callbacks=callbacks,
            )

        return self.model

    def predict(
        self, X: np.ndarray, threshold: float = SIGNAL_THRESHOLD
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with probability threshold.
        Returns (predicted_labels, probabilities).
        Only emits signals when confidence exceeds threshold.
        """
        proba = self.model.predict(X)

        # Apply threshold: only signal when confident
        labels = np.full(len(proba), -1, dtype=int)  # -1 = no signal
        labels[proba >= threshold] = 1       # BUY
        labels[proba <= (1 - threshold)] = 0  # SELL

        return labels, proba

    def walk_forward_evaluate(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = WALK_FORWARD_FOLDS
    ) -> Dict:
        """
        Full walk-forward evaluation.
        Returns metrics across all folds.
        """
        splits = self.walk_forward_split(X, y, n_splits)
        all_y_true = []
        all_y_pred = []
        all_y_proba = []

        print(f"\n[LightGBM] Walk-Forward Validation ({n_splits} folds)")
        print("=" * 50)

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Use last 20% of training data as validation for early stopping
            val_size = int(len(X_train) * 0.2)
            X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
            y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

            self.train(X_tr, y_tr, X_val, y_val)

            labels, proba = self.predict(X_test, threshold=0.5)

            all_y_true.extend(y_test)
            all_y_pred.extend(labels)
            all_y_proba.extend(proba)

            # Per-fold metrics
            try:
                auc = roc_auc_score(y_test, proba)
            except ValueError:
                auc = 0.5

            fold_result = {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "roc_auc": auc,
                "precision": precision_score(y_test, labels >= 0.5, zero_division=0),
                "recall": recall_score(y_test, labels >= 0.5, zero_division=0),
            }
            self.cv_results.append(fold_result)

            print(f"  Fold {fold+1}: AUC={auc:.4f}, "
                  f"Train={len(X_train):,}, Test={len(X_test):,}")

        # Aggregate metrics
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)

        # Use 0.5 threshold for overall metrics
        overall_labels = (all_y_proba >= 0.5).astype(int)

        results = {
            "cv_results": self.cv_results,
            "overall_auc": roc_auc_score(all_y_true, all_y_proba),
            "overall_precision": precision_score(all_y_true, overall_labels, zero_division=0),
            "overall_recall": recall_score(all_y_true, overall_labels, zero_division=0),
            "overall_f1": f1_score(all_y_true, overall_labels, zero_division=0),
            "classification_report": classification_report(
                all_y_true, overall_labels, target_names=["SELL", "BUY"]
            ),
            "confusion_matrix": confusion_matrix(all_y_true, overall_labels),
            "y_true": all_y_true,
            "y_pred": overall_labels,
            "y_proba": all_y_proba,
        }

        print(f"\n  Overall AUC: {results['overall_auc']:.4f}")
        print(f"  Overall F1:  {results['overall_f1']:.4f}")
        print(f"\n{results['classification_report']}")

        return results

    def save(self, filepath: str = None):
        """Save the trained model."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "lightgbm_model.txt")
        self.model.save_model(filepath)
        # Also save metadata
        meta = {
            "feature_names": self.feature_names,
            "best_params": self.best_params,
            "cv_results": self.cv_results,
        }
        joblib.dump(meta, filepath.replace(".txt", "_meta.pkl"))
        print(f"[LightGBM] Model saved to {filepath}")

    def load(self, filepath: str = None):
        """Load a saved model."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "lightgbm_model.txt")
        self.model = lgb.Booster(model_file=filepath)
        meta = joblib.load(filepath.replace(".txt", "_meta.pkl"))
        self.feature_names = meta["feature_names"]
        self.best_params = meta["best_params"]
        self.cv_results = meta["cv_results"]
        print(f"[LightGBM] Model loaded from {filepath}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        importance = self.model.feature_importance(importance_type="gain")
        feat_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return feat_imp

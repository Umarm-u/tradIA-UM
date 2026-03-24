"""
tradIA Model Interpretability — SHAP Analysis
Feature importance analysis to verify the model learns SMC/ICT concepts.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Optional

from src.config import RESULTS_DIR


def analyze_feature_importance(
    model,
    X: np.ndarray,
    feature_names: list,
    model_name: str = "LightGBM",
    save_dir: str = RESULTS_DIR,
    max_display: int = 30,
) -> pd.DataFrame:
    """
    Generate SHAP-based feature importance analysis.
    Produces summary plot and dependence plots for top features.
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        import shap

        print(f"[SHAP] Computing SHAP values for {model_name}...")

        # Use a sample for computational efficiency
        sample_size = min(5000, len(X))
        X_sample = X[:sample_size]

        if model_name == "LightGBM":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, shap_values may be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1 (BUY) values
        else:
            # Kernel SHAP for other models (slower)
            explainer = shap.KernelExplainer(
                model.predict if hasattr(model, 'predict') else model,
                shap.sample(X_sample, 100)
            )
            shap_values = explainer.shap_values(X_sample[:500])

        # 1. Summary plot (bar)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )
        plt.title(f"tradIA — {model_name} Feature Importance (SHAP)", fontsize=14)
        plt.tight_layout()
        bar_path = os.path.join(save_dir, f"shap_importance_{model_name.lower()}.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Bar plot saved to {bar_path}")

        # 2. Summary plot (dot/beeswarm)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title(f"tradIA — {model_name} SHAP Summary", fontsize=14)
        plt.tight_layout()
        dot_path = os.path.join(save_dir, f"shap_summary_{model_name.lower()}.png")
        plt.savefig(dot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Summary plot saved to {dot_path}")

        # 3. Compute mean absolute SHAP values (feature ranking)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feat_importance = pd.DataFrame({
            "feature": feature_names,
            "shap_importance": mean_abs_shap,
        }).sort_values("shap_importance", ascending=False)

        # Save importance table
        csv_path = os.path.join(save_dir, f"feature_importance_{model_name.lower()}.csv")
        feat_importance.to_csv(csv_path, index=False)
        print(f"  Importance CSV saved to {csv_path}")

        # 4. Print top features grouped by category
        print(f"\n  Top {min(20, len(feat_importance))} features by SHAP importance:")
        print(f"  {'Rank':<6}{'Feature':<35}{'SHAP Importance':<15}")
        print(f"  {'-'*56}")
        for idx, row in feat_importance.head(20).iterrows():
            print(f"  {feat_importance.index.get_loc(idx)+1:<6}"
                  f"{row['feature']:<35}{row['shap_importance']:.6f}")

        # 5. Categorize features and check SMC concept coverage
        smc_categories = {
            "Swing Structure": ["swing_high", "swing_low", "dist_to_swing", "bars_since_swing"],
            "Market Structure": ["bos_", "market_structure", "structure_shift", "structure_bias"],
            "FVG": ["fvg", "in_fvg"],
            "Liquidity": ["liquidity", "liq_swept"],
            "Displacement": ["displacement", "is_displacement"],
            "CISD": ["cisd", "range_expansion"],
            "Order Flow": ["consecutive_", "buying_pressure", "selling_pressure", "body_momentum"],
            "Multi-TF": ["htf_"],
            "Volatility": ["atr_", "volatility_", "momentum_"],
            "Candle": ["body_size", "wick", "body_to_range", "candle_range", "is_bullish", "is_doji"],
            "Context": ["hour_", "day_", "session_", "dist_to_daily", "position_in_daily"],
        }

        print(f"\n  SMC Concept Coverage in Top 20 Features:")
        top_20_features = set(feat_importance.head(20)["feature"].values)
        for category, keywords in smc_categories.items():
            matching = [f for f in top_20_features
                       if any(kw in f for kw in keywords)]
            status = "✓" if matching else "✗"
            count = len(matching)
            print(f"    {status} {category:<20} ({count} features in top 20)")

        return feat_importance

    except ImportError:
        print("[SHAP] SHAP library not available. Using built-in importance.")
        return _builtin_importance(model, feature_names, model_name, save_dir)

    except Exception as e:
        print(f"[SHAP] Error computing SHAP values: {e}")
        print("[SHAP] Falling back to built-in importance.")
        return _builtin_importance(model, feature_names, model_name, save_dir)


def _builtin_importance(
    model, feature_names: list, model_name: str, save_dir: str
) -> pd.DataFrame:
    """Fallback: use LightGBM's built-in feature importance."""
    try:
        importance = model.feature_importance(importance_type="gain")
        feat_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        plt.figure(figsize=(12, 8))
        top_n = feat_importance.head(30)
        plt.barh(range(len(top_n)), top_n["importance"].values)
        plt.yticks(range(len(top_n)), top_n["feature"].values)
        plt.xlabel("Importance (Gain)")
        plt.title(f"tradIA — {model_name} Feature Importance")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        filepath = os.path.join(save_dir, f"feature_importance_{model_name.lower()}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Importance plot saved to {filepath}")

        return feat_importance
    except Exception as e:
        print(f"  Could not compute importance: {e}")
        return pd.DataFrame()

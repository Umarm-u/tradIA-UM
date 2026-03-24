"""
tradIA - SELL Model Pipeline Runner (Enhanced)
All improvements integrated:
  1. Adaptive threshold search (0.55–0.80 range, precision-optimized)
  2. Optuna hyperparameter optimization (enabled by default, 30 trials)
  3. Meta-labeling as second-stage filter
  4. Configurable class ratio (default 2:1 for precision)
  5. Feature selection — remove bottom 20% low-importance features

Pipeline:
  1. Data loading & validation
  2. Feature engineering (ICT/SMC + SELL-specific)
  3. SELL label generation
  4. Feature selection (importance-based pruning)
  5. LightGBM training with Optuna
  6. Adaptive threshold search
  7. Meta-labeling filter
  8. Backtesting with strategy SL/TP
  9. SHAP interpretability
"""
import os
import sys
import time
import argparse

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_PATH, MODELS_DIR, RESULTS_DIR,
    OPTUNA_N_TRIALS, SIGNAL_THRESHOLD, RANDOM_SEED,
)
from src.data_loader import load_and_prepare_data
from src.feature_engineering import engineer_features, get_feature_columns
from src.sell_label_generator import generate_sell_labels
from src.model_lightgbm import LightGBMPipeline
from src.meta_labeler import MetaLabeler
from src.backtester import run_sell_backtest, compute_equity_curve, compute_backtest_metrics
from src.evaluation import (
    plot_equity_curve, plot_trade_distribution,
    analyze_signal_robustness,
)
from src.interpretability import analyze_feature_importance


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT 1: ADAPTIVE THRESHOLD SEARCH
# ═══════════════════════════════════════════════════════════════

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_thresh: float = 0.50,
    max_thresh: float = 0.85,
    step: float = 0.025,
    min_trades: int = 30,
) -> dict:
    """
    Search for the threshold that maximizes precision while maintaining
    a minimum number of signals. Returns dict with best threshold and metrics.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    results = []
    thresholds = np.arange(min_thresh, max_thresh + step, step)

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        n_signals = y_pred.sum()

        if n_signals < min_trades:
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Precision-weighted F-score (beta=0.5 favors precision)
        if prec + rec > 0:
            f05 = (1 + 0.25) * (prec * rec) / (0.25 * prec + rec)
        else:
            f05 = 0

        results.append({
            "threshold": round(thresh, 3),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f0.5": f05,
            "n_signals": int(n_signals),
        })

    if not results:
        return {"threshold": 0.55, "precision": 0, "recall": 0, "f1": 0, "f0.5": 0, "n_signals": 0}

    # Select best by F0.5 (precision-weighted)
    results_df = pd.DataFrame(results)
    best_idx = results_df["f0.5"].idxmax()
    best = results_df.iloc[best_idx].to_dict()

    print(f"\n  Threshold Search Results (top 5 by F0.5):")
    print(f"  {'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'F0.5':>7} {'Signals':>8}")
    print(f"  {'-'*49}")
    top5 = results_df.nlargest(5, "f0.5")
    for _, row in top5.iterrows():
        marker = " <--" if row["threshold"] == best["threshold"] else ""
        print(f"  {row['threshold']:>7.3f} {row['precision']:>7.4f} {row['recall']:>7.4f} "
              f"{row['f1']:>7.4f} {row['f0.5']:>7.4f} {int(row['n_signals']):>8}{marker}")

    return best


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT 5: FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════

def select_important_features(
    model,
    feature_cols: list,
    X: np.ndarray,
    keep_ratio: float = 0.80,
) -> tuple:
    """
    Remove bottom (1-keep_ratio) features by importance.
    Returns (selected_feature_names, selected_X, removed_feature_names).
    """
    importances = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    n_keep = max(int(len(feature_cols) * keep_ratio), 10)
    selected = [f for f, _ in feat_imp[:n_keep]]
    removed = [f for f, _ in feat_imp[n_keep:]]
    selected_indices = [feature_cols.index(f) for f in selected]

    X_selected = X[:, selected_indices]

    print(f"\n  Feature Selection: {len(feature_cols)} -> {len(selected)} features")
    print(f"  Removed {len(removed)} low-importance features:")
    for f, imp in feat_imp[n_keep:n_keep + 5]:
        print(f"    - {f} (importance: {imp:.1f})")
    if len(removed) > 5:
        print(f"    ... and {len(removed) - 5} more")

    return selected, X_selected, removed


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_sell_pipeline(
    skip_optuna: bool = False,
    optuna_trials: int = 30,
    validate_labels: bool = True,
    signal_threshold: float = None,  # None = auto-search
    neg_ratio: int = 2,  # Negative:Positive sampling ratio
    enable_meta: bool = True,
    feature_keep_ratio: float = 0.80,
):
    """
    Run the enhanced SELL model training pipeline with all improvements.

    Improvements over baseline:
      1. Adaptive threshold search (replaces fixed 0.55)
      2. Optuna enabled by default (30 trials)
      3. Meta-labeling second-stage filter
      4. Configurable neg_ratio (default 2:1 instead of 3:1)
      5. Feature selection (keep top 80% by importance)
    """
    print("\n" + "=" * 70)
    print("   tradIA — SELL Model Pipeline (Enhanced)")
    print("   Buyside Liquidity Sweep Strategy")
    print("=" * 70)
    print(f"   Optuna:         {'SKIP' if skip_optuna else f'{optuna_trials} trials'}")
    print(f"   Label valid.:   {'ON' if validate_labels else 'OFF'}")
    print(f"   Threshold:      {'AUTO-SEARCH' if signal_threshold is None else signal_threshold}")
    print(f"   Neg:Pos ratio:  {neg_ratio}:1")
    print(f"   Meta-labeling:  {'ON' if enable_meta else 'OFF'}")
    print(f"   Feature keep:   {feature_keep_ratio:.0%}")
    print("=" * 70 + "\n")

    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Data Loading
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 1: Data Loading & Validation")
    print("-" * 50)
    df = load_and_prepare_data()

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 2: Feature Engineering (ICT/SMC + SELL)")
    print("-" * 50)
    df = engineer_features(df)

    # ═══════════════════════════════════════════════════════════
    # STEP 3: SELL Label Generation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 3: SELL Label Generation")
    print("-" * 50)
    df, setups_df = generate_sell_labels(df, validate_outcome=validate_labels)

    label_count = df["label"].notna().sum()
    total_rows = len(df)
    print(f"\n  Labels: {label_count:,} SELL setups out of {total_rows:,} candles")
    print(f"  Label rate: {label_count / total_rows:.4%}")

    if label_count < 50:
        print("\n  [WARNING] Very few labels. Trying without outcome validation...")
        df, setups_df = generate_sell_labels(df, validate_outcome=False)
        label_count = df["label"].notna().sum()
        print(f"  Labels (no validation): {label_count:,}")
        if label_count < 30:
            print("  [ERROR] Too few labels for training. Aborting.")
            return None

    # Save setups
    if not setups_df.empty:
        setups_path = os.path.join(RESULTS_DIR, "sell_setups.csv")
        setups_df.to_csv(setups_path, index=False)
        print(f"  Setups saved to {setups_path}")

    # ═══════════════════════════════════════════════════════════
    # STEP 3b: IMPROVEMENT 4 — Configurable class ratio
    # ═══════════════════════════════════════════════════════════
    labeled_mask = df["label"].notna()
    positive_count = int(labeled_mask.sum())

    negative_indices = df.index[~labeled_mask]
    np.random.seed(RANDOM_SEED)

    n_negatives = min(positive_count * neg_ratio, len(negative_indices))
    neg_sample_idx = np.random.choice(
        len(negative_indices), size=n_negatives, replace=False
    )
    neg_sample_idx.sort()
    selected_negatives = negative_indices[neg_sample_idx]
    df.loc[selected_negatives, "label"] = 0

    total_labeled = df["label"].notna().sum()
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    print(f"\n  Training dataset: {total_labeled:,} samples ({neg_ratio}:1 neg:pos ratio)")
    print(f"    Positive (SELL): {pos:,} ({pos/total_labeled:.1%})")
    print(f"    Negative:        {neg:,} ({neg/total_labeled:.1%})")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Feature columns
    # ═══════════════════════════════════════════════════════════
    feature_cols = get_feature_columns(df)
    print(f"\n  Total features: {len(feature_cols)}")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: PHASE 1 — Initial LightGBM Training
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 5: LightGBM SELL Model Training (Phase 1)")
    print("-" * 50)

    lgbm = LightGBMPipeline()
    X, y, timestamps = lgbm.prepare_data(df, feature_cols)
    print(f"  Training data: {len(X):,} labeled samples")
    print(f"  Class: SELL={int(y.sum()):,}, No-setup={int(len(y) - y.sum()):,}")

    # Optuna — IMPROVEMENT 2
    if not skip_optuna:
        print(f"\n  Running Optuna hyperparameter search ({optuna_trials} trials)...")
        lgbm.optimize_hyperparameters(X, y, n_trials=optuna_trials)

    # Walk-forward evaluation
    print("\n  Running walk-forward evaluation...")
    lgbm_results = lgbm.walk_forward_evaluate(X, y)

    # Train initial model
    val_size = int(len(X) * 0.15)
    lgbm.train(X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:])

    # ═══════════════════════════════════════════════════════════
    # STEP 5b: IMPROVEMENT 5 — Feature Selection
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 5b: Feature Selection (Importance-based)")
    print("-" * 50)

    selected_features, X_selected, removed_features = select_important_features(
        lgbm.model, feature_cols, X, keep_ratio=feature_keep_ratio
    )

    # Retrain with selected features
    print("\n  Retraining with selected features...")
    lgbm2 = LightGBMPipeline()

    if not skip_optuna:
        print(f"  Re-running Optuna with {len(selected_features)} features...")
        lgbm2.optimize_hyperparameters(X_selected, y, n_trials=optuna_trials)

    val_size2 = int(len(X_selected) * 0.15)
    lgbm2.train(
        X_selected[:-val_size2], y[:-val_size2],
        X_selected[-val_size2:], y[-val_size2:]
    )

    # Save model
    sell_model_path = os.path.join(MODELS_DIR, "sell_lightgbm_model.txt")
    lgbm2.save(sell_model_path)

    # Final predictions
    _, final_proba = lgbm2.predict(X_selected)

    # ═══════════════════════════════════════════════════════════
    # STEP 6: IMPROVEMENT 1 — Adaptive Threshold Search
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 6: Signal Generation & Threshold Search")
    print("-" * 50)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    labeled_df = df.dropna(subset=["label"])
    min_len = min(len(labeled_df), len(final_proba))
    y_true_full = labeled_df["label"].values[:min_len]

    try:
        auc = roc_auc_score(y_true_full, final_proba[:min_len])
        print(f"\n  Model AUC: {auc:.4f}")
    except Exception as e:
        auc = 0.5
        print(f"  AUC computation failed: {e}")

    if signal_threshold is None:
        # Auto-search for best threshold
        best = find_optimal_threshold(y_true_full, final_proba[:min_len])
        optimal_threshold = best["threshold"]
        print(f"\n  Selected threshold: {optimal_threshold:.3f} "
              f"(precision={best['precision']:.4f}, F0.5={best['f0.5']:.4f})")
    else:
        optimal_threshold = signal_threshold
        print(f"  Using fixed threshold: {optimal_threshold}")

    # Generate signals at optimal threshold
    sell_signals = np.full(len(final_proba), -1, dtype=int)
    sell_signals[final_proba >= optimal_threshold] = 1

    n_signals = (sell_signals == 1).sum()
    print(f"\n  SELL signals generated: {n_signals:,}")
    print(f"  Signal rate: {n_signals / len(sell_signals):.2%}")

    # Classification metrics at chosen threshold
    y_pred_thresh = (final_proba[:min_len] >= optimal_threshold).astype(int)
    prec = precision_score(y_true_full, y_pred_thresh, zero_division=0)
    rec = recall_score(y_true_full, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_true_full, y_pred_thresh, zero_division=0)
    print(f"\n  Final Metrics (threshold={optimal_threshold:.3f}):")
    print(f"    AUC:       {auc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")

    # ═══════════════════════════════════════════════════════════
    # STEP 7: IMPROVEMENT 3 — Meta-Labeling
    # ═══════════════════════════════════════════════════════════
    meta_confidence = final_proba.copy()

    if enable_meta:
        print("\n" + "-" * 50)
        print("  STEP 7: Meta-Labeling Filter")
        print("-" * 50)

        try:
            meta = MetaLabeler(threshold=0.50)

            # Create meta-labels: use strategy SL/TP to determine profitability
            # For SELL signals, profitable = price dropped to TP before SL
            primary_labels = (final_proba >= optimal_threshold).astype(int)
            meta_labels = meta.create_meta_labels(
                labeled_df.iloc[:min_len], primary_labels[:min_len]
            )

            # Extract meta features
            X_meta, meta_feat_names = meta.extract_meta_features(
                labeled_df.iloc[:min_len], final_proba[:min_len]
            )

            meta_min = min(len(X_meta), len(meta_labels))

            if meta_labels[:meta_min].sum() >= 20:
                # Train meta model
                meta.train(X_meta[:meta_min], meta_labels[:meta_min], meta_feat_names)

                # Get meta predictions
                meta_proba = meta.predict(X_meta[:meta_min])
                meta_confidence_raw = meta.get_trade_confidence(
                    final_proba[:meta_min], meta_proba
                )

                # Apply meta filter: only keep signals where meta confidence > threshold
                meta_threshold = 0.30
                meta_pass = meta_proba >= meta_threshold

                # Update sell_signals with meta filter
                for i in range(min(len(sell_signals), meta_min)):
                    if sell_signals[i] == 1 and not meta_pass[i]:
                        sell_signals[i] = -1  # Filter out low-confidence signal

                n_filtered = n_signals - (sell_signals == 1).sum()
                n_remaining = (sell_signals == 1).sum()
                print(f"\n  Meta-filter results:")
                print(f"    Signals before meta:  {n_signals:,}")
                print(f"    Filtered out:         {n_filtered:,}")
                print(f"    Signals after meta:   {n_remaining:,}")
                print(f"    Filter rate:          {n_filtered/max(n_signals,1):.1%}")

                # Update confidence for backtest
                meta_confidence[:meta_min] = meta_confidence_raw

                # Save meta model
                meta.save(os.path.join(MODELS_DIR, "sell_meta_model.txt"))
                n_signals = n_remaining
            else:
                print(f"  [WARNING] Too few profitable meta-labels ({int(meta_labels[:meta_min].sum())}). "
                      f"Skipping meta-labeling.")

        except Exception as e:
            print(f"  [WARNING] Meta-labeling failed: {e}")
            print("  Continuing without meta-labeling...")

    # ═══════════════════════════════════════════════════════════
    # STEP 8: Backtesting
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 8: SELL Backtesting")
    print("-" * 50)

    backtest_results = run_sell_backtest(
        df, sell_signals, setups_df,
        confidence=meta_confidence,
        model_name="SELL_enhanced",
    )

    # Plot backtest results
    if backtest_results and backtest_results.get("trades_df") is not None:
        trades_df = backtest_results["trades_df"]
        if not trades_df.empty:
            plot_equity_curve(backtest_results["equity_df"], "SELL_enhanced")
            plot_trade_distribution(trades_df, "SELL_enhanced")
            if len(trades_df) >= 50:
                analyze_signal_robustness(trades_df, "SELL_enhanced")

    # ═══════════════════════════════════════════════════════════
    # STEP 9: Feature Importance & SHAP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("  STEP 9: Feature Importance & Interpretability")
    print("-" * 50)

    feat_imp = analyze_feature_importance(
        lgbm2.model, X_selected, selected_features,
        model_name="SELL_Enhanced_LightGBM"
    )

    # ═══════════════════════════════════════════════════════════
    # STEP 10: Save Comprehensive Report
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    report_path = os.path.join(RESULTS_DIR, "sell_enhanced_summary.txt")
    with open(report_path, "w") as f:
        f.write("tradIA SELL Pipeline — Enhanced Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Runtime:             {elapsed/60:.1f} minutes\n")
        f.write(f"Optuna:              {'SKIP' if skip_optuna else f'{optuna_trials} trials'}\n")
        f.write(f"Neg:Pos ratio:       {neg_ratio}:1\n")
        f.write(f"Features (initial):  {len(feature_cols)}\n")
        f.write(f"Features (selected): {len(selected_features)}\n")
        f.write(f"Features removed:    {len(removed_features)}\n")
        f.write(f"SELL setups:         {positive_count}\n")
        f.write(f"Model AUC:           {auc:.4f}\n")
        f.write(f"Threshold:           {optimal_threshold:.3f}\n")
        f.write(f"Precision:           {prec:.4f}\n")
        f.write(f"Recall:              {rec:.4f}\n")
        f.write(f"F1:                  {f1:.4f}\n")
        f.write(f"SELL signals:        {n_signals}\n")
        f.write(f"Meta-labeling:       {'ON' if enable_meta else 'OFF'}\n")
        if backtest_results and backtest_results.get("metrics"):
            m = backtest_results["metrics"]
            f.write(f"\nBacktest Results:\n")
            for k, v in m.items():
                f.write(f"  {k:25s}: {v}\n")

    print(f"\n  Report saved to {report_path}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("   tradIA SELL Pipeline (Enhanced) Complete!")
    print(f"   Total time:       {elapsed / 60:.1f} minutes")
    print(f"   SELL setups:      {positive_count:,}")
    print(f"   Features used:    {len(selected_features)} / {len(feature_cols)}")
    print(f"   Threshold:        {optimal_threshold:.3f}")
    print(f"   Precision:        {prec:.4f}")
    print(f"   SELL signals:     {n_signals:,}")
    if backtest_results and backtest_results.get("metrics"):
        m = backtest_results["metrics"]
        print(f"   Win rate:         {m.get('win_rate', 0):.2%}")
        print(f"   Profit factor:    {m.get('profit_factor', 0):.2f}")
        print(f"   Total return:     {m.get('total_return_pct', 0):.2f}%")
    print(f"   Model:            {sell_model_path}")
    print(f"   Results:          {RESULTS_DIR}")
    print("=" * 70 + "\n")

    return {
        "lgbm_results": lgbm_results,
        "backtest_results": backtest_results,
        "feature_importance": feat_imp,
        "setups_df": setups_df,
        "optimal_threshold": optimal_threshold,
        "selected_features": selected_features,
        "removed_features": removed_features,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tradIA SELL Model Pipeline (Enhanced)")
    parser.add_argument(
        "--skip-optuna", action="store_true",
        help="Skip Optuna hyperparameter optimization"
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=30,
        help="Number of Optuna trials (default: 30)"
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Disable outcome validation for labels"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed signal threshold (default: auto-search)"
    )
    parser.add_argument(
        "--neg-ratio", type=int, default=2,
        help="Negative:Positive sampling ratio (default: 2)"
    )
    parser.add_argument(
        "--no-meta", action="store_true",
        help="Disable meta-labeling"
    )
    parser.add_argument(
        "--feature-keep", type=float, default=0.80,
        help="Fraction of features to keep (default: 0.80)"
    )

    args = parser.parse_args()
    run_sell_pipeline(
        skip_optuna=args.skip_optuna,
        optuna_trials=args.optuna_trials,
        validate_labels=not args.no_validate,
        signal_threshold=args.threshold,
        neg_ratio=args.neg_ratio,
        enable_meta=not args.no_meta,
        feature_keep_ratio=args.feature_keep,
    )

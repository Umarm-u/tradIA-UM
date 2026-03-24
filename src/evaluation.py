"""
tradIA Model Evaluation Framework
Classification metrics, signal quality analysis, and reporting.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Optional

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)

from src.config import RESULTS_DIR, SIGNAL_THRESHOLD


def evaluate_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict:
    """
    Comprehensive classification evaluation.
    Returns metrics dict and prints formatted report.
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "precision_buy": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_sell": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_buy": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_sell": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_buy": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_sell": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "total_predictions": len(y_true),
        "buy_signals": int(y_pred.sum()),
        "sell_signals": int((1 - y_pred).sum()),
    }

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Classification Report")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=["SELL", "BUY"]))
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision (AP): {metrics['avg_precision']:.4f}")
    print(f"  Total Predictions:  {metrics['total_predictions']:,}")
    print(f"  BUY signals:        {metrics['buy_signals']:,}")
    print(f"  SELL signals:       {metrics['sell_signals']:,}")
    print(f"{'=' * 60}\n")

    return metrics


def evaluate_signal_quality(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float = SIGNAL_THRESHOLD,
    lookahead: int = 16,
    model_name: str = "Model",
) -> Dict:
    """
    Evaluate the quality of trading signals.
    Measures win rate, average profit, and signal consistency.
    """
    labeled_df = df.dropna(subset=["label"]).copy()

    # Align lengths
    min_len = min(len(labeled_df), len(y_proba))
    labeled_df = labeled_df.iloc[:min_len].copy()
    proba = y_proba[:min_len]

    # Generate signals using threshold
    buy_mask = proba >= threshold
    sell_mask = proba <= (1 - threshold)
    no_signal = ~buy_mask & ~sell_mask

    close = labeled_df["close"].values
    n = len(close)

    # Calculate P&L for each signal
    buy_pnl = []
    sell_pnl = []

    for i in range(n - lookahead):
        future_price = close[i + lookahead]
        entry_price = close[i]
        pnl_pct = (future_price - entry_price) / entry_price * 100

        if buy_mask[i]:
            buy_pnl.append(pnl_pct)
        elif sell_mask[i]:
            sell_pnl.append(-pnl_pct)  # inverted for sell

    signal_metrics = {
        "model": model_name,
        "threshold": threshold,
        "total_candles": n,
        "buy_signals": int(buy_mask.sum()),
        "sell_signals": int(sell_mask.sum()),
        "no_signal": int(no_signal.sum()),
        "signal_rate": (buy_mask.sum() + sell_mask.sum()) / n,
    }

    if buy_pnl:
        signal_metrics["buy_win_rate"] = sum(1 for p in buy_pnl if p > 0) / len(buy_pnl)
        signal_metrics["buy_avg_pnl"] = np.mean(buy_pnl)
        signal_metrics["buy_median_pnl"] = np.median(buy_pnl)
    else:
        signal_metrics["buy_win_rate"] = 0
        signal_metrics["buy_avg_pnl"] = 0
        signal_metrics["buy_median_pnl"] = 0

    if sell_pnl:
        signal_metrics["sell_win_rate"] = sum(1 for p in sell_pnl if p > 0) / len(sell_pnl)
        signal_metrics["sell_avg_pnl"] = np.mean(sell_pnl)
        signal_metrics["sell_median_pnl"] = np.median(sell_pnl)
    else:
        signal_metrics["sell_win_rate"] = 0
        signal_metrics["sell_avg_pnl"] = 0
        signal_metrics["sell_median_pnl"] = 0

    # Overall
    all_pnl = buy_pnl + sell_pnl
    if all_pnl:
        signal_metrics["overall_win_rate"] = sum(1 for p in all_pnl if p > 0) / len(all_pnl)
        signal_metrics["overall_avg_pnl"] = np.mean(all_pnl)
        signal_metrics["profit_factor"] = (
            sum(p for p in all_pnl if p > 0) / max(abs(sum(p for p in all_pnl if p < 0)), 1e-10)
        )
    else:
        signal_metrics["overall_win_rate"] = 0
        signal_metrics["overall_avg_pnl"] = 0
        signal_metrics["profit_factor"] = 0

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Signal Quality Report")
    print(f"{'=' * 60}")
    print(f"  Threshold:         {threshold}")
    print(f"  BUY signals:       {signal_metrics['buy_signals']:,}")
    print(f"  SELL signals:      {signal_metrics['sell_signals']:,}")
    print(f"  No signal:         {signal_metrics['no_signal']:,}")
    print(f"  Signal rate:       {signal_metrics['signal_rate']:.2%}")
    print(f"  ---")
    print(f"  BUY win rate:      {signal_metrics['buy_win_rate']:.2%}")
    print(f"  BUY avg PnL:       {signal_metrics['buy_avg_pnl']:.3f}%")
    print(f"  SELL win rate:     {signal_metrics['sell_win_rate']:.2%}")
    print(f"  SELL avg PnL:      {signal_metrics['sell_avg_pnl']:.3f}%")
    print(f"  ---")
    print(f"  Overall win rate:  {signal_metrics['overall_win_rate']:.2%}")
    print(f"  Profit factor:     {signal_metrics['profit_factor']:.2f}")
    print(f"{'=' * 60}\n")

    return signal_metrics


def plot_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_dir: str = RESULTS_DIR,
) -> None:
    """Generate and save evaluation plots."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"tradIA — {model_name} Evaluation", fontsize=16, fontweight="bold")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0, 0].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.4f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[0, 1].plot(recall, precision, "r-", linewidth=2, label=f"AP = {ap:.4f}")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, (y_proba >= 0.5).astype(int))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["SELL", "BUY"], yticklabels=["SELL", "BUY"],
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("Actual")

    # 4. Probability Distribution
    axes[1, 1].hist(
        y_proba[y_true == 0], bins=50, alpha=0.6, label="SELL (actual)", color="red"
    )
    axes[1, 1].hist(
        y_proba[y_true == 1], bins=50, alpha=0.6, label="BUY (actual)", color="green"
    )
    axes[1, 1].axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Prediction Probability Distribution")
    axes[1, 1].set_xlabel("Predicted Probability (BUY)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_evaluation.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Plots saved to {filepath}")


def generate_evaluation_report(
    results: Dict,
    signal_metrics: Dict,
    model_name: str = "Model",
    save_dir: str = RESULTS_DIR,
) -> str:
    """Generate a text-based evaluation report."""
    os.makedirs(save_dir, exist_ok=True)

    report_lines = [
        f"{'=' * 60}",
        f"tradIA — {model_name} Complete Evaluation Report",
        f"{'=' * 60}",
        "",
        "## Classification Metrics",
        f"ROC-AUC:             {results.get('roc_auc', results.get('overall_auc', 0)):.4f}",
        f"Average Precision:   {results.get('avg_precision', 0):.4f}",
        f"F1 (macro):          {results.get('f1_macro', results.get('overall_f1', 0)):.4f}",
        "",
        "## Signal Quality",
        f"Signal Rate:         {signal_metrics.get('signal_rate', 0):.2%}",
        f"Overall Win Rate:    {signal_metrics.get('overall_win_rate', 0):.2%}",
        f"Profit Factor:       {signal_metrics.get('profit_factor', 0):.2f}",
        f"BUY Win Rate:        {signal_metrics.get('buy_win_rate', 0):.2%}",
        f"SELL Win Rate:       {signal_metrics.get('sell_win_rate', 0):.2%}",
        "",
        f"{'=' * 60}",
    ]

    report_text = "\n".join(report_lines)
    filepath = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_report.txt")
    with open(filepath, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n[Eval] Report saved to {filepath}")

    return report_text


# ──────────────────────────────────────────────────────────────
# Backtest Evaluation
# ──────────────────────────────────────────────────────────────

def plot_equity_curve(
    equity_df: pd.DataFrame,
    model_name: str = "Model",
    save_dir: str = RESULTS_DIR,
) -> None:
    """Plot equity curve and drawdown."""
    if equity_df.empty:
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    fig.suptitle(f"tradIA - {model_name} Backtest", fontsize=16, fontweight="bold")

    # Equity curve
    ax1.plot(equity_df["trade_num"], equity_df["equity"], "b-", linewidth=1.5)
    ax1.fill_between(
        equity_df["trade_num"], equity_df["equity"],
        equity_df["equity"].iloc[0], alpha=0.1, color="blue"
    )
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=equity_df["equity"].iloc[0], color="gray", linestyle="--", alpha=0.5)

    # Drawdown
    ax2.fill_between(
        equity_df["trade_num"], equity_df["drawdown_pct"], 0,
        color="red", alpha=0.4
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(save_dir, f"backtest_equity_{model_name.lower()}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Equity curve saved to {fpath}")


def plot_trade_distribution(
    trades_df: pd.DataFrame,
    model_name: str = "Model",
    save_dir: str = RESULTS_DIR,
) -> None:
    """Plot histogram of trade returns."""
    if trades_df.empty:
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"tradIA - {model_name} Trade Distribution", fontsize=14, fontweight="bold")

    pnl = trades_df["net_pnl_pct"].values

    axes[0].hist(pnl, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[0].axvline(x=np.mean(pnl), color="green", linestyle="-", alpha=0.7,
                    label=f"Mean: {np.mean(pnl):.3f}%")
    axes[0].set_title("Return Distribution (all trades)")
    axes[0].set_xlabel("Return (%)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per direction
    for d, color in [("BUY", "green"), ("SELL", "red")]:
        d_pnl = trades_df[trades_df["direction"] == d]["net_pnl_pct"].values
        if len(d_pnl) > 0:
            axes[1].hist(d_pnl, bins=30, alpha=0.5, label=f"{d} ({len(d_pnl)})",
                         color=color, edgecolor="white")

    axes[1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_title("Return by Direction")
    axes[1].set_xlabel("Return (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(save_dir, f"backtest_trades_{model_name.lower()}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Trade distribution saved to {fpath}")


def analyze_signal_robustness(
    trades_df: pd.DataFrame,
    model_name: str = "Model",
    save_dir: str = RESULTS_DIR,
    rolling_window: int = 50,
) -> None:
    """Analyze signal frequency and win rate stability over time."""
    if trades_df.empty or len(trades_df) < rolling_window:
        return

    os.makedirs(save_dir, exist_ok=True)

    wins = (trades_df["net_pnl_pct"] > 0).astype(int)
    rolling_wr = wins.rolling(rolling_window, min_periods=10).mean()
    rolling_pnl = trades_df["net_pnl_pct"].rolling(rolling_window, min_periods=10).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"tradIA - {model_name} Strategy Robustness", fontsize=14, fontweight="bold")

    ax1.plot(rolling_wr.values, "b-", linewidth=1.5, label=f"Rolling {rolling_window}-trade WR")
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Break-even")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Rolling Win Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(rolling_pnl.values, "g-", linewidth=1.5,
             label=f"Rolling {rolling_window}-trade avg PnL")
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Avg Return (%)")
    ax2.set_xlabel("Trade #")
    ax2.set_title("Rolling Average PnL")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(save_dir, f"robustness_{model_name.lower()}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Robustness analysis saved to {fpath}")


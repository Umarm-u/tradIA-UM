"""
tradIA Backtesting Framework
Walk-forward backtester with realistic trade simulation
(commission, slippage, ATR-based TP/SL exits).
"""
import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple

from src.config import (
    BACKTEST_COMMISSION, BACKTEST_SLIPPAGE,
    BACKTEST_TP_ATR_MULT, BACKTEST_SL_ATR_MULT,
    BACKTEST_MAX_HOLD, INITIAL_CAPITAL, RESULTS_DIR,
)


def simulate_trades(
    df: pd.DataFrame,
    signals: np.ndarray,
    confidence: np.ndarray = None,
    tp_mult: float = BACKTEST_TP_ATR_MULT,
    sl_mult: float = BACKTEST_SL_ATR_MULT,
    max_hold: int = BACKTEST_MAX_HOLD,
    commission: float = BACKTEST_COMMISSION,
    slippage: float = BACKTEST_SLIPPAGE,
) -> pd.DataFrame:
    """
    Simulate trades from signal array.

    Entry: at the next candle's open after signal.
    Exit: at TP (tp_mult * ATR), SL (sl_mult * ATR), or time-barrier.

    Returns DataFrame of trade results.
    """
    labeled = df.dropna(subset=["label"])
    n = min(len(labeled), len(signals))
    labeled = labeled.iloc[:n]

    opens = labeled["open"].values
    highs = labeled["high"].values
    lows = labeled["low"].values
    closes = labeled["close"].values
    atr = labeled["atr_14"].values
    timestamps = labeled.index

    trades = []

    i = 0
    while i < n - 1:
        if signals[i] == -1:
            i += 1
            continue

        direction = signals[i]  # 1=BUY, 0=SELL

        # Entry at next candle's open
        entry_idx = i + 1
        if entry_idx >= n or np.isnan(atr[i]) or atr[i] == 0:
            i += 1
            continue

        entry_price = opens[entry_idx]
        # Apply slippage
        if direction == 1:
            entry_price *= (1 + slippage)
        else:
            entry_price *= (1 - slippage)

        # Set barriers
        if direction == 1:  # BUY
            tp_price = entry_price + tp_mult * atr[i]
            sl_price = entry_price - sl_mult * atr[i]
        else:  # SELL
            tp_price = entry_price - tp_mult * atr[i]
            sl_price = entry_price + sl_mult * atr[i]

        # Scan forward for exit
        exit_price = None
        exit_reason = "time"
        exit_idx = entry_idx

        for j in range(entry_idx, min(entry_idx + max_hold, n)):
            if direction == 1:  # BUY
                if highs[j] >= tp_price:
                    exit_price = tp_price * (1 - slippage)
                    exit_reason = "TP"
                    exit_idx = j
                    break
                elif lows[j] <= sl_price:
                    exit_price = sl_price * (1 - slippage)
                    exit_reason = "SL"
                    exit_idx = j
                    break
            else:  # SELL
                if lows[j] <= tp_price:
                    exit_price = tp_price * (1 + slippage)
                    exit_reason = "TP"
                    exit_idx = j
                    break
                elif highs[j] >= sl_price:
                    exit_price = sl_price * (1 + slippage)
                    exit_reason = "SL"
                    exit_idx = j
                    break

        if exit_price is None:
            exit_idx = min(entry_idx + max_hold - 1, n - 1)
            exit_price = closes[exit_idx]

        # PnL calculation
        if direction == 1:
            gross_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            gross_pnl_pct = (entry_price - exit_price) / entry_price

        net_pnl_pct = gross_pnl_pct - 2 * commission  # entry + exit commission

        trade = {
            "entry_time": timestamps[entry_idx] if entry_idx < len(timestamps) else None,
            "exit_time": timestamps[exit_idx] if exit_idx < len(timestamps) else None,
            "direction": "BUY" if direction == 1 else "SELL",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "gross_pnl_pct": gross_pnl_pct * 100,
            "net_pnl_pct": net_pnl_pct * 100,
            "holding_bars": exit_idx - entry_idx,
            "confidence": confidence[i] if confidence is not None and i < len(confidence) else 0,
        }
        trades.append(trade)

        # Skip ahead past this trade
        i = exit_idx + 1

    return pd.DataFrame(trades)


def compute_equity_curve(
    trades_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Compute cumulative equity curve from trade results."""
    if trades_df.empty:
        return pd.DataFrame({"equity": [initial_capital]})

    equity = [initial_capital]
    for _, trade in trades_df.iterrows():
        pnl = equity[-1] * (trade["net_pnl_pct"] / 100)
        equity.append(equity[-1] + pnl)

    eq_df = pd.DataFrame({
        "trade_num": range(len(equity)),
        "equity": equity,
    })

    # Drawdown
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["drawdown_pct"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"] * 100

    return eq_df


def compute_backtest_metrics(trades_df: pd.DataFrame) -> Dict:
    """Compute comprehensive backtesting metrics."""
    if trades_df.empty:
        return {"error": "No trades generated"}

    pnl = trades_df["net_pnl_pct"].values
    n_trades = len(pnl)

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    # Equity for drawdown
    eq = compute_equity_curve(trades_df)

    # Time span
    if "entry_time" in trades_df.columns and trades_df["entry_time"].iloc[0] is not None:
        t0 = pd.Timestamp(trades_df["entry_time"].iloc[0])
        t1 = pd.Timestamp(trades_df["entry_time"].iloc[-1])
        days = max((t1 - t0).days, 1)
    else:
        days = 1

    metrics = {
        "total_trades": n_trades,
        "trades_per_day": n_trades / days,
        "win_rate": len(wins) / n_trades if n_trades > 0 else 0,
        "avg_return_pct": np.mean(pnl),
        "median_return_pct": np.median(pnl),
        "total_return_pct": np.sum(pnl),
        "avg_win_pct": np.mean(wins) if len(wins) > 0 else 0,
        "avg_loss_pct": np.mean(losses) if len(losses) > 0 else 0,
        "max_drawdown_pct": eq["drawdown_pct"].min(),
        "profit_factor": (
            np.sum(wins) / max(np.abs(np.sum(losses)), 1e-10)
            if len(losses) > 0 else float("inf")
        ),
        "sharpe_ratio": (
            np.mean(pnl) / max(np.std(pnl), 1e-10) * np.sqrt(252 * 4)
            # Annualized: ~4 trades/day assumption
        ),
        "calmar_ratio": (
            np.sum(pnl) / max(abs(eq["drawdown_pct"].min()), 1e-10)
        ),
        "avg_holding_bars": trades_df["holding_bars"].mean(),
        "exit_reason_counts": trades_df["exit_reason"].value_counts().to_dict(),
    }

    # Per-direction breakdown
    for d in ["BUY", "SELL"]:
        d_trades = trades_df[trades_df["direction"] == d]
        if len(d_trades) > 0:
            d_pnl = d_trades["net_pnl_pct"].values
            d_wins = d_pnl[d_pnl > 0]
            metrics[f"{d.lower()}_trades"] = len(d_trades)
            metrics[f"{d.lower()}_win_rate"] = len(d_wins) / len(d_trades)
            metrics[f"{d.lower()}_avg_return"] = np.mean(d_pnl)
        else:
            metrics[f"{d.lower()}_trades"] = 0
            metrics[f"{d.lower()}_win_rate"] = 0
            metrics[f"{d.lower()}_avg_return"] = 0

    return metrics


def run_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    confidence: np.ndarray = None,
    model_name: str = "Model",
) -> Dict:
    """
    Run the complete backtesting pipeline and print results.

    Returns dict with trades_df, equity_df, and metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Backtesting Results")
    print(f"{'=' * 60}")

    trades_df = simulate_trades(df, signals, confidence)

    if trades_df.empty:
        print("  No trades generated. Adjust signal thresholds.")
        return {"trades_df": trades_df, "equity_df": pd.DataFrame(), "metrics": {}}

    equity_df = compute_equity_curve(trades_df)
    metrics = compute_backtest_metrics(trades_df)

    print(f"  Total trades:        {metrics['total_trades']:,}")
    print(f"  Trades per day:      {metrics['trades_per_day']:.2f}")
    print(f"  Win rate:            {metrics['win_rate']:.2%}")
    print(f"  Avg return/trade:    {metrics['avg_return_pct']:.3f}%")
    print(f"  Total return:        {metrics['total_return_pct']:.2f}%")
    print(f"  Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"  Sharpe ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"  ---")
    print(f"  BUY trades:          {metrics.get('buy_trades', 0):,} "
          f"(win rate: {metrics.get('buy_win_rate', 0):.2%})")
    print(f"  SELL trades:         {metrics.get('sell_trades', 0):,} "
          f"(win rate: {metrics.get('sell_win_rate', 0):.2%})")
    print(f"  Exit reasons:        {metrics['exit_reason_counts']}")
    print(f"  Avg holding bars:    {metrics['avg_holding_bars']:.1f}")
    print(f"{'=' * 60}\n")

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, f"backtest_report_{model_name.lower()}.txt")
    with open(report_path, "w") as f:
        f.write(f"tradIA Backtest Report - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        for k, v in metrics.items():
            if k != "exit_reason_counts":
                f.write(f"{k:25s}: {v}\n")
            else:
                f.write(f"{'exit_reasons':25s}: {v}\n")
    print(f"[Backtest] Report saved to {report_path}")

    # Save trades log
    trades_path = os.path.join(RESULTS_DIR, f"trades_{model_name.lower()}.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"[Backtest] Trade log saved to {trades_path}")

    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "metrics": metrics,
    }


def simulate_sell_trades(
    df: pd.DataFrame,
    signals: np.ndarray,
    setups_df: pd.DataFrame,
    confidence: np.ndarray = None,
    commission: float = BACKTEST_COMMISSION,
    slippage: float = BACKTEST_SLIPPAGE,
    trailing_stop: bool = True,
    trailing_activation: float = 0.5,
    trailing_distance: float = 0.3,
) -> pd.DataFrame:
    """
    Simulate SELL trades using strategy-specific entry/SL/TP from setups_df.

    Trailing stop: once price moves trailing_activation × TP_distance toward TP,
    the SL tightens to trail at trailing_distance × SL_distance from lowest low.
    """
    from src.config import (
        SELL_TRAILING_STOP, SELL_TRAILING_ACTIVATION, SELL_TRAILING_DISTANCE,
    )

    if trailing_stop is True:
        trailing_stop = SELL_TRAILING_STOP
    trailing_activation = SELL_TRAILING_ACTIVATION
    trailing_distance_pct = SELL_TRAILING_DISTANCE

    labeled = df.dropna(subset=["label"])
    n = min(len(labeled), len(signals))

    highs = labeled["high"].values
    lows = labeled["low"].values
    closes = labeled["close"].values
    timestamps = labeled.index

    # Build a lookup from labeled index → setup info
    setup_lookup = {}
    if not setups_df.empty:
        for _, setup in setups_df.iterrows():
            ts = setup["timestamp"]
            if ts in labeled.index:
                loc = labeled.index.get_loc(ts)
                if isinstance(loc, int) or isinstance(loc, np.integer):
                    setup_lookup[int(loc)] = setup

    trades = []
    i = 0
    while i < n - 1:
        if signals[i] != 1:  # Only SELL signals (label=1 in our scheme)
            i += 1
            continue

        # Look up the setup details
        if i not in setup_lookup:
            i += 1
            continue

        setup = setup_lookup[i]
        entry_price = setup["entry_price"] * (1 - slippage)
        sl_price = setup["sl_price"]
        tp_price = setup["tp_price"]
        sl_distance = sl_price - entry_price

        # Trailing stop state
        trail_active = False
        current_sl = sl_price
        best_price = entry_price  # Track lowest price for trailing

        # Scan forward for exit
        exit_price = None
        exit_reason = "time"
        exit_idx = i + 1
        max_hold = 96  # 24 hours

        for j in range(i + 1, min(i + max_hold + 1, n)):
            # Update best (lowest) price for SELL trailing
            if lows[j] < best_price:
                best_price = lows[j]

            # Check if trailing stop should activate
            if trailing_stop and not trail_active:
                price_moved = entry_price - best_price
                activation_target = trailing_activation * (entry_price - tp_price)
                if price_moved >= activation_target:
                    trail_active = True
                    # Set trailing SL below entry to lock in some profit
                    trail_dist = trailing_distance_pct * sl_distance
                    current_sl = best_price + trail_dist

            # Update trailing SL if active (tighten as price moves favorably)
            if trail_active:
                new_sl = best_price + trailing_distance_pct * sl_distance
                if new_sl < current_sl:
                    current_sl = new_sl

            # SL hit: price goes UP to SL (use current_sl which may be trailed)
            if highs[j] >= current_sl:
                exit_price = current_sl * (1 + slippage)
                exit_reason = "trail_SL" if trail_active else "SL"
                exit_idx = j
                break
            # TP hit: price goes DOWN to TP
            if lows[j] <= tp_price:
                exit_price = tp_price * (1 + slippage)
                exit_reason = "TP"
                exit_idx = j
                break

        if exit_price is None:
            exit_idx = min(i + max_hold, n - 1)
            exit_price = closes[exit_idx]

        # PnL for SELL: profit when price goes down
        gross_pnl_pct = (entry_price - exit_price) / entry_price
        net_pnl_pct = gross_pnl_pct - 2 * commission

        trade = {
            "entry_time": timestamps[i] if i < len(timestamps) else None,
            "exit_time": timestamps[exit_idx] if exit_idx < len(timestamps) else None,
            "direction": "SELL",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "exit_reason": exit_reason,
            "trail_activated": trail_active,
            "gross_pnl_pct": gross_pnl_pct * 100,
            "net_pnl_pct": net_pnl_pct * 100,
            "holding_bars": exit_idx - i,
            "confidence": confidence[i] if confidence is not None and i < len(confidence) else 0,
        }
        trades.append(trade)

        i = exit_idx + 1

    return pd.DataFrame(trades)


def run_sell_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    setups_df: pd.DataFrame,
    confidence: np.ndarray = None,
    model_name: str = "SELL",
) -> Dict:
    """
    Run the SELL-specific backtesting pipeline.
    """
    print(f"\n{'=' * 60}")
    print(f"  {model_name} — SELL Backtesting Results")
    print(f"{'=' * 60}")

    trades_df = simulate_sell_trades(df, signals, setups_df, confidence)

    if trades_df.empty:
        print("  No trades generated. Adjust signal thresholds.")
        return {"trades_df": trades_df, "equity_df": pd.DataFrame(), "metrics": {}}

    equity_df = compute_equity_curve(trades_df)
    metrics = compute_backtest_metrics(trades_df)

    print(f"  Total trades:        {metrics['total_trades']:,}")
    print(f"  Win rate:            {metrics['win_rate']:.2%}")
    print(f"  Avg return/trade:    {metrics['avg_return_pct']:.3f}%")
    print(f"  Total return:        {metrics['total_return_pct']:.2f}%")
    print(f"  Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"  Sharpe ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"  Exit reasons:        {metrics['exit_reason_counts']}")
    print(f"  Avg holding bars:    {metrics['avg_holding_bars']:.1f}")
    print(f"{'=' * 60}\n")

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, f"backtest_report_{model_name.lower()}.txt")
    with open(report_path, "w") as f:
        f.write(f"tradIA Backtest Report - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        for k, v in metrics.items():
            if k != "exit_reason_counts":
                f.write(f"{k:25s}: {v}\n")
            else:
                f.write(f"{'exit_reasons':25s}: {v}\n")
    print(f"[Backtest] Report saved to {report_path}")

    trades_path = os.path.join(RESULTS_DIR, f"trades_{model_name.lower()}.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"[Backtest] Trade log saved to {trades_path}")

    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "metrics": metrics,
    }


def simulate_buy_trades(
    df: pd.DataFrame,
    signals: np.ndarray,
    setups_df: pd.DataFrame,
    confidence: np.ndarray = None,
    commission: float = BACKTEST_COMMISSION,
    slippage: float = BACKTEST_SLIPPAGE,
    trailing_stop: bool = True,
) -> pd.DataFrame:
    """
    Simulate BUY trades using strategy-specific entry/SL/TP from setups_df.
    Mirror of simulate_sell_trades for LONG positions.

    Trailing stop: once price moves toward TP, SL tightens to trail highest high.
    """
    from src.config import (
        BUY_TRAILING_STOP, BUY_TRAILING_ACTIVATION, BUY_TRAILING_DISTANCE,
    )

    if trailing_stop is True:
        trailing_stop = BUY_TRAILING_STOP
    trailing_activation = BUY_TRAILING_ACTIVATION
    trailing_distance_pct = BUY_TRAILING_DISTANCE

    labeled = df.dropna(subset=["label"])
    n = min(len(labeled), len(signals))

    highs = labeled["high"].values
    lows = labeled["low"].values
    closes = labeled["close"].values
    timestamps = labeled.index

    # Build a lookup from labeled index → setup info
    setup_lookup = {}
    if not setups_df.empty:
        for _, setup in setups_df.iterrows():
            ts = setup["timestamp"]
            if ts in labeled.index:
                loc = labeled.index.get_loc(ts)
                if isinstance(loc, int) or isinstance(loc, np.integer):
                    setup_lookup[int(loc)] = setup

    trades = []
    i = 0
    while i < n - 1:
        if signals[i] != 1:
            i += 1
            continue

        if i not in setup_lookup:
            i += 1
            continue

        setup = setup_lookup[i]
        entry_price = setup["entry_price"] * (1 + slippage)
        sl_price = setup["sl_price"]
        tp_price = setup["tp_price"]
        sl_distance = entry_price - sl_price

        # Trailing stop state
        trail_active = False
        current_sl = sl_price
        best_price = entry_price  # Track highest price for trailing

        exit_price = None
        exit_reason = "time"
        exit_idx = i + 1
        max_hold = 96

        for j in range(i + 1, min(i + max_hold + 1, n)):
            # Update best (highest) price for BUY trailing
            if highs[j] > best_price:
                best_price = highs[j]

            # Check if trailing stop should activate
            if trailing_stop and not trail_active:
                price_moved = best_price - entry_price
                activation_target = trailing_activation * (tp_price - entry_price)
                if price_moved >= activation_target:
                    trail_active = True
                    trail_dist = trailing_distance_pct * sl_distance
                    current_sl = best_price - trail_dist

            # Update trailing SL if active
            if trail_active:
                new_sl = best_price - trailing_distance_pct * sl_distance
                if new_sl > current_sl:
                    current_sl = new_sl

            # SL hit: price goes DOWN to SL
            if lows[j] <= current_sl:
                exit_price = current_sl * (1 - slippage)
                exit_reason = "trail_SL" if trail_active else "SL"
                exit_idx = j
                break
            # TP hit: price goes UP to TP
            if highs[j] >= tp_price:
                exit_price = tp_price * (1 - slippage)
                exit_reason = "TP"
                exit_idx = j
                break

        if exit_price is None:
            exit_idx = min(i + max_hold, n - 1)
            exit_price = closes[exit_idx]

        # PnL for BUY: profit when price goes up
        gross_pnl_pct = (exit_price - entry_price) / entry_price
        net_pnl_pct = gross_pnl_pct - 2 * commission

        trade = {
            "entry_time": timestamps[i] if i < len(timestamps) else None,
            "exit_time": timestamps[exit_idx] if exit_idx < len(timestamps) else None,
            "direction": "BUY",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "exit_reason": exit_reason,
            "trail_activated": trail_active,
            "gross_pnl_pct": gross_pnl_pct * 100,
            "net_pnl_pct": net_pnl_pct * 100,
            "holding_bars": exit_idx - i,
            "confidence": confidence[i] if confidence is not None and i < len(confidence) else 0,
        }
        trades.append(trade)

        i = exit_idx + 1

    return pd.DataFrame(trades)


def run_buy_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    setups_df: pd.DataFrame,
    confidence: np.ndarray = None,
    model_name: str = "BUY",
) -> Dict:
    """
    Run the BUY-specific backtesting pipeline.
    """
    print(f"\n{'=' * 60}")
    print(f"  {model_name} — BUY Backtesting Results")
    print(f"{'=' * 60}")

    trades_df = simulate_buy_trades(df, signals, setups_df, confidence)

    if trades_df.empty:
        print("  No trades generated. Adjust signal thresholds.")
        return {"trades_df": trades_df, "equity_df": pd.DataFrame(), "metrics": {}}

    equity_df = compute_equity_curve(trades_df)
    metrics = compute_backtest_metrics(trades_df)

    print(f"  Total trades:        {metrics['total_trades']:,}")
    print(f"  Win rate:            {metrics['win_rate']:.2%}")
    print(f"  Avg return/trade:    {metrics['avg_return_pct']:.3f}%")
    print(f"  Total return:        {metrics['total_return_pct']:.2f}%")
    print(f"  Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"  Sharpe ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"  Exit reasons:        {metrics['exit_reason_counts']}")
    print(f"  Avg holding bars:    {metrics['avg_holding_bars']:.1f}")
    print(f"{'=' * 60}\n")

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, f"backtest_report_{model_name.lower()}.txt")
    with open(report_path, "w") as f:
        f.write(f"tradIA Backtest Report - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        for k, v in metrics.items():
            if k != "exit_reason_counts":
                f.write(f"{k:25s}: {v}\n")
            else:
                f.write(f"{'exit_reasons':25s}: {v}\n")
    print(f"[Backtest] Report saved to {report_path}")

    trades_path = os.path.join(RESULTS_DIR, f"trades_{model_name.lower()}.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"[Backtest] Trade log saved to {trades_path}")

    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "metrics": metrics,
    }

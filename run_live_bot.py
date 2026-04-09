"""
tradIA Live Trading Bot — Main Entry Point
Runs 24/7, synced to 15-minute candle closes.

Usage:
    python run_live_bot.py                 # uses settings from .env
    python run_live_bot.py --dry-run       # force dry-run mode
    python run_live_bot.py --once          # run one iteration and exit

Environment:
    Copy .env.example to .env and fill in your Binance API credentials.
"""
import os
import sys
import time
import signal
import argparse
from datetime import datetime, timezone

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main(dry_run: bool = None, run_once: bool = False):
    """
    Main trading bot loop.

    Args:
        dry_run: override DRY_RUN from environment
        run_once: execute one cycle and exit (for testing)
    """
    # ── Import after path setup ──
    from live.config_live import (
        SYMBOL, TIMEFRAME, TIMEFRAME_MINUTES, LEVERAGE,
        DRY_RUN, BUY_THRESHOLD, SELL_THRESHOLD,
        CANDLE_FETCH_DELAY, SESSION_FILTER_ENABLED,
        BINANCE_TESTNET,
    )
    from live.logger import log
    from live.binance_client import BinanceClient
    from live.data_pipeline import LiveDataPipeline
    from live.signal_engine import SignalEngine, Signal
    from live.order_manager import OrderManager
    from live.risk_manager import RiskManager
    from live.state_manager import StateManager

    # Override dry_run if specified via CLI
    if dry_run is not None:
        import live.config_live as cfg
        cfg.DRY_RUN = dry_run
        # Re-import for local reference
        actual_dry_run = dry_run
    else:
        actual_dry_run = DRY_RUN

    # ── Graceful shutdown handler ──
    shutdown_requested = [False]

    def shutdown_handler(signum, frame):
        log.warning(f"Shutdown signal received ({signum}). Finishing current cycle...")
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # ══════════════════════════════════════════
    # STARTUP
    # ══════════════════════════════════════════
    log.info("=" * 60)
    log.info("  tradIA Live Trading Bot")
    log.info("=" * 60)
    env_mode = "TESTNET" if BINANCE_TESTNET else "LIVE"
    log.info(f"  Environment:  {env_mode}")
    log.info(f"  Symbol:       {SYMBOL}")
    log.info(f"  Timeframe:    {TIMEFRAME}")
    log.info(f"  Leverage:     {LEVERAGE}x")
    log.info(f"  DRY RUN:      {actual_dry_run}")
    log.info(f"  BUY thresh:   {BUY_THRESHOLD}")
    log.info(f"  SELL thresh:  {SELL_THRESHOLD}")
    log.info(f"  Session:      {'ON' if SESSION_FILTER_ENABLED else 'OFF'}")
    log.info("=" * 60)

    if actual_dry_run:
        log.info(">>> RUNNING IN DRY-RUN MODE — no real orders will be placed <<<")

    # ── Initialize components ──
    try:
        # 1. Binance Client
        log.info("[1/5] Connecting to Binance...")
        client = BinanceClient()
        client.connect()

        balance = client.get_balance()
        total_balance = client.get_total_balance()
        log.info(f"  Available balance: ${balance:.2f} USDT")
        log.info(f"  Total balance:    ${total_balance:.2f} USDT")

        # 2. Signal Engine (load models)
        log.info("[2/5] Loading trading models...")
        signal_engine = SignalEngine()
        signal_engine.load_models()

        # 3. Data Pipeline (warmup with historical candles)
        log.info("[3/5] Initializing data pipeline...")
        pipeline = LiveDataPipeline(client)
        pipeline.initialize()

        # 4. Risk Manager
        log.info("[4/5] Initializing risk manager...")
        risk_mgr = RiskManager()

        # 5. Order Manager
        log.info("[5/5] Initializing order manager...")
        order_mgr = OrderManager(client, risk_mgr)

        # State Manager
        state_mgr = StateManager()

        # Restore state if exists
        saved_state = state_mgr.load()
        if saved_state:
            if saved_state.get("position"):
                log.info("Restoring saved position state...")
                order_mgr.position = saved_state["position"]
                if saved_state.get("trailing_state"):
                    ts = saved_state["trailing_state"]
                    order_mgr.trail_active = ts.get("trail_active", False)
                    order_mgr.best_price = ts.get("best_price", 0)
                    order_mgr.current_sl = ts.get("current_sl", 0)
                    order_mgr.original_sl = ts.get("original_sl", 0)
                    order_mgr.tp_price = ts.get("tp_price", 0)
                    order_mgr.entry_price = ts.get("entry_price", 0)
                    order_mgr.sl_distance = ts.get("sl_distance", 0)
                    order_mgr.sl_placed = ts.get("sl_placed", False)
                    order_mgr.tp_placed = ts.get("tp_placed", False)
                log.info(
                    f"  Restored position: {order_mgr.position['direction']} "
                    f"@ ${order_mgr.entry_price:.2f}"
                )
            if saved_state.get("daily_pnl"):
                risk_mgr.daily_pnl = saved_state["daily_pnl"]
                risk_mgr.daily_trades = saved_state.get("daily_trades", 0)

        # Check for existing position on Binance
        existing_pos = client.get_open_position()
        if existing_pos and not order_mgr.has_position:
            log.warning(
                f"Found existing Binance position not tracked by bot: "
                f"{existing_pos['side']} {existing_pos['quantity']} "
                f"@ ${existing_pos['entry_price']:.2f}"
            )
            log.info("Adopting untracked position into bot state...")
            direction = existing_pos["side"]   # 'LONG' or 'SHORT'
            adopt_side = "BUY" if direction == "LONG" else "SELL"
            adopt_close = "SELL" if direction == "LONG" else "BUY"
            order_mgr.position = {
                "direction": direction,
                "side": adopt_side,
                "close_side": adopt_close,
                "quantity": existing_pos["quantity"],
                "entry_price": existing_pos["entry_price"],
                "entry_time": time.time(),
                "order_id": "ADOPTED",
            }
            order_mgr.entry_price = existing_pos["entry_price"]
            order_mgr.best_price = existing_pos["entry_price"]
            log.info(
                f"Adopted {direction} position: "
                f"qty={existing_pos['quantity']}, "
                f"entry=${existing_pos['entry_price']:.2f}"
            )
            log.warning(
                "Adopted position has no SL/TP — trailing stop logic "
                "will attempt to place protection on the next cycle."
            )

    except Exception as e:
        log.error(f"Startup failed: {e}")
        raise

    log.info("")
    log.info("Startup complete! Entering main trading loop...")
    log.info("")

    # ══════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════
    cycle_count = 0

    while not shutdown_requested[0]:
        try:
            cycle_count += 1

            # ── Wait for next candle close ──
            if cycle_count > 1 and not run_once:
                wait_seconds = _seconds_until_next_candle(TIMEFRAME_MINUTES)
                wait_seconds += CANDLE_FETCH_DELAY  # small delay to ensure candle is closed

                if wait_seconds > 0:
                    next_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    log.info(
                        f"Waiting {wait_seconds:.0f}s for next candle close "
                        f"(~{wait_seconds/60:.1f} min from {next_time} UTC)..."
                    )
                    # Sleep in small chunks so we can respond to shutdown
                    for _ in range(int(wait_seconds)):
                        if shutdown_requested[0]:
                            break
                        time.sleep(1)

                if shutdown_requested[0]:
                    break

            now = datetime.now(timezone.utc)
            log.info(f"─── Cycle {cycle_count} @ {now:%Y-%m-%d %H:%M:%S} UTC ───")

            # ── 1. Fetch latest candle ──
            features, is_new = pipeline.update()

            if not is_new and cycle_count > 1:
                log.debug("No new candle yet, skipping cycle")
                if run_once:
                    break
                continue

            if features is None:
                log.warning("No features available, skipping cycle")
                if run_once:
                    break
                continue

            current_price = pipeline.get_latest_close()
            atr = pipeline.get_latest_atr()
            log.info(f"  Price: ${current_price:.2f} | ATR(14): ${atr:.2f}")

            # ── 2. Check existing position ──
            if order_mgr.has_position:
                # Check if position was closed (SL/TP hit on Binance)
                trade_result = order_mgr.check_position_status()

                if trade_result:
                    # Position was closed
                    log.info(
                        f"  POSITION CLOSED: "
                        f"{trade_result['direction']} | PnL: ${trade_result.get('pnl', 0):.4f}"
                    )
                    state_mgr.save(
                        position=None,
                        daily_pnl=risk_mgr.daily_pnl,
                        daily_trades=risk_mgr.daily_trades,
                        last_candle_time=str(pipeline.last_candle_time),
                    )
                else:
                    # Position still open — ensure SL/TP are placed, then update trailing stop
                    order_mgr.ensure_sl_tp()

                    latest_high = float(features.get("high", current_price))
                    latest_low = float(features.get("low", current_price))
                    order_mgr.update_trailing_stop(latest_high, latest_low)

                    pos = order_mgr.get_position_summary()
                    log.info(
                        f"  Position open: {pos['direction']} @ ${pos['entry_price']:.2f} | "
                        f"SL: ${pos['current_sl']:.2f} | TP: ${pos['tp_price']:.2f} | "
                        f"Trail: {'YES' if pos['trail_active'] else 'NO'}"
                    )

                    # Save state
                    state_mgr.save(
                        position=order_mgr.position,
                        daily_pnl=risk_mgr.daily_pnl,
                        daily_trades=risk_mgr.daily_trades,
                        last_candle_time=str(pipeline.last_candle_time),
                        trailing_state={
                            "trail_active": order_mgr.trail_active,
                            "best_price": order_mgr.best_price,
                            "current_sl": order_mgr.current_sl,
                            "original_sl": order_mgr.original_sl,
                            "tp_price": order_mgr.tp_price,
                            "entry_price": order_mgr.entry_price,
                            "sl_distance": order_mgr.sl_distance,
                            "sl_placed": order_mgr.sl_placed,
                            "tp_placed": order_mgr.tp_placed,
                        },
                    )

            # ── 3. Generate signal (only if no position) ──
            if not order_mgr.has_position:
                # Check daily loss limit
                balance = client.get_balance() if not actual_dry_run else client.get_total_balance()

                if not risk_mgr.check_daily_loss_limit(balance):
                    log.warning("Daily loss limit reached — skipping signal generation")
                    if run_once:
                        break
                    continue

                # Generate signal
                candle_time = pipeline.last_candle_time
                sig, details = signal_engine.generate_signal(features, candle_time)

                log.info(
                    f"  Signal: {sig.value} | "
                    f"BUY={details['buy_proba']:.3f} | SELL={details['sell_proba']:.3f} | "
                    f"{details['reason']}"
                )

                # Execute trade if we have a signal
                if sig in (Signal.LONG, Signal.SHORT):
                    # Hard guard: verify no position exists on Binance
                    if not actual_dry_run:
                        binance_pos = client.get_open_position()
                        if binance_pos:
                            log.warning(
                                f"BLOCKED: Binance already has a "
                                f"{binance_pos['side']} position — "
                                f"skipping new trade to avoid stacking"
                            )
                            if run_once:
                                break
                            continue

                    log.info(f"  >>> EXECUTING {sig.value} TRADE <<<")

                    success = order_mgr.open_trade(
                        signal=sig,
                        atr=atr,
                        current_price=current_price,
                        balance=balance,
                    )

                    if success:
                        # Save state immediately
                        state_mgr.save(
                            position=order_mgr.position,
                            daily_pnl=risk_mgr.daily_pnl,
                            daily_trades=risk_mgr.daily_trades,
                            last_candle_time=str(pipeline.last_candle_time),
                            trailing_state={
                                "trail_active": order_mgr.trail_active,
                                "best_price": order_mgr.best_price,
                                "current_sl": order_mgr.current_sl,
                                "original_sl": order_mgr.original_sl,
                                "tp_price": order_mgr.tp_price,
                                "entry_price": order_mgr.entry_price,
                                "sl_distance": order_mgr.sl_distance,
                                "sl_placed": order_mgr.sl_placed,
                                "tp_placed": order_mgr.tp_placed,
                            },
                        )
                    else:
                        log.warning("Trade execution failed")

            # Save state
            state_mgr.save(
                position=order_mgr.get_position_summary(),
                daily_pnl=risk_mgr.daily_pnl,
                daily_trades=risk_mgr.daily_trades,
                last_candle_time=str(pipeline.last_candle_time),
                trailing_state={
                    "trail_active": order_mgr.trail_active,
                    "best_price": order_mgr.best_price,
                    "current_sl": order_mgr.current_sl,
                    "original_sl": order_mgr.original_sl,
                    "tp_price": order_mgr.tp_price,
                    "entry_price": order_mgr.entry_price,
                    "sl_distance": order_mgr.sl_distance,
                    "sl_placed": order_mgr.sl_placed,
                    "tp_placed": order_mgr.tp_placed,
                } if order_mgr.has_position else None,
            )

            if run_once:
                log.info("Single cycle complete (--once mode)")
                break

        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
            shutdown_requested[0] = True

        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)
            # Don't crash on errors — wait and retry
            if run_once:
                break
            log.info("Waiting 60s before retrying...")
            time.sleep(60)

    # ══════════════════════════════════════════
    # SHUTDOWN
    # ══════════════════════════════════════════
    log.info("")
    log.info("=" * 60)
    log.info("  Shutting down tradIA Live Bot...")
    log.info("=" * 60)

    # Save final state
    state_mgr.save(
        position=order_mgr.get_position_summary(),
        daily_pnl=risk_mgr.daily_pnl,
        daily_trades=risk_mgr.daily_trades,
        last_candle_time=str(pipeline.last_candle_time) if pipeline.last_candle_time else "",
        trailing_state={
            "trail_active": order_mgr.trail_active,
            "best_price": order_mgr.best_price,
            "current_sl": order_mgr.current_sl,
            "original_sl": order_mgr.original_sl,
            "tp_price": order_mgr.tp_price,
            "entry_price": order_mgr.entry_price,
            "sl_distance": order_mgr.sl_distance,
            "sl_placed": order_mgr.sl_placed,
            "tp_placed": order_mgr.tp_placed,
        } if order_mgr.has_position else None,
    )

    if order_mgr.has_position:
        log.warning(
            f"Bot stopped with open position: "
            f"{order_mgr.position['direction']} @ ${order_mgr.entry_price:.2f}"
        )
        log.warning("Position will be managed by SL/TP orders on Binance.")
        log.warning("State saved — position will resume on restart.")

    daily = risk_mgr.get_daily_summary()
    log.info(f"  Daily PnL:    ${daily['daily_pnl']:.4f}")
    log.info(f"  Daily trades: {daily['daily_trades']}")
    log.info("  Goodbye!")
    log.info("=" * 60)


def _seconds_until_next_candle(timeframe_minutes: int) -> float:
    """Calculate seconds until the next candle close."""
    now = datetime.now(timezone.utc)
    current_minute = now.minute
    current_second = now.second

    # Find next multiple of timeframe_minutes
    minutes_into_candle = current_minute % timeframe_minutes
    minutes_remaining = timeframe_minutes - minutes_into_candle

    if minutes_remaining == timeframe_minutes and current_second == 0:
        # We're exactly at a candle boundary
        return 0

    seconds_remaining = (minutes_remaining * 60) - current_second
    return max(seconds_remaining, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tradIA Live Trading Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Run in dry-run mode (no real orders)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (real orders — use with caution!)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (for testing)",
    )

    args = parser.parse_args()

    # Determine dry_run mode
    if args.live:
        dry_run_override = False
    elif args.dry_run:
        dry_run_override = True
    else:
        dry_run_override = None  # use .env setting

    main(dry_run=dry_run_override, run_once=args.once)

"""
tradIA Live Trading — Order Manager
Handles order execution, SL/TP placement, trailing stops,
and position lifecycle management.
"""
import time
from typing import Optional, Dict

from live.binance_client import BinanceClient
from live.risk_manager import RiskManager
from live.signal_engine import Signal
from live.config_live import (
    BACKTEST_SL_ATR_MULT, BACKTEST_TP_ATR_MULT,
    SELL_TP_RATIO, BUY_TP_RATIO,
    SELL_TRAILING_STOP, SELL_TRAILING_ACTIVATION, SELL_TRAILING_DISTANCE,
    BUY_TRAILING_STOP, BUY_TRAILING_ACTIVATION, BUY_TRAILING_DISTANCE,
    DRY_RUN, LEVERAGE,
)
from live.logger import log


class OrderManager:
    """
    Manages the full lifecycle of a trade:
    entry → SL/TP placement → trailing stop updates → exit detection.
    """

    def __init__(self, client: BinanceClient, risk_manager: RiskManager):
        self.client = client
        self.risk = risk_manager

        # Current position state
        self.position: Optional[Dict] = None       # active position details
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None

        # Trailing stop state
        self.trail_active: bool = False
        self.best_price: float = 0.0      # best price since entry (for trailing)
        self.current_sl: float = 0.0       # current SL price (may be trailed)
        self.original_sl: float = 0.0      # original SL price
        self.tp_price: float = 0.0
        self.entry_price: float = 0.0
        self.sl_distance: float = 0.0      # absolute SL distance

    @property
    def has_position(self) -> bool:
        """Check if we have an open position."""
        return self.position is not None

    def open_trade(
        self,
        signal: Signal,
        atr: float,
        current_price: float,
        balance: float,
    ) -> bool:
        """
        Open a new trade based on the signal.

        Args:
            signal: LONG or SHORT
            atr: current ATR(14) for SL/TP calculation
            current_price: current market price
            balance: available USDT balance

        Returns:
            True if trade was opened successfully
        """
        if self.has_position:
            log.warning("Cannot open trade — position already open")
            return False

        if atr <= 0:
            log.warning(f"Invalid ATR={atr}, cannot calculate SL/TP")
            return False

        # ── Calculate SL and TP ──
        if signal == Signal.LONG:
            sl_distance = BACKTEST_SL_ATR_MULT * atr
            tp_distance = sl_distance * BUY_TP_RATIO
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
            order_side = "BUY"
            close_side = "SELL"
            direction = "LONG"

        elif signal == Signal.SHORT:
            sl_distance = BACKTEST_SL_ATR_MULT * atr
            tp_distance = sl_distance * SELL_TP_RATIO
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance
            order_side = "SELL"
            close_side = "BUY"
            direction = "SHORT"

        else:
            log.error(f"Invalid signal for trade: {signal}")
            return False

        # ── Position sizing ──
        quantity = self.risk.calculate_position_size(
            balance=balance,
            current_price=current_price,
            sl_distance=sl_distance,
            leverage=LEVERAGE,
        )

        if quantity <= 0:
            log.warning("Position size is 0 — skipping trade")
            return False

        # ── Validate trade ──
        if not self.risk.validate_trade(balance, quantity, current_price):
            log.warning("Trade validation failed — skipping")
            return False

        # ── Execute entry ──
        log.info(
            f"{'='*50}\n"
            f"  OPENING {direction} TRADE\n"
            f"  Price:    ${current_price:.2f}\n"
            f"  Quantity: {quantity:.6f}\n"
            f"  SL:       ${sl_price:.2f} (dist: ${sl_distance:.2f})\n"
            f"  TP:       ${tp_price:.2f} (dist: ${tp_distance:.2f})\n"
            f"  ATR(14):  ${atr:.2f}\n"
            f"  Risk:     ${quantity * sl_distance:.4f} USDT\n"
            f"{'='*50}"
        )

        entry_order = self.client.place_market_order(order_side, quantity)
        if entry_order is None:
            log.error("Entry order failed!")
            return False

        # Get actual fill price
        if entry_order.get("dry_run"):
            actual_entry = current_price
        else:
            actual_entry = float(entry_order.get("avgPrice", current_price))
            # Recalculate SL/TP based on actual fill
            if signal == Signal.LONG:
                sl_price = actual_entry - sl_distance
                tp_price = actual_entry + tp_distance
            else:
                sl_price = actual_entry + sl_distance
                tp_price = actual_entry - tp_distance

        # ── Place SL order ──
        sl_order = self.client.place_stop_loss(close_side, quantity, sl_price)

        # ── Place TP order ──
        tp_order = self.client.place_take_profit(close_side, quantity, tp_price)

        # ── Update state ──
        self.position = {
            "direction": direction,
            "side": order_side,
            "close_side": close_side,
            "quantity": quantity,
            "entry_price": actual_entry,
            "entry_time": time.time(),
            "order_id": entry_order.get("orderId"),
        }
        self.entry_price = actual_entry
        self.original_sl = sl_price
        self.current_sl = sl_price
        self.tp_price = tp_price
        self.sl_distance = sl_distance
        self.best_price = actual_entry
        self.trail_active = False

        self.sl_order_id = sl_order.get("orderId") if sl_order else None
        self.tp_order_id = tp_order.get("orderId") if tp_order else None

        log.info(f"{direction} trade opened @ ${actual_entry:.2f}")
        return True

    def update_trailing_stop(self, latest_high: float, latest_low: float) -> bool:
        """
        Update trailing stop based on latest candle high/low.
        Mirrors the exact trailing stop logic from the backtester.

        Returns True if SL was updated.
        """
        if not self.has_position:
            return False

        direction = self.position["direction"]
        updated = False

        if direction == "LONG":
            trailing_enabled = BUY_TRAILING_STOP
            trail_activation = BUY_TRAILING_ACTIVATION
            trail_distance_pct = BUY_TRAILING_DISTANCE

            # Track best (highest) price
            if latest_high > self.best_price:
                self.best_price = latest_high

            # Check if trailing should activate
            if trailing_enabled and not self.trail_active:
                price_moved = self.best_price - self.entry_price
                activation_target = trail_activation * (self.tp_price - self.entry_price)
                if price_moved >= activation_target:
                    self.trail_active = True
                    trail_dist = trail_distance_pct * self.sl_distance
                    self.current_sl = self.best_price - trail_dist
                    log.info(
                        f"TRAILING STOP ACTIVATED (LONG): "
                        f"best_price=${self.best_price:.2f}, "
                        f"new_sl=${self.current_sl:.2f}"
                    )
                    updated = True

            # Update trailing SL
            if self.trail_active:
                new_sl = self.best_price - trail_distance_pct * self.sl_distance
                if new_sl > self.current_sl:
                    old_sl = self.current_sl
                    self.current_sl = new_sl
                    log.info(
                        f"Trailing SL tightened (LONG): "
                        f"${old_sl:.2f} → ${new_sl:.2f}"
                    )
                    updated = True

        elif direction == "SHORT":
            trailing_enabled = SELL_TRAILING_STOP
            trail_activation = SELL_TRAILING_ACTIVATION
            trail_distance_pct = SELL_TRAILING_DISTANCE

            # Track best (lowest) price
            if latest_low < self.best_price:
                self.best_price = latest_low

            # Check if trailing should activate
            if trailing_enabled and not self.trail_active:
                price_moved = self.entry_price - self.best_price
                activation_target = trail_activation * (self.entry_price - self.tp_price)
                if price_moved >= activation_target:
                    self.trail_active = True
                    trail_dist = trail_distance_pct * self.sl_distance
                    self.current_sl = self.best_price + trail_dist
                    log.info(
                        f"TRAILING STOP ACTIVATED (SHORT): "
                        f"best_price=${self.best_price:.2f}, "
                        f"new_sl=${self.current_sl:.2f}"
                    )
                    updated = True

            # Update trailing SL
            if self.trail_active:
                new_sl = self.best_price + trail_distance_pct * self.sl_distance
                if new_sl < self.current_sl:
                    old_sl = self.current_sl
                    self.current_sl = new_sl
                    log.info(
                        f"Trailing SL tightened (SHORT): "
                        f"${old_sl:.2f} → ${new_sl:.2f}"
                    )
                    updated = True

        # Update SL order on Binance if changed
        if updated and not DRY_RUN:
            try:
                self.client.update_stop_loss(
                    self.position["close_side"],
                    self.position["quantity"],
                    self.current_sl,
                )
            except Exception as e:
                log.error(f"Failed to update SL order on Binance: {e}")

        return updated

    def check_position_status(self) -> Optional[Dict]:
        """
        Check if the current position has been closed (SL/TP hit).
        Returns trade result dict if closed, None if still open.
        """
        if not self.has_position:
            return None

        if DRY_RUN:
            # In dry run, we rely on trailing stop logic and manual checks
            return None

        # Check actual position on Binance
        binance_pos = self.client.get_open_position()

        if binance_pos is None:
            # Position was closed (SL or TP hit)
            log.info(f"Position closed! Detecting exit details...")

            # Calculate PnL
            pnl = self._calculate_closed_pnl()

            result = {
                "direction": self.position["direction"],
                "entry_price": self.entry_price,
                "sl_price": self.original_sl,
                "tp_price": self.tp_price,
                "trail_activated": self.trail_active,
                "best_price": self.best_price,
                "pnl": pnl,
            }

            # Record PnL
            self.risk.record_trade_pnl(pnl)

            # Reset state
            self._reset_position()

            log.info(
                f"Trade result: PnL=${pnl:.4f} | "
                f"Trail active: {result['trail_activated']}"
            )
            return result

        return None

    def force_close(self) -> Optional[Dict]:
        """Force close the current position (emergency / shutdown)."""
        if not self.has_position:
            return None

        log.warning("FORCE CLOSING position!")

        # Cancel all open orders first
        self.client.cancel_open_orders()

        # Place closing market order
        self.client.place_market_order(
            self.position["close_side"],
            self.position["quantity"],
        )

        result = {
            "direction": self.position["direction"],
            "entry_price": self.entry_price,
            "exit_reason": "force_close",
        }

        self._reset_position()
        return result

    def _calculate_closed_pnl(self) -> float:
        """Estimate PnL from the closed position."""
        try:
            # Try to get from Binance recent trades
            current_price = self.client.get_current_price()
            qty = self.position["quantity"]

            if self.position["direction"] == "LONG":
                pnl = (current_price - self.entry_price) * qty
            else:
                pnl = (self.entry_price - current_price) * qty

            return pnl
        except Exception:
            return 0.0

    def _reset_position(self):
        """Reset all position state."""
        self.position = None
        self.sl_order_id = None
        self.tp_order_id = None
        self.trail_active = False
        self.best_price = 0.0
        self.current_sl = 0.0
        self.original_sl = 0.0
        self.tp_price = 0.0
        self.entry_price = 0.0
        self.sl_distance = 0.0

    def get_position_summary(self) -> Optional[Dict]:
        """Get current position details for logging/state save."""
        if not self.has_position:
            return None

        return {
            "direction": self.position["direction"],
            "quantity": self.position["quantity"],
            "entry_price": self.entry_price,
            "current_sl": self.current_sl,
            "original_sl": self.original_sl,
            "tp_price": self.tp_price,
            "trail_active": self.trail_active,
            "best_price": self.best_price,
            "sl_distance": self.sl_distance,
            "entry_time": self.position.get("entry_time", 0),
        }

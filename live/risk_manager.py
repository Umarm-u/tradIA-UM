"""
tradIA Live Trading — Risk Manager
Position sizing, daily loss limits, and trade validation.
Designed for micro-accounts (~$10 USDT).
"""
import math
from datetime import datetime, timezone

from live.config_live import (
    RISK_PER_TRADE, MAX_DAILY_LOSS, MIN_TRADE_USDT, MIN_QUANTITY, LEVERAGE,
)
from live.logger import log


class RiskManager:
    """
    Manages risk per trade, daily loss tracking, and position sizing.

    For micro-accounts ($10), uses minimum possible position sizes
    while still enforcing risk management rules.
    """

    def __init__(self):
        self.daily_pnl: float = 0.0       # cumulative PnL today (USDT)
        self.daily_trades: int = 0
        self._last_reset_date: str = ""

    def calculate_position_size(
        self,
        balance: float,
        current_price: float,
        sl_distance: float,
        leverage: int = LEVERAGE,
    ) -> float:
        """
        Calculate position size based on risk per trade.

        For a $10 account with 2% risk = $0.20 max loss per trade.
        Position size = (risk_amount / sl_distance_pct) / price.

        Args:
            balance: available USDT balance
            current_price: current mark price of the asset
            sl_distance: absolute SL distance in price units
            leverage: leverage multiplier

        Returns:
            quantity: position size in base asset (e.g. BTC)
        """
        if sl_distance <= 0 or current_price <= 0 or balance <= 0:
            log.warning(
                f"Invalid inputs for position sizing: balance={balance}, "
                f"price={current_price}, sl_dist={sl_distance}"
            )
            return 0.0

        # Risk amount in USDT
        risk_amount = balance * RISK_PER_TRADE

        # Position size based on SL distance
        # If SL distance is $500 and risk is $0.20, we can open 0.20/500 = 0.0004 BTC
        quantity = risk_amount / sl_distance

        # Check minimum notional requirement (Binance Futures = $100)
        notional = quantity * current_price
        if notional < MIN_TRADE_USDT:
            # Bump to minimum notional
            quantity = MIN_TRADE_USDT / current_price
            actual_risk = quantity * sl_distance
            log.warning(
                f"Position size bumped to meet min notional (${MIN_TRADE_USDT}): "
                f"qty={quantity:.6f}, notional=${quantity * current_price:.2f}, "
                f"actual_risk=${actual_risk:.2f} (vs configured ${risk_amount:.2f})"
            )

        # Ensure minimum quantity
        if quantity < MIN_QUANTITY:
            quantity = MIN_QUANTITY
            log.info(f"Position size bumped to minimum quantity: {MIN_QUANTITY}")

        # Final notional check
        final_notional = quantity * current_price

        # Check margin requirement
        required_margin = final_notional / leverage
        if required_margin > balance * 0.90:  # leave 10% buffer
            max_notional = balance * 0.90 * leverage
            if max_notional < MIN_TRADE_USDT:
                log.error(
                    f"INSUFFICIENT BALANCE: need ${MIN_TRADE_USDT / leverage:.2f} margin "
                    f"for min notional ${MIN_TRADE_USDT}, have ${balance:.2f}. "
                    f"Deposit more funds or increase leverage."
                )
                return 0.0
            # Scale down to fit available margin
            max_qty = max_notional / current_price
            quantity = min(quantity, max_qty)
            log.warning(
                f"Position size limited by margin: "
                f"required=${required_margin:.2f}, available=${balance:.2f}, "
                f"adjusted qty={quantity:.6f}"
            )

        log.info(
            f"Position size calculated: qty={quantity:.6f}, "
            f"risk=${risk_amount:.2f}, SL dist=${sl_distance:.2f}, "
            f"notional=${quantity * current_price:.2f}"
        )

        return quantity

    def check_daily_loss_limit(self, balance: float) -> bool:
        """
        Check if daily loss limit has been reached.

        Returns:
            True if we can still trade, False if limit exceeded.
        """
        self._check_daily_reset()

        max_loss = balance * MAX_DAILY_LOSS
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= max_loss:
            log.warning(
                f"DAILY LOSS LIMIT REACHED: PnL=${self.daily_pnl:.2f}, "
                f"limit=-${max_loss:.2f}. Trading stopped for today."
            )
            return False
        return True

    def record_trade_pnl(self, pnl: float):
        """Record a trade's PnL for daily tracking."""
        self._check_daily_reset()
        self.daily_pnl += pnl
        self.daily_trades += 1
        log.info(
            f"Trade PnL: ${pnl:.4f} | Daily PnL: ${self.daily_pnl:.4f} | "
            f"Daily trades: {self.daily_trades}"
        )

    def validate_trade(self, balance: float, quantity: float, price: float) -> bool:
        """
        Validate that a trade meets all requirements.
        Returns True if the trade is valid.
        """
        notional = quantity * price

        if notional < MIN_TRADE_USDT:
            log.warning(f"Trade below minimum notional: ${notional:.2f} < ${MIN_TRADE_USDT}")
            return False

        required_margin = notional / LEVERAGE
        if required_margin > balance:
            log.warning(
                f"Insufficient margin: required=${required_margin:.2f}, "
                f"available=${balance:.2f}"
            )
            return False

        if not self.check_daily_loss_limit(balance):
            return False

        return True

    def _check_daily_reset(self):
        """Reset daily counters at UTC midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            if self._last_reset_date:  # don't log on first call
                log.info(
                    f"Daily reset: previous day PnL=${self.daily_pnl:.4f}, "
                    f"trades={self.daily_trades}"
                )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self._last_reset_date = today

    def get_daily_summary(self) -> dict:
        """Return current daily trading summary."""
        return {
            "date": self._last_reset_date,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
        }

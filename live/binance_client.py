"""
tradIA Live Trading — Binance Futures API Client
Wraps python-binance for USDT-M Futures operations.
Includes automatic retry with exponential backoff.
"""
import time
import math
from typing import Optional, Dict, List
from decimal import Decimal, ROUND_DOWN

import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceRequestException

from live.config_live import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    SYMBOL, TIMEFRAME, LEVERAGE,
    API_RETRY_ATTEMPTS, API_RETRY_DELAY, API_TIMEOUT,
    DRY_RUN,
)
from live.logger import log


class BinanceClient:
    """Binance USDT-M Futures API wrapper with retry logic."""

    def __init__(self):
        self.client: Optional[Client] = None
        self.symbol = SYMBOL
        self.symbol_info: Dict = {}
        self.price_precision: int = 2
        self.qty_precision: int = 3
        self.tick_size: float = 0.10
        self.min_qty: float = 0.001
        self.min_notional: float = 5.0

    # ──────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────

    def connect(self):
        """Establish connection to Binance Futures."""
        log.info("Connecting to Binance Futures API...")

        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise ValueError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file"
            )

        self.client = Client(
            BINANCE_API_KEY,
            BINANCE_API_SECRET,
            requests_params={"timeout": API_TIMEOUT},
        )

        # Test connectivity
        server_time = self.client.get_server_time()
        log.info(f"Connected to Binance. Server time: {server_time['serverTime']}")

        # Fetch symbol info for precision
        self._load_symbol_info()

        # Set leverage
        self._set_leverage()

        # Set margin type to ISOLATED for better risk control
        self._set_margin_type()

        log.info(
            f"Symbol: {self.symbol} | Leverage: {LEVERAGE}x | "
            f"Price precision: {self.price_precision} | Qty precision: {self.qty_precision}"
        )

    def _load_symbol_info(self):
        """Load trading rules for the symbol (precision, min qty, etc.)."""
        info = self.client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == self.symbol:
                self.symbol_info = s
                self.price_precision = int(s["pricePrecision"])
                self.qty_precision = int(s["quantityPrecision"])

                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        self.tick_size = float(f["tickSize"])
                    if f["filterType"] == "LOT_SIZE":
                        self.min_qty = float(f["minQty"])
                    if f["filterType"] == "MIN_NOTIONAL":
                        self.min_notional = float(f.get("notional", 5.0))
                break

        log.debug(
            f"Symbol info loaded: tick_size={self.tick_size}, "
            f"min_qty={self.min_qty}, min_notional={self.min_notional}"
        )

    def _set_leverage(self):
        """Set leverage for the trading pair."""
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=LEVERAGE
            )
            log.info(f"Leverage set to {LEVERAGE}x")
        except BinanceAPIException as e:
            if e.code == -4028:  # leverage not changed
                log.debug(f"Leverage already at {LEVERAGE}x")
            else:
                raise

    def _set_margin_type(self):
        """Set margin type to ISOLATED."""
        try:
            self.client.futures_change_margin_type(
                symbol=self.symbol, marginType="ISOLATED"
            )
            log.info("Margin type set to ISOLATED")
        except BinanceAPIException as e:
            if e.code == -4046:  # already set
                log.debug("Margin type already ISOLATED")
            else:
                raise

    # ──────────────────────────────────────────
    # Market Data
    # ──────────────────────────────────────────

    def fetch_klines(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch historical klines/candles from Binance Futures.
        Returns DataFrame with columns matching training data format.
        """
        klines = self._retry(
            self.client.futures_klines,
            symbol=self.symbol,
            interval=TIMEFRAME,
            limit=limit,
        )

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Set datetime index (matching training data format)
        df.set_index("open_time", inplace=True)
        df.index.name = "open_time"

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]].copy()

        # Drop the last candle if it's still forming (not closed)
        # We check if the last candle's close time is in the future
        if len(df) > 1:
            now_ms = int(time.time() * 1000)
            last_close_time = int(klines[-1][6])
            if last_close_time > now_ms:
                df = df.iloc[:-1]

        log.debug(f"Fetched {len(df)} candles: {df.index[0]} → {df.index[-1]}")
        return df

    def get_current_price(self) -> float:
        """Get current mark price."""
        ticker = self._retry(
            self.client.futures_mark_price, symbol=self.symbol
        )
        return float(ticker["markPrice"])

    # ──────────────────────────────────────────
    # Account Info
    # ──────────────────────────────────────────

    def get_balance(self) -> float:
        """Get available USDT balance for futures trading."""
        account = self._retry(self.client.futures_account)
        for asset in account["assets"]:
            if asset["asset"] == "USDT":
                return float(asset["availableBalance"])
        return 0.0

    def get_total_balance(self) -> float:
        """Get total wallet balance (including in positions)."""
        account = self._retry(self.client.futures_account)
        for asset in account["assets"]:
            if asset["asset"] == "USDT":
                return float(asset["walletBalance"])
        return 0.0

    def get_open_position(self) -> Optional[Dict]:
        """
        Get the current open position for our symbol.
        Returns None if no position.
        """
        positions = self._retry(
            self.client.futures_position_information,
            symbol=self.symbol
        )

        for pos in positions:
            try:
                qty = float(pos.get("positionAmt", 0))

                if qty != 0:
                    return {
                        "side": "LONG" if qty > 0 else "SHORT",
                        "quantity": abs(qty),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                        "leverage": int(pos.get("leverage", 0)),  # SAFE
                        "mark_price": float(pos.get("markPrice", 0)),
                    }

            except Exception as e:
                log.warning(f"Error parsing position: {e} | raw: {pos}")

        return None

    # ──────────────────────────────────────────
    # Order Execution
    # ──────────────────────────────────────────

    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """
        Place a market order on Binance Futures.

        Args:
            side: 'BUY' or 'SELL'
            quantity: order quantity in base asset (e.g. BTC)
        """
        price = self.get_current_price()
        quantity = self._round_qty(quantity, price=price)

        if DRY_RUN:
            log.info(
                f"[DRY RUN] Market {side} {quantity} {self.symbol} @ ~{price:.2f}"
            )
            return {
                "orderId": "DRY_RUN",
                "side": side,
                "quantity": quantity,
                "price": price,
                "status": "FILLED",
                "dry_run": True,
            }

        order = self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
        )

        avg_price = float(order.get("avgPrice", 0))
        log.info(
            f"Market {side} {quantity} {self.symbol} filled @ {avg_price:.2f} "
            f"(orderId: {order['orderId']})"
        )
        return order

    def place_stop_loss(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """
        Place a stop-market order (SL).
        side should be the CLOSING side (SELL for longs, BUY for shorts).

        Uses the direct REST endpoint to avoid python-binance routing
        STOP_MARKET orders to the algoOrder endpoint.
        """
        quantity = self._round_qty(quantity)
        stop_price = self._round_price(stop_price)

        if DRY_RUN:
            log.info(
                f"[DRY RUN] Stop-Market {side} {quantity} {self.symbol} @ {stop_price}"
            )
            return {"orderId": "DRY_RUN_SL", "dry_run": True}

        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "quantity": quantity,
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
        }
        order = self._retry(
            self.client._request_futures_api,
            "post", "order", True, data=params,
        )

        log.info(f"SL order placed: {side} {quantity} @ {stop_price} (orderId: {order['orderId']})")
        return order

    def place_take_profit(self, side: str, quantity: float, tp_price: float) -> Optional[Dict]:
        """
        Place a take-profit-market order.
        side should be the CLOSING side.

        Uses the direct REST endpoint to avoid python-binance routing
        TAKE_PROFIT_MARKET orders to the algoOrder endpoint.
        """
        quantity = self._round_qty(quantity)
        tp_price = self._round_price(tp_price)

        if DRY_RUN:
            log.info(
                f"[DRY RUN] TP-Market {side} {quantity} {self.symbol} @ {tp_price}"
            )
            return {"orderId": "DRY_RUN_TP", "dry_run": True}

        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_price,
            "quantity": quantity,
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
        }
        order = self._retry(
            self.client._request_futures_api,
            "post", "order", True, data=params,
        )

        log.info(f"TP order placed: {side} {quantity} @ {tp_price} (orderId: {order['orderId']})")
        return order

    def cancel_open_orders(self) -> int:
        """Cancel all open orders for our symbol. Returns count cancelled."""
        if DRY_RUN:
            log.info(f"[DRY RUN] Would cancel all open orders for {self.symbol}")
            return 0

        result = self._retry(
            self.client.futures_cancel_all_open_orders,
            symbol=self.symbol,
        )
        count = result.get("code", 0) if isinstance(result, dict) else 0
        log.info(f"Cancelled open orders for {self.symbol}")
        return count

    def update_stop_loss(self, side: str, quantity: float, new_stop_price: float) -> Optional[Dict]:
        """
        Update trailing stop by cancelling old SL and placing new one.
        """
        self.cancel_open_orders()
        return self.place_stop_loss(side, quantity, new_stop_price)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _round_price(self, price: float) -> float:
        """Round price to the symbol's tick size grid."""
        tick = Decimal(str(self.tick_size))
        return float(Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN))

    def _round_qty(self, qty: float, price: float = None) -> float:
        """Round quantity to symbol's quantity precision.

        Default behaviour is floor-rounding (safe for SL/TP reduce-only
        orders).  When *price* is supplied the method checks whether the
        floor-rounded quantity would violate Binance's minimum notional
        ($100) and, if so, rounds UP instead.
        """
        factor = 10 ** self.qty_precision
        rounded = math.floor(qty * factor) / factor

        if price is not None:
            if rounded * price < self.min_notional:
                rounded = math.ceil(qty * factor) / factor
                log.debug(
                    f"_round_qty: ceil-rounded to {rounded} "
                    f"(notional=${rounded * price:.2f}) to meet min notional"
                )

        return rounded

    def _retry(self, func, *args, **kwargs):
        """Execute API call with retry and exponential backoff."""
        last_error = None
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except (BinanceAPIException, BinanceRequestException) as e:
                last_error = e
                wait = API_RETRY_DELAY * (2 ** attempt)
                log.warning(
                    f"API error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e} "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)
            except Exception as e:
                last_error = e
                wait = API_RETRY_DELAY * (2 ** attempt)
                log.warning(
                    f"Unexpected error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e} "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)

        log.error(f"API call failed after {API_RETRY_ATTEMPTS} attempts: {last_error}")
        raise last_error

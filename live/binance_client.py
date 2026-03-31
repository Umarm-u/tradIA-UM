# (imports unchanged)
import time
import math
from typing import Optional, Dict
from decimal import Decimal, ROUND_DOWN

import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceRequestException

from live.config_live import *
from live.logger import log

class BinanceClient:

    def __init__(self):
        self.client = None
        self.symbol = SYMBOL

        self.price_precision = 2
        self.qty_precision = 3
        self.tick_size = 0.10
        self.min_qty = 0.001
        self.min_notional = 100.0

    # ───────────── CONNECTION ─────────────

    def connect(self):
        self.client = Client(
            BINANCE_API_KEY,
            BINANCE_API_SECRET,
            requests_params={"timeout": API_TIMEOUT},
        )

        self._load_symbol_info()
        self._set_leverage()
        self._set_margin_type()

    def _load_symbol_info(self):
        info = self._retry(self.client.futures_exchange_info)

        for s in info["symbols"]:
            if s["symbol"] == self.symbol:
                self.price_precision = s["pricePrecision"]
                self.qty_precision = s["quantityPrecision"]

                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        self.tick_size = float(f["tickSize"])
                    elif f["filterType"] == "LOT_SIZE":
                        self.min_qty = float(f["minQty"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        self.min_notional = float(f.get("notional", 100.0))

                log.info(
                    f"Symbol info loaded: pricePrecision={self.price_precision}, "
                    f"qtyPrecision={self.qty_precision}, tickSize={self.tick_size}, "
                    f"minQty={self.min_qty}, minNotional={self.min_notional}"
                )
                return

        raise ValueError(f"Symbol {self.symbol} not found on Binance Futures")

    def _set_leverage(self):
        try:
            self._retry(
                self.client.futures_change_leverage,
                symbol=self.symbol,
                leverage=LEVERAGE,
            )
            log.info(f"Leverage set to {LEVERAGE}x")
        except BinanceAPIException as e:
            log.warning(f"Could not set leverage: {e}")

    def _set_margin_type(self):
        try:
            self.client.futures_change_margin_type(
                symbol=self.symbol,
                marginType="ISOLATED",
            )
            log.info("Margin type set to ISOLATED")
        except BinanceAPIException as e:
            if "No need to change margin type" in str(e):
                log.info("Margin type already ISOLATED")
            else:
                log.warning(f"Could not set margin type: {e}")

    # ───────────── ACCOUNT ─────────────

    def get_balance(self) -> float:
        account = self._retry(self.client.futures_account)

        for asset in account["assets"]:
            if asset["asset"] == "USDT":
                return float(asset["availableBalance"])

        return 0.0

    def get_total_balance(self) -> float:
        account = self._retry(self.client.futures_account)

        for asset in account["assets"]:
            if asset["asset"] == "USDT":
                return float(asset["walletBalance"])

        return 0.0

    def get_open_position(self):
        positions = self._retry(
            self.client.futures_position_information,
            symbol=self.symbol
        )

        for pos in positions:
            qty = float(pos.get("positionAmt", 0))

            if qty != 0:
                return {
                    "side": "LONG" if qty > 0 else "SHORT",
                    "quantity": abs(qty),
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "leverage": int(pos.get("leverage", 0)),
                }

        return None

    # ───────────── ORDERS ─────────────

    def place_market_order(self, side: str, quantity: float):
        price = self.get_current_price()
        quantity = self._round_qty(quantity, price)

        order = self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
        )

        avg_price = float(order.get("avgPrice") or order.get("price") or price)

        log.info(f"Market {side} {quantity} filled @ {avg_price:.2f}")

        return {
            "order": order,
            "executedQty": quantity,
            "price": avg_price,
        }

    def place_stop_loss(self, side: str, quantity: float = None, stop_price: float = None):
        # Support both (side, stop_price) and (side, quantity, stop_price) signatures
        if stop_price is None:
            stop_price = quantity
        stop_price = self._round_price(stop_price)

        return self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            type="STOP_MARKET",
            stopPrice=stop_price,
            closePosition=True,
            workingType="CONTRACT_PRICE",
        )

    def place_take_profit(self, side: str, quantity: float = None, tp_price: float = None):
        # Support both (side, tp_price) and (side, quantity, tp_price) signatures
        if tp_price is None:
            tp_price = quantity
        tp_price = self._round_price(tp_price)

        return self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=tp_price,
            closePosition=True,
            workingType="CONTRACT_PRICE",
        )

    # ───────────── DATA ─────────────

    def fetch_klines(self, limit: int = 500) -> pd.DataFrame:
        """Fetch historical klines/candlestick data."""
        raw = self._retry(
            self.client.futures_klines,
            symbol=self.symbol,
            interval=TIMEFRAME,
            limit=limit,
        )

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df[["open", "high", "low", "close", "volume"]]

        log.info(f"Fetched {len(df)} klines")
        return df

    # ───────────── ORDER MANAGEMENT ─────────────

    def update_stop_loss(self, side: str, quantity: float, stop_price: float):
        """Cancel existing SL orders and place a new one."""
        # Cancel all open orders (SL/TP) and re-place
        try:
            self._retry(
                self.client.futures_cancel_all_open_orders,
                symbol=self.symbol,
            )
        except BinanceAPIException as e:
            log.warning(f"Could not cancel orders before SL update: {e}")

        return self.place_stop_loss(side, stop_price=stop_price)

    def cancel_open_orders(self):
        """Cancel all open orders for the symbol."""
        try:
            self._retry(
                self.client.futures_cancel_all_open_orders,
                symbol=self.symbol,
            )
            log.info(f"All open orders cancelled for {self.symbol}")
        except BinanceAPIException as e:
            log.warning(f"Could not cancel open orders: {e}")

    # ───────────── HELPERS ─────────────

    def get_current_price(self) -> float:
        ticker = self._retry(self.client.futures_mark_price, symbol=self.symbol)
        return float(ticker["markPrice"])

    def _round_price(self, price: float) -> float:
        tick = Decimal(str(self.tick_size))
        return float(Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN))

    def _round_qty(self, qty: float, price: float = None) -> float:
        factor = 10 ** self.qty_precision
        rounded = math.floor(qty * factor) / factor

        if price is not None and rounded * price < self.min_notional:
            rounded = math.ceil(qty * factor) / factor

        return rounded

    def _retry(self, func, *args, **kwargs):
        last_error = None

        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except (BinanceAPIException, BinanceRequestException) as e:
                last_error = e
                wait = API_RETRY_DELAY * (2 ** attempt)
                log.warning(f"API error: {e} → retry in {wait}s")
                time.sleep(wait)

        raise last_error
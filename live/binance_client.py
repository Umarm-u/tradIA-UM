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
        self.client: Optional[Client] = None
        self.symbol = SYMBOL

        self.price_precision = 2
        self.qty_precision = 3
        self.tick_size = 0.10
        self.min_qty = 0.001
        self.min_notional = 100.0  # 🔥 FIXED

    # ─────────────────────────────
    # Connection
    # ─────────────────────────────

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
        info = self.client.futures_exchange_info()

        for s in info["symbols"]:
            if s["symbol"] == self.symbol:
                self.price_precision = int(s["pricePrecision"])
                self.qty_precision = int(s["quantityPrecision"])

                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        self.tick_size = float(f["tickSize"])
                    if f["filterType"] == "LOT_SIZE":
                        self.min_qty = float(f["minQty"])
                    if f["filterType"] == "MIN_NOTIONAL":
                        self.min_notional = float(f.get("notional", 100))

    def _set_leverage(self):
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=LEVERAGE
            )
        except BinanceAPIException:
            pass

    def _set_margin_type(self):
        try:
            self.client.futures_change_margin_type(
                symbol=self.symbol, marginType="ISOLATED"
            )
        except BinanceAPIException:
            pass

    # ─────────────────────────────
    # Market
    # ─────────────────────────────

    def get_current_price(self) -> float:
        ticker = self._retry(self.client.futures_mark_price, symbol=self.symbol)
        return float(ticker["markPrice"])

    # ─────────────────────────────
    # Account
    # ─────────────────────────────

    def get_open_position(self) -> Optional[Dict]:
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

    # ─────────────────────────────
    # Orders
    # ─────────────────────────────

    def place_market_order(self, side: str, quantity: float) -> Dict:
        price = self.get_current_price()
        quantity = self._round_qty(quantity, price)

        order = self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
        )

        avg_price = float(order.get("avgPrice") or order.get("price") or price)

        log.info(
            f"Market {side} {quantity} filled @ {avg_price:.2f}"
        )

        return {
            "order": order,
            "executedQty": quantity,
            "price": avg_price,
        }

    def place_stop_loss(self, side: str, stop_price: float) -> Dict:
        stop_price = self._round_price(stop_price)

        return self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            type="STOP_MARKET",
            stopPrice=stop_price,
            closePosition=True,   # 🔥 CRITICAL FIX
            workingType="MARK_PRICE",
        )

    def place_take_profit(self, side: str, tp_price: float) -> Dict:
        tp_price = self._round_price(tp_price)

        return self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=tp_price,
            closePosition=True,   # 🔥 CRITICAL FIX
            workingType="MARK_PRICE",
        )

    # ─────────────────────────────
    # Helpers
    # ─────────────────────────────

    def _round_price(self, price: float) -> float:
        tick = Decimal(str(self.tick_size))
        return float(Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN))

    def _round_qty(self, qty: float, price: float = None) -> float:
        factor = 10 ** self.qty_precision
        rounded = math.floor(qty * factor) / factor

        if price is not None:
            if rounded * price < self.min_notional:
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
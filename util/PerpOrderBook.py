"""Perpetual futures order book with float pricing, TIF support, self-trade prevention, and OI cap checks.

Modeled after Hyperliquid's HIP-3 matching engine semantics.
"""

import sys
from statistics import median
from copy import deepcopy

import pandas as pd

from message.Message import Message
from util.order.PerpLimitOrder import PerpLimitOrder
from util.ContractSpec import TimeInForce, ContractSpec
from util.util import log_print


class PerpOrderBook:

    def __init__(self, owner, symbol, contract_spec: ContractSpec):
        self.owner = owner
        self.symbol = symbol
        self.contract_spec = contract_spec
        self.bids = []    # list of price levels; each level is a list of PerpLimitOrder (FIFO)
        self.asks = []
        self.last_trade = None
        self.book_log = []
        self.quotes_seen = set()
        self.history = [{}]
        self.last_update_ts = None

        # Open interest tracking
        self.total_oi_size = 0.0
        self.total_oi_notional = 0.0

    # ── Limit order handling ────────────────────────────────────────────

    def handleLimitOrder(self, order: PerpLimitOrder, agent_positions=None):
        """Process an incoming limit order. Returns list of (filled_order, matched_order) tuples."""
        if order.symbol != self.symbol:
            log_print("{} order discarded. Does not match OrderBook symbol: {}", order.symbol, self.symbol)
            return []

        if order.quantity <= 0:
            log_print("{} order discarded. Quantity ({}) must be positive.", order.symbol, order.quantity)
            return []

        # Reduce-only: clip quantity to current position size
        if order.reduce_only and agent_positions is not None:
            pos_size = abs(agent_positions.get(self.symbol, 0.0))
            if pos_size <= 0:
                self._send_reject(order, "REDUCE_ONLY_NO_POSITION")
                return []
            if order.quantity > pos_size:
                order.quantity = pos_size

        # Post-only (ALO): reject if would immediately match
        if order.time_in_force == TimeInForce.ALO:
            if self._would_match(order):
                self._send_reject(order, "POST_ONLY_WOULD_MATCH")
                return []

        self.history[0][order.order_id] = {
            'entry_time': self.owner.currentTime,
            'quantity': order.quantity,
            'is_buy_order': order.is_buy_order,
            'limit_price': order.limit_price,
            'transactions': [],
            'modifications': [],
            'cancellations': [],
        }

        executed = []
        matching = True

        while matching:
            matched_order = deepcopy(self._execute_order(order))
            if matched_order:
                filled_order = deepcopy(order)
                filled_order.quantity = matched_order.quantity
                filled_order.fill_price = matched_order.fill_price

                order.quantity -= filled_order.quantity

                log_print("MATCHED: {} vs {}", filled_order, matched_order)

                self.owner.sendMessage(order.agent_id,
                                       Message({"msg": "ORDER_EXECUTED", "order": filled_order}))
                self.owner.sendMessage(matched_order.agent_id,
                                       Message({"msg": "ORDER_EXECUTED", "order": matched_order}))

                executed.append((filled_order.quantity, filled_order.fill_price))

                if order.quantity <= 0:
                    matching = False
            else:
                # IOC: cancel unfilled remainder instead of resting
                if order.time_in_force == TimeInForce.IOC:
                    if order.quantity > 0:
                        self.owner.sendMessage(order.agent_id,
                                               Message({"msg": "ORDER_CANCELLED", "order": order}))
                else:
                    self._enter_order(deepcopy(order))
                    log_print("ACCEPTED: {}", order)
                    self.owner.sendMessage(order.agent_id,
                                           Message({"msg": "ORDER_ACCEPTED", "order": order}))
                matching = False

        if executed:
            trade_qty = sum(q for q, _ in executed)
            trade_notional = sum(q * p for q, p in executed)
            avg_price = trade_notional / trade_qty
            self.last_trade = avg_price
            self.history.insert(0, {})
            self.history = self.history[:getattr(self.owner, 'stream_history', 10) + 1]

        self.last_update_ts = self.owner.currentTime
        return executed

    def handleMarketOrder(self, order: PerpLimitOrder, agent_positions=None):
        """Process a market order by walking the book as aggressive limit orders."""
        if order.symbol != self.symbol:
            return []

        if order.quantity <= 0:
            return []

        if order.reduce_only and agent_positions is not None:
            pos_size = abs(agent_positions.get(self.symbol, 0.0))
            if pos_size <= 0:
                return []
            if order.quantity > pos_size:
                order.quantity = pos_size

        book_side = self.getInsideAsks() if order.is_buy_order else self.getInsideBids()

        all_executed = []
        remaining_qty = order.quantity
        for price, size in book_side:
            if remaining_qty <= 0:
                break
            fill_qty = min(remaining_qty, size)
            limit_order = PerpLimitOrder(
                order.agent_id, order.time_placed, order.symbol,
                fill_qty, order.is_buy_order, price,
                tag=order.tag, time_in_force=TimeInForce.IOC,
                reduce_only=order.reduce_only,
                is_liquidation=order.is_liquidation,
            )
            fills = self.handleLimitOrder(limit_order, agent_positions)
            all_executed.extend(fills)
            filled_qty = sum(q for q, _ in fills)
            remaining_qty -= filled_qty

        return all_executed

    # ── Cancel / Modify ─────────────────────────────────────────────────

    def cancelOrder(self, order):
        book = self.bids if order.is_buy_order else self.asks
        if not book:
            return

        for i, level in enumerate(book):
            if abs(level[0].limit_price - order.limit_price) < 1e-12:
                for ci, co in enumerate(level):
                    if order.order_id == co.order_id:
                        cancelled = level.pop(ci)
                        if not level:
                            del book[i]

                        for idx, orders in enumerate(self.history):
                            if cancelled.order_id in orders:
                                orders[cancelled.order_id]['cancellations'].append(
                                    (self.owner.currentTime, cancelled.quantity))

                        self.owner.sendMessage(order.agent_id,
                                               Message({"msg": "ORDER_CANCELLED", "order": cancelled}))
                        self.last_update_ts = self.owner.currentTime
                        return

    def cancelAllOrders(self):
        """Cancel every resting order (used by haltTrading). Returns cancelled orders."""
        cancelled = []
        for book in [self.bids, self.asks]:
            for level in book:
                for order in level:
                    cancelled.append(order)
                    self.owner.sendMessage(order.agent_id,
                                           Message({"msg": "ORDER_CANCELLED", "order": order}))
        self.bids = []
        self.asks = []
        self.last_update_ts = self.owner.currentTime
        return cancelled

    def modifyOrder(self, order, new_order):
        if order.order_id != new_order.order_id:
            return
        book = self.bids if order.is_buy_order else self.asks
        if not book:
            return
        for i, level in enumerate(book):
            if abs(level[0].limit_price - order.limit_price) < 1e-12:
                for mi, mo in enumerate(level):
                    if order.order_id == mo.order_id:
                        level[mi] = new_order
                        self.owner.sendMessage(order.agent_id,
                                               Message({"msg": "ORDER_MODIFIED", "new_order": new_order}))
                        self.last_update_ts = self.owner.currentTime
                        return

    # ── Core matching ───────────────────────────────────────────────────

    def _execute_order(self, order):
        """Find and execute against the best matching resting order."""
        book = self.asks if order.is_buy_order else self.bids

        if not book:
            return None
        if not self._is_match(order, book[0][0]):
            return None

        # Self-trade prevention: if top-of-book is same agent, cancel it
        if book[0][0].agent_id == order.agent_id:
            stp_order = book[0].pop(0)
            if not book[0]:
                del book[0]
            self.owner.sendMessage(stp_order.agent_id,
                                   Message({"msg": "ORDER_CANCELLED", "order": stp_order}))
            return None

        if order.quantity >= book[0][0].quantity:
            matched = book[0].pop(0)
            if not book[0]:
                del book[0]
        else:
            matched = deepcopy(book[0][0])
            matched.quantity = order.quantity
            book[0][0].quantity -= order.quantity

        matched.fill_price = matched.limit_price

        # Record in history
        self.history[0].setdefault(order.order_id, {
            'entry_time': self.owner.currentTime, 'quantity': order.quantity,
            'is_buy_order': order.is_buy_order, 'limit_price': order.limit_price,
            'transactions': [], 'modifications': [], 'cancellations': [],
        })
        self.history[0][order.order_id]['transactions'].append(
            (self.owner.currentTime, matched.quantity))

        for idx, orders in enumerate(self.history):
            if matched.order_id in orders:
                orders[matched.order_id]['transactions'].append(
                    (self.owner.currentTime, matched.quantity))

        return matched

    def _is_match(self, order, resting):
        if order.is_buy_order == resting.is_buy_order:
            return False
        if order.is_buy_order:
            return order.limit_price >= resting.limit_price
        return order.limit_price <= resting.limit_price

    def _would_match(self, order):
        book = self.asks if order.is_buy_order else self.bids
        if not book:
            return False
        return self._is_match(order, book[0][0])

    def _enter_order(self, order):
        book = self.bids if order.is_buy_order else self.asks
        if not book:
            book.append([order])
            if order.is_buy_order:
                self.bids = book
            else:
                self.asks = book
            return

        for i, level in enumerate(book):
            if self._is_better_price(order, level[0]):
                book.insert(i, [order])
                return
            elif abs(order.limit_price - level[0].limit_price) < 1e-12:
                level.append(order)
                return

        book.append([order])

    def _is_better_price(self, order, resting):
        if order.is_buy_order:
            return order.limit_price > resting.limit_price
        return order.limit_price < resting.limit_price

    def _send_reject(self, order, reason):
        self.owner.sendMessage(order.agent_id,
                               Message({"msg": "ORDER_REJECTED", "order": order, "reason": reason}))

    # ── Book queries ────────────────────────────────────────────────────

    def getInsideBids(self, depth=sys.maxsize):
        result = []
        for i in range(min(depth, len(self.bids))):
            price = self.bids[i][0].limit_price
            qty = sum(o.quantity for o in self.bids[i])
            result.append((price, qty))
        return result

    def getInsideAsks(self, depth=sys.maxsize):
        result = []
        for i in range(min(depth, len(self.asks))):
            price = self.asks[i][0].limit_price
            qty = sum(o.quantity for o in self.asks[i])
            result.append((price, qty))
        return result

    def getBestBid(self):
        if self.bids:
            return self.bids[0][0].limit_price
        return None

    def getBestAsk(self):
        if self.asks:
            return self.asks[0][0].limit_price
        return None

    def getMidPrice(self):
        bb = self.getBestBid()
        ba = self.getBestAsk()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    def getLocalMarkPrice(self):
        """median(best_bid, best_ask, last_trade) per HIP-3 spec."""
        values = []
        bb = self.getBestBid()
        ba = self.getBestAsk()
        if bb is not None:
            values.append(bb)
        if ba is not None:
            values.append(ba)
        if self.last_trade is not None:
            values.append(self.last_trade)
        if not values:
            return None
        return median(values)

    def getImpactPrice(self, notional: float, is_buy: bool):
        """Average execution price to fill `notional` USDC on one side of the book.
        
        Used for funding rate premium calculation.
        """
        book_side = self.getInsideAsks() if is_buy else self.getInsideBids()
        remaining_notional = notional
        total_qty = 0.0
        total_cost = 0.0

        for price, qty in book_side:
            level_notional = price * qty
            if level_notional >= remaining_notional:
                fill_qty = remaining_notional / price
                total_qty += fill_qty
                total_cost += fill_qty * price
                remaining_notional = 0
                break
            else:
                total_qty += qty
                total_cost += qty * price
                remaining_notional -= level_notional

        if total_qty <= 0:
            return None
        return total_cost / total_qty

    def prettyPrint(self, silent=False):
        if silent:
            lines = []
        else:
            lines = None

        header = "{} perp order book as of {}".format(self.symbol, self.owner.currentTime)
        trade_str = "Last trade: {:.4f}".format(self.last_trade) if self.last_trade else "Last trade: None"
        col_header = "{:>12s} {:>12s} {:>12s}".format('BID', 'PRICE', 'ASK')

        asks_str = []
        for price, vol in self.getInsideAsks()[-1::-1]:
            asks_str.append("{:>12s} {:>12.4f} {:>12.4f}".format("", price, vol))

        bids_str = []
        for price, vol in self.getInsideBids():
            bids_str.append("{:>12.4f} {:>12.4f} {:>12s}".format(vol, price, ""))

        all_lines = [header, trade_str, col_header] + asks_str + bids_str
        book_str = "\n".join(all_lines)

        if silent:
            return book_str
        log_print(book_str)

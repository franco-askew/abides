"""Perpetual futures order book."""

import sys
from statistics import median

import pandas as pd

from util.order.PerpLimitOrder import PerpLimitOrder


class PerpOrderBook:
    def __init__(self, owner, symbol, contract_spec):
        self.owner = owner
        self.symbol = symbol
        self.contract_spec = contract_spec
        self.bids = []
        self.asks = []
        self.order_index = {}
        self.last_trade = None
        self.history = [{}]
        self.last_update_ts = None
        self._transacted_volume = {
            "unrolled_transactions": pd.DataFrame(columns=["execution_time", "quantity"]),
            "history_previous_length": 0,
        }

    def _book_for_side(self, is_buy_order):
        return self.bids if is_buy_order else self.asks

    def _counter_book_for_side(self, is_buy_order):
        return self.asks if is_buy_order else self.bids

    def _is_match(self, order, resting):
        if order.is_buy_order == resting.is_buy_order:
            return False
        if order.is_buy_order:
            return order.limit_price >= resting.limit_price
        return order.limit_price <= resting.limit_price

    def would_match(self, order):
        book = self._counter_book_for_side(order.is_buy_order)
        if not book:
            return False
        return self._is_match(order, book[0][0])

    def enter_order(self, order: PerpLimitOrder):
        book = self._book_for_side(order.is_buy_order)
        inserted = False
        for idx, level in enumerate(book):
            reference = level[0]
            if self._is_better_price(order, reference):
                book.insert(idx, [order])
                inserted = True
                break
            if abs(order.limit_price - reference.limit_price) < 1e-12:
                level.append(order)
                inserted = True
                break
        if not inserted:
            book.append([order])
        self.order_index[order.order_id] = order
        self.history[0][order.order_id] = {
            "entry_time": self.owner.currentTime,
            "quantity": order.quantity,
            "is_buy_order": order.is_buy_order,
            "limit_price": order.limit_price,
            "transactions": [],
            "modifications": [],
            "cancellations": [],
        }
        self.last_update_ts = self.owner.currentTime

    def _is_better_price(self, order, resting):
        if order.is_buy_order:
            return order.limit_price > resting.limit_price
        return order.limit_price < resting.limit_price

    def cancel_order(self, order_id: int):
        order = self.order_index.pop(order_id, None)
        if order is None:
            return None
        book = self._book_for_side(order.is_buy_order)
        for level_idx, level in enumerate(book):
            for order_idx, candidate in enumerate(level):
                if candidate.order_id == order_id:
                    cancelled = level.pop(order_idx)
                    if not level:
                        del book[level_idx]
                    self._record_cancellation(cancelled)
                    self.last_update_ts = self.owner.currentTime
                    return cancelled
        return order

    def cancel_all_orders(self):
        cancelled = list(self.order_index.values())
        self.bids = []
        self.asks = []
        self.order_index = {}
        for order in cancelled:
            self._record_cancellation(order)
        self.last_update_ts = self.owner.currentTime
        return cancelled

    def amend_order_in_place(self, order_id: int, new_quantity: float) -> bool:
        """Amend an order's quantity in place, preserving queue position.

        Only valid for same-price, same-side amendments. Returns True if
        the order was found and amended, False otherwise.
        """
        order = self.order_index.get(order_id)
        if order is None:
            return False
        order.quantity = new_quantity
        self.history[0].setdefault(
            order_id,
            {
                "entry_time": self.owner.currentTime,
                "quantity": new_quantity,
                "is_buy_order": order.is_buy_order,
                "limit_price": order.limit_price,
                "transactions": [],
                "modifications": [],
                "cancellations": [],
            },
        )
        self.history[0][order_id]["modifications"].append((self.owner.currentTime, new_quantity, order.limit_price))
        self.last_update_ts = self.owner.currentTime
        return True

    def modify_order(self, old_order_id: int, new_order: PerpLimitOrder):
        cancelled = self.cancel_order(old_order_id)
        if cancelled is None:
            return None
        self.enter_order(new_order)
        self.history[0][new_order.order_id]["modifications"].append((self.owner.currentTime, new_order.quantity, new_order.limit_price))
        return cancelled

    def match_order(self, incoming_order: PerpLimitOrder, maker_validator=None):
        fills = []
        cancelled = []
        book = self._counter_book_for_side(incoming_order.is_buy_order)

        while incoming_order.quantity > 1e-12 and book:
            resting = book[0][0]
            if not self._is_match(incoming_order, resting):
                break

            if resting.agent_id == incoming_order.agent_id:
                cancelled.append(self.cancel_order(resting.order_id))
                book = self._counter_book_for_side(incoming_order.is_buy_order)
                continue

            fill_qty = min(incoming_order.quantity, resting.quantity)
            if maker_validator is not None:
                ok, reason = maker_validator(resting, fill_qty)
                if not ok:
                    cancelled.append(self.cancel_order(resting.order_id))
                    book = self._counter_book_for_side(incoming_order.is_buy_order)
                    continue

            matched_order = self._remove_quantity(resting, fill_qty)
            filled_order = incoming_order.clone()
            filled_order.quantity = fill_qty
            filled_order.fill_price = matched_order.fill_price
            incoming_order.quantity -= fill_qty
            fills.append((filled_order, matched_order))

            if incoming_order.quantity <= 1e-12:
                break
            book = self._counter_book_for_side(incoming_order.is_buy_order)

        if fills:
            trade_qty = sum(filled.quantity for filled, _ in fills)
            trade_notional = sum(filled.quantity * filled.fill_price for filled, _ in fills)
            self.last_trade = trade_notional / trade_qty
            self.history.insert(0, {})
            self.history = self.history[: getattr(self.owner, "stream_history", 10) + 1]
            self.last_update_ts = self.owner.currentTime

        return fills, [order for order in cancelled if order is not None]

    def _remove_quantity(self, resting, fill_qty):
        book = self._book_for_side(resting.is_buy_order)
        if fill_qty >= resting.quantity - 1e-12:
            matched = book[0].pop(0)
            if not book[0]:
                del book[0]
            self.order_index.pop(matched.order_id, None)
        else:
            matched = resting.clone()
            matched.quantity = fill_qty
            resting.quantity -= fill_qty

        matched.fill_price = matched.limit_price
        self._record_transaction(resting.order_id, fill_qty)
        self._record_transaction(matched.order_id, fill_qty)
        return matched

    def _record_transaction(self, order_id, quantity):
        self.history[0].setdefault(
            order_id,
            {
                "entry_time": self.owner.currentTime,
                "quantity": quantity,
                "is_buy_order": True,
                "limit_price": 0,
                "transactions": [],
                "modifications": [],
                "cancellations": [],
            },
        )
        self.history[0][order_id]["transactions"].append((self.owner.currentTime, quantity))
        for orders in self.history[1:]:
            if order_id in orders:
                orders[order_id]["transactions"].append((self.owner.currentTime, quantity))

    def _record_cancellation(self, order):
        for orders in self.history:
            if order.order_id in orders:
                orders[order.order_id]["cancellations"].append((self.owner.currentTime, order.quantity))

    def getInsideBids(self, depth=sys.maxsize):
        result = []
        for idx in range(min(depth, len(self.bids))):
            price = self.bids[idx][0].limit_price
            qty = sum(order.quantity for order in self.bids[idx])
            result.append((price, qty))
        return result

    def getInsideAsks(self, depth=sys.maxsize):
        result = []
        for idx in range(min(depth, len(self.asks))):
            price = self.asks[idx][0].limit_price
            qty = sum(order.quantity for order in self.asks[idx])
            result.append((price, qty))
        return result

    def getBestBid(self):
        return self.bids[0][0].limit_price if self.bids else None

    def getBestAsk(self):
        return self.asks[0][0].limit_price if self.asks else None

    def getLocalMarkPrice(self):
        values = []
        best_bid = self.getBestBid()
        best_ask = self.getBestAsk()
        if best_bid is not None:
            values.append(best_bid)
        if best_ask is not None:
            values.append(best_ask)
        if self.last_trade is not None:
            values.append(self.last_trade)
        if not values:
            return None
        return median(values)

    def getImpactPrice(self, notional: float, is_buy: bool):
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
                remaining_notional = 0.0
                break
            total_qty += qty
            total_cost += qty * price
            remaining_notional -= level_notional

        if total_qty <= 0:
            return None
        return total_cost / total_qty

    def get_transacted_volume(self, lookback_period="10min"):
        recent = self._get_recent_history()
        self._update_unrolled_transactions(recent)
        df = self._transacted_volume["unrolled_transactions"]
        if df.empty:
            return 0
        window_start = self.owner.currentTime - pd.to_timedelta(lookback_period)
        return df.loc[df["execution_time"] >= window_start, "quantity"].sum()

    def _get_recent_history(self):
        prev_len = self._transacted_volume["history_previous_length"]
        cur_len = len(self.history)
        if prev_len == 0:
            self._transacted_volume["history_previous_length"] = cur_len
            return self.history
        if prev_len == cur_len:
            return []
        idx = cur_len - prev_len
        recent = self.history[:idx]
        self._transacted_volume["history_previous_length"] = cur_len
        return recent

    def _update_unrolled_transactions(self, recent_history):
        rows = []
        for entry in recent_history:
            for _, value in entry.items():
                for txn in value.get("transactions", []):
                    rows.append(txn)
        if rows:
            new_df = pd.DataFrame(rows, columns=["execution_time", "quantity"])
            old_df = self._transacted_volume["unrolled_transactions"]
            self._transacted_volume["unrolled_transactions"] = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(keep="last")

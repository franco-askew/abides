"""Perpetual futures limit order with HIP-3 features: TIF, reduce-only, liquidation flag."""

from util.order.Order import Order
from util.ContractSpec import TimeInForce
from copy import deepcopy


class PerpLimitOrder(Order):

    def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, limit_price,
                 order_id=None, tag=None, time_in_force=TimeInForce.GTC,
                 reduce_only=False, is_liquidation=False,
                 trigger_price=None, trigger_type=None):

        super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id, tag=tag)

        self.limit_price: float = float(limit_price)
        self.time_in_force = time_in_force
        self.reduce_only = reduce_only
        self.is_liquidation = is_liquidation
        self.trigger_price = float(trigger_price) if trigger_price is not None else None
        self.trigger_type = trigger_type  # "STOP_MARKET", "STOP_LIMIT", "TAKE_MARKET", "TAKE_LIMIT"

    def __str__(self):
        filled = ''
        if self.fill_price:
            filled = " (filled @ {:.4f})".format(self.fill_price)
        flags = []
        if self.time_in_force != TimeInForce.GTC:
            flags.append(self.time_in_force.value)
        if self.reduce_only:
            flags.append("RO")
        if self.is_liquidation:
            flags.append("LIQ")
        flag_str = " [{}]".format(",".join(flags)) if flags else ""
        tag_str = " [{}]".format(self.tag) if self.tag else ""
        return "(Agent {} @ {}{}): {} {:.4f} {} @ {:.4f}{}{}".format(
            self.agent_id, self.time_placed, tag_str,
            "BUY" if self.is_buy_order else "SELL",
            self.quantity, self.symbol, self.limit_price,
            filled, flag_str)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        order = PerpLimitOrder(
            self.agent_id, self.time_placed, self.symbol, self.quantity,
            self.is_buy_order, self.limit_price,
            order_id=self.order_id, tag=self.tag,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            is_liquidation=self.is_liquidation,
            trigger_price=self.trigger_price,
            trigger_type=self.trigger_type,
        )
        Order._order_ids.discard(order.order_id)
        order.fill_price = self.fill_price
        return order

    def __deepcopy__(self, memodict={}):
        order = PerpLimitOrder(
            deepcopy(self.agent_id, memodict),
            deepcopy(self.time_placed, memodict),
            deepcopy(self.symbol, memodict),
            deepcopy(self.quantity, memodict),
            deepcopy(self.is_buy_order, memodict),
            deepcopy(self.limit_price, memodict),
            order_id=deepcopy(self.order_id, memodict),
            tag=deepcopy(self.tag, memodict),
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            is_liquidation=self.is_liquidation,
            trigger_price=deepcopy(self.trigger_price, memodict),
            trigger_type=self.trigger_type,
        )
        order.fill_price = deepcopy(self.fill_price, memodict)
        return order

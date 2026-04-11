"""Perpetual futures limit order with Hyperliquid-specific metadata."""

from util.ContractSpec import MarginType, TimeInForce
from util.order.Order import Order


class PerpLimitOrder(Order):
    def __init__(
        self,
        agent_id,
        time_placed,
        symbol,
        quantity,
        is_buy_order,
        limit_price,
        order_id=None,
        tag=None,
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        is_liquidation=False,
        trigger_price=None,
        trigger_type=None,
        requested_leverage=None,
        margin_type=MarginType.INHERIT,
        parent_order_id=None,
        tpsl_group_id=None,
        tpsl_mode=None,
        trigger_slippage_bps=None,
        dynamic_size=False,
        is_market_order=False,
    ):
        super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id, tag=tag)

        self.limit_price = float(limit_price)
        self.time_in_force = time_in_force
        self.reduce_only = reduce_only
        self.is_liquidation = is_liquidation
        self.trigger_price = float(trigger_price) if trigger_price is not None else None
        self.trigger_type = trigger_type
        self.requested_leverage = requested_leverage
        self.margin_type = margin_type if isinstance(margin_type, MarginType) else MarginType(margin_type)
        self.parent_order_id = parent_order_id
        self.tpsl_group_id = tpsl_group_id
        self.tpsl_mode = tpsl_mode
        self.trigger_slippage_bps = trigger_slippage_bps
        self.dynamic_size = dynamic_size
        self.is_market_order = is_market_order

    def __str__(self):
        filled = ""
        if self.fill_price:
            filled = " (filled @ {:.6f})".format(self.fill_price)
        flags = []
        if self.is_market_order:
            flags.append("MKT")
        if self.time_in_force != TimeInForce.GTC:
            flags.append(self.time_in_force.value)
        if self.reduce_only:
            flags.append("RO")
        if self.is_liquidation:
            flags.append("LIQ")
        if self.margin_type != MarginType.INHERIT:
            flags.append(self.margin_type.value.upper())
        if self.trigger_type:
            flags.append(self.trigger_type)
        flag_str = " [{}]".format(",".join(flags)) if flags else ""
        tag_str = " [{}]".format(self.tag) if self.tag else ""
        return "(Agent {} @ {}{}): {} {:.6f} {} @ {:.6f}{}{}".format(
            self.agent_id,
            self.time_placed,
            tag_str,
            "BUY" if self.is_buy_order else "SELL",
            self.quantity,
            self.symbol,
            self.limit_price,
            filled,
            flag_str,
        )

    def __repr__(self):
        return self.__str__()

    def clone(self):
        order = PerpLimitOrder(
            self.agent_id,
            self.time_placed,
            self.symbol,
            self.quantity,
            self.is_buy_order,
            self.limit_price,
            order_id=self.order_id,
            tag=self.tag,
            time_in_force=self.time_in_force,
            reduce_only=self.reduce_only,
            is_liquidation=self.is_liquidation,
            trigger_price=self.trigger_price,
            trigger_type=self.trigger_type,
            requested_leverage=self.requested_leverage,
            margin_type=self.margin_type,
            parent_order_id=self.parent_order_id,
            tpsl_group_id=self.tpsl_group_id,
            tpsl_mode=self.tpsl_mode,
            trigger_slippage_bps=self.trigger_slippage_bps,
            dynamic_size=self.dynamic_size,
            is_market_order=self.is_market_order,
        )
        order.fill_price = self.fill_price
        return order

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memodict=None):
        order = self.clone()
        if memodict is not None:
            memodict[id(self)] = order
        return order

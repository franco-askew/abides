"""Canonical perpetual futures account state."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from util.ContractSpec import MarginMode, MarginTable, MarginType


@dataclass
class Position:
    size: float = 0.0
    entry_price: float = 0.0
    leverage: int = 10
    margin_type: MarginType = MarginType.CROSS
    isolated_margin: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @property
    def is_isolated(self) -> bool:
        return self.margin_type == MarginType.ISOLATED

    @property
    def abs_size(self) -> float:
        return abs(self.size)

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.size * (mark_price - self.entry_price)

    def notional_value(self, mark_price: float) -> float:
        return abs(self.size) * mark_price


@dataclass
class OrderHold:
    order_id: int
    symbol: str
    is_buy_order: bool
    leverage: int
    margin_type: MarginType
    remaining_qty: float
    cross_reserved: float = 0.0
    isolated_reserved: float = 0.0
    opening_qty_reserved: float = 0.0
    placement_price: float = 0.0

    @property
    def total_reserved(self) -> float:
        return self.cross_reserved + self.isolated_reserved


class PerpAccount:
    def __init__(self, starting_balance: float = 0.0):
        self.balance: float = float(starting_balance)
        self.starting_balance: float = float(starting_balance)
        self.positions: Dict[str, Position] = {}
        self.order_holds: Dict[int, OrderHold] = {}
        self.total_fees_paid: float = 0.0
        self.total_funding_paid: float = 0.0
        self.total_realized_pnl: float = 0.0

    def get_position(self, symbol: str) -> Position:
        pos = self.positions.get(symbol)
        return pos if pos is not None else Position()

    def get_position_size(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.size if pos else 0.0

    def has_position(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and abs(pos.size) > 1e-12

    def cross_reserved_margin(self) -> float:
        return sum(hold.cross_reserved for hold in self.order_holds.values())

    def isolated_reserved_margin(self) -> float:
        return sum(hold.isolated_reserved for hold in self.order_holds.values())

    def total_isolated_margin(self) -> float:
        return sum(pos.isolated_margin for pos in self.positions.values() if pos.is_isolated)

    def cross_unrealized_pnl(self, mark_prices: Dict[str, float]) -> float:
        pnl = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_isolated or pos.size == 0:
                continue
            pnl += pos.unrealized_pnl(mark_prices.get(symbol, pos.entry_price))
        return pnl

    def cross_account_value(self, mark_prices: Dict[str, float]) -> float:
        return self.balance + self.cross_unrealized_pnl(mark_prices)

    def cross_maintenance_margin(
        self,
        mark_prices: Dict[str, float],
        margin_tables: Dict[str, MarginTable],
    ) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_isolated or pos.size == 0:
                continue
            mark_price = mark_prices.get(symbol, pos.entry_price)
            total += margin_tables[symbol].get_maintenance_margin(pos.notional_value(mark_price))
        return total

    def cross_available_margin(
        self,
        mark_prices: Dict[str, float],
        margin_tables: Dict[str, MarginTable],
        exclude_order_id: Optional[int] = None,
    ) -> float:
        reserved = self.cross_reserved_margin()
        if exclude_order_id is not None and exclude_order_id in self.order_holds:
            reserved -= self.order_holds[exclude_order_id].cross_reserved
        return self.cross_account_value(mark_prices) - self.cross_maintenance_margin(mark_prices, margin_tables) - reserved

    def available_cash(self, exclude_order_id: Optional[int] = None) -> float:
        isolated_reserved = self.isolated_reserved_margin()
        if exclude_order_id is not None and exclude_order_id in self.order_holds:
            isolated_reserved -= self.order_holds[exclude_order_id].isolated_reserved
        return self.balance - isolated_reserved

    def isolated_account_value(self, symbol: str, mark_price: float) -> float:
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_isolated:
            return 0.0
        return pos.isolated_margin + pos.unrealized_pnl(mark_price)

    def isolated_maintenance_margin(
        self,
        symbol: str,
        mark_price: float,
        margin_table: Optional[MarginTable] = None,
    ) -> float:
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_isolated or pos.size == 0:
            return 0.0
        if margin_table is None:
            return pos.notional_value(mark_price) * 0.05
        return margin_table.get_maintenance_margin(pos.notional_value(mark_price))

    def total_equity(self, mark_prices: Dict[str, float]) -> float:
        equity = self.balance
        for symbol, pos in self.positions.items():
            mark_price = mark_prices.get(symbol, pos.entry_price)
            equity += pos.unrealized_pnl(mark_price)
            if pos.is_isolated:
                equity += pos.isolated_margin
        return equity

    def estimate_opening_qty(self, symbol: str, quantity: float, is_buy_order: bool) -> float:
        pos = self.get_position(symbol)
        if pos.size == 0:
            return quantity
        if (pos.size > 0 and is_buy_order) or (pos.size < 0 and not is_buy_order):
            return quantity
        return max(0.0, quantity - abs(pos.size))

    def ensure_position(self, symbol: str, leverage: int, margin_type: MarginType) -> Position:
        pos = self.positions.get(symbol)
        if pos is None:
            pos = Position(leverage=leverage, margin_type=margin_type)
            self.positions[symbol] = pos
        return pos

    def add_hold(self, hold: OrderHold) -> None:
        self.order_holds[hold.order_id] = hold

    def release_hold(self, order_id: int) -> Optional[OrderHold]:
        return self.order_holds.pop(order_id, None)

    def resize_hold(self, order_id: int, remaining_qty: float, cross_reserved: float, isolated_reserved: float, opening_qty_reserved: float) -> None:
        hold = self.order_holds.get(order_id)
        if hold is None:
            return
        hold.remaining_qty = remaining_qty
        hold.cross_reserved = cross_reserved
        hold.isolated_reserved = isolated_reserved
        hold.opening_qty_reserved = opening_qty_reserved

    def apply_fill(
        self,
        symbol: str,
        fill_qty: float,
        fill_price: float,
        is_buy: bool,
        leverage: int,
        fee: float,
        margin_type: MarginType = MarginType.CROSS,
        order_id: Optional[int] = None,
        margin_mode: MarginMode = MarginMode.NORMAL,
    ):
        signed_qty = fill_qty if is_buy else -fill_qty
        self.balance -= fee
        self.total_fees_paid += fee

        pos = self.positions.get(symbol)
        if pos is None:
            pos = Position(leverage=leverage, margin_type=margin_type)
            self.positions[symbol] = pos

        hold = self.order_holds.get(order_id) if order_id is not None else None

        if pos.size == 0:
            pos.size = signed_qty
            pos.entry_price = fill_price
            pos.leverage = leverage
            pos.margin_type = margin_type
            if margin_type == MarginType.ISOLATED:
                margin_to_allocate = min(hold.isolated_reserved if hold else 0.0, fill_qty * fill_price / leverage)
                pos.isolated_margin += margin_to_allocate
                self.balance -= margin_to_allocate
        elif (pos.size > 0 and is_buy) or (pos.size < 0 and not is_buy):
            old_notional = abs(pos.size) * pos.entry_price
            new_notional = fill_qty * fill_price
            total_size = abs(pos.size) + fill_qty
            pos.entry_price = (old_notional + new_notional) / total_size if total_size > 0 else fill_price
            pos.size += signed_qty
            pos.leverage = leverage
            if pos.margin_type == MarginType.ISOLATED:
                margin_to_allocate = min(hold.isolated_reserved if hold else 0.0, fill_qty * fill_price / leverage)
                pos.isolated_margin += margin_to_allocate
                self.balance -= margin_to_allocate
        else:
            reduce_qty = min(fill_qty, abs(pos.size))
            realized_pnl = reduce_qty * (fill_price - pos.entry_price) * (1 if pos.size > 0 else -1)
            self.balance += realized_pnl
            self.total_realized_pnl += realized_pnl

            if pos.is_isolated and reduce_qty > 0:
                margin_release = pos.isolated_margin * (reduce_qty / abs(pos.size))
                pos.isolated_margin -= margin_release
                self.balance += margin_release

            remaining_after_reduce = abs(pos.size) - reduce_qty
            flip_qty = fill_qty - reduce_qty

            if remaining_after_reduce <= 1e-12 and flip_qty <= 1e-12:
                pos.size = 0.0
                pos.entry_price = 0.0
                pos.isolated_margin = 0.0
            elif remaining_after_reduce <= 1e-12 and flip_qty > 0:
                pos.size = flip_qty if is_buy else -flip_qty
                pos.entry_price = fill_price
                pos.leverage = leverage
                pos.margin_type = margin_type
                if margin_type == MarginType.ISOLATED:
                    margin_to_allocate = min(hold.isolated_reserved if hold else 0.0, flip_qty * fill_price / leverage)
                    pos.isolated_margin = margin_to_allocate
                    self.balance -= margin_to_allocate
                else:
                    pos.isolated_margin = 0.0
            else:
                pos.size += signed_qty

        if abs(pos.size) < 1e-12:
            pos.size = 0.0
            pos.entry_price = 0.0
            pos.isolated_margin = 0.0

        if hold is not None:
            hold.remaining_qty = max(0.0, hold.remaining_qty - fill_qty)
            if hold.opening_qty_reserved > 0:
                opening_consumed = min(fill_qty, hold.opening_qty_reserved)
                if hold.margin_type == MarginType.CROSS:
                    reserve_release = hold.cross_reserved * (opening_consumed / hold.opening_qty_reserved) if hold.opening_qty_reserved > 0 else 0.0
                    hold.cross_reserved -= reserve_release
                else:
                    reserve_release = hold.isolated_reserved * (opening_consumed / hold.opening_qty_reserved) if hold.opening_qty_reserved > 0 else 0.0
                    hold.isolated_reserved -= reserve_release
                hold.opening_qty_reserved -= opening_consumed
            if hold.remaining_qty <= 1e-12:
                self.order_holds.pop(hold.order_id, None)

    def apply_funding(self, symbol: str, payment: float) -> None:
        pos = self.positions.get(symbol)
        if pos is None or pos.size == 0:
            return
        if pos.is_isolated:
            pos.isolated_margin -= payment
        else:
            self.balance -= payment
        self.total_funding_paid += payment

    def settle_position(self, symbol: str, mark_price: float) -> float:
        pos = self.positions.get(symbol)
        if pos is None or pos.size == 0:
            return 0.0
        pnl = pos.unrealized_pnl(mark_price)
        self.balance += pnl
        self.total_realized_pnl += pnl
        if pos.is_isolated:
            self.balance += pos.isolated_margin
            pos.isolated_margin = 0.0
        pos.size = 0.0
        pos.entry_price = 0.0
        return pnl

"""Perpetual futures account with cross and isolated margin support.

Models a single agent's account on a HIP-3 DEX.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from util.ContractSpec import MarginMode, MarginTable


@dataclass
class Position:
    size: float = 0.0              # positive = long, negative = short
    entry_price: float = 0.0
    leverage: int = 10
    is_isolated: bool = False
    isolated_margin: float = 0.0   # only used when is_isolated=True

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_short(self):
        return self.size < 0

    @property
    def abs_size(self):
        return abs(self.size)

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.size * (mark_price - self.entry_price)

    def notional_value(self, mark_price: float) -> float:
        return self.abs_size * mark_price


class PerpAccount:
    """A single agent's account on the HIP-3 DEX."""

    def __init__(self, starting_balance: float = 100000.0):
        self.balance: float = starting_balance
        self.starting_balance: float = starting_balance
        self.positions: Dict[str, Position] = {}
        self.total_fees_paid: float = 0.0
        self.total_funding_paid: float = 0.0
        self.total_realized_pnl: float = 0.0

    def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position())

    def get_position_size(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.size if pos else 0.0

    def has_position(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.size != 0.0

    # ── Account valuation ───────────────────────────────────────────────

    def cross_account_value(self, mark_prices: Dict[str, float]) -> float:
        """Balance + unrealized PnL on all cross positions."""
        value = self.balance
        for sym, pos in self.positions.items():
            if not pos.is_isolated:
                mp = mark_prices.get(sym, pos.entry_price)
                value += pos.unrealized_pnl(mp)
        return value

    def cross_maintenance_margin(self, mark_prices: Dict[str, float],
                                  margin_tables: Dict[str, MarginTable]) -> float:
        """Total maintenance margin required for all cross positions."""
        total = 0.0
        for sym, pos in self.positions.items():
            if pos.is_isolated or pos.size == 0:
                continue
            mp = mark_prices.get(sym, pos.entry_price)
            notional = pos.notional_value(mp)
            mt = margin_tables.get(sym)
            if mt:
                total += mt.get_maintenance_margin(notional)
            else:
                total += notional * 0.05  # fallback 5%
        return total

    def isolated_account_value(self, symbol: str, mark_price: float) -> float:
        """Isolated margin + unrealized PnL for a specific isolated position."""
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_isolated:
            return 0.0
        return pos.isolated_margin + pos.unrealized_pnl(mark_price)

    def isolated_maintenance_margin(self, symbol: str, mark_price: float,
                                     margin_table: Optional[MarginTable] = None) -> float:
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_isolated or pos.size == 0:
            return 0.0
        notional = pos.notional_value(mark_price)
        if margin_table:
            return margin_table.get_maintenance_margin(notional)
        return notional * 0.05

    # ── Margin checks ───────────────────────────────────────────────────

    def check_initial_margin(self, symbol: str, additional_notional: float,
                              leverage: int, mark_prices: Dict[str, float],
                              margin_tables: Dict[str, MarginTable],
                              is_isolated: bool = False) -> bool:
        """Check if the agent has sufficient margin to open/increase a position."""
        required_margin = additional_notional / leverage
        if is_isolated:
            return self.balance >= required_margin
        else:
            available = self.cross_account_value(mark_prices) - self.cross_maintenance_margin(mark_prices, margin_tables)
            return available >= required_margin

    # ── Order execution accounting ──────────────────────────────────────

    def apply_fill(self, symbol: str, fill_qty: float, fill_price: float,
                   is_buy: bool, leverage: int, fee: float,
                   is_isolated: bool = False, margin_mode: MarginMode = MarginMode.NORMAL):
        """Apply a fill to this account. Handles position open, increase, reduce, close, and flip."""
        signed_qty = fill_qty if is_buy else -fill_qty
        self.balance -= fee
        self.total_fees_paid += fee

        pos = self.positions.get(symbol)
        if pos is None:
            pos = Position(leverage=leverage, is_isolated=is_isolated)
            self.positions[symbol] = pos

        if pos.size == 0:
            # Opening a new position
            pos.size = signed_qty
            pos.entry_price = fill_price
            pos.leverage = leverage
            pos.is_isolated = is_isolated
            if is_isolated:
                margin_needed = abs(signed_qty) * fill_price / leverage
                pos.isolated_margin = margin_needed
                self.balance -= margin_needed
        elif (pos.size > 0 and is_buy) or (pos.size < 0 and not is_buy):
            # Increasing existing position: weighted average entry
            old_notional = abs(pos.size) * pos.entry_price
            new_notional = fill_qty * fill_price
            total_size = abs(pos.size) + fill_qty
            pos.entry_price = (old_notional + new_notional) / total_size if total_size > 0 else fill_price
            pos.size += signed_qty
            if is_isolated:
                additional_margin = fill_qty * fill_price / leverage
                pos.isolated_margin += additional_margin
                self.balance -= additional_margin
        else:
            # Reducing, closing, or flipping
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
                # Fully closed
                pos.size = 0.0
                pos.entry_price = 0.0
                pos.isolated_margin = 0.0
            elif remaining_after_reduce <= 1e-12 and flip_qty > 0:
                # Flipped to other side
                pos.size = flip_qty if is_buy else -flip_qty
                pos.entry_price = fill_price
                if is_isolated:
                    margin_needed = flip_qty * fill_price / leverage
                    pos.isolated_margin = margin_needed
                    self.balance -= margin_needed
                else:
                    pos.isolated_margin = 0.0
            else:
                # Partially reduced
                pos.size += signed_qty

        # Clean up zero positions
        if abs(pos.size) < 1e-12:
            pos.size = 0.0
            pos.entry_price = 0.0
            pos.isolated_margin = 0.0

    def apply_funding(self, symbol: str, payment: float):
        """Apply a funding payment. Positive = agent pays, negative = agent receives."""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        if pos.is_isolated:
            pos.isolated_margin -= payment
        else:
            self.balance -= payment
        self.total_funding_paid += payment

    def settle_position(self, symbol: str, mark_price: float):
        """Settle a position at mark price (used by haltTrading). Returns realized PnL."""
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

    def total_equity(self, mark_prices: Dict[str, float]) -> float:
        """Total account equity including all positions."""
        equity = self.balance
        for sym, pos in self.positions.items():
            mp = mark_prices.get(sym, pos.entry_price)
            equity += pos.unrealized_pnl(mp)
            if pos.is_isolated:
                equity += pos.isolated_margin
        return equity

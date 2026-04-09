"""Clearinghouse: manages all PerpAccounts and enforces margin rules across the HIP-3 DEX."""

from typing import Dict, Optional, List, Tuple
from util.PerpAccount import PerpAccount, Position
from util.ContractSpec import ContractSpec, MarginTable, MarginMode, FeeSchedule


class Clearinghouse:

    def __init__(self, contract_specs: Dict[str, ContractSpec], fee_schedule: FeeSchedule):
        self.accounts: Dict[int, PerpAccount] = {}   # agent_id -> PerpAccount
        self.contract_specs = contract_specs
        self.fee_schedule = fee_schedule
        self.deployer_fees_collected = 0.0
        self.protocol_fees_collected = 0.0

        # Aggregate OI tracking per symbol
        self.open_interest_size: Dict[str, float] = {s: 0.0 for s in contract_specs}
        self.open_interest_notional: Dict[str, float] = {s: 0.0 for s in contract_specs}

    def get_or_create_account(self, agent_id: int, starting_balance: float = 100000.0) -> PerpAccount:
        if agent_id not in self.accounts:
            self.accounts[agent_id] = PerpAccount(starting_balance)
        return self.accounts[agent_id]

    def get_account(self, agent_id: int) -> Optional[PerpAccount]:
        return self.accounts.get(agent_id)

    def get_margin_table(self, symbol: str) -> Optional[MarginTable]:
        spec = self.contract_specs.get(symbol)
        return spec.margin_table if spec else None

    def get_margin_tables(self) -> Dict[str, MarginTable]:
        return {s: spec.margin_table for s, spec in self.contract_specs.items() if spec.margin_table}

    def get_margin_mode(self, symbol: str) -> MarginMode:
        spec = self.contract_specs.get(symbol)
        return spec.margin_mode if spec else MarginMode.NORMAL

    # ── Fill processing ─────────────────────────────────────────────────

    def process_fill(self, agent_id: int, symbol: str, fill_qty: float,
                     fill_price: float, is_buy: bool, is_taker: bool,
                     leverage: int = None, is_isolated: bool = False,
                     is_liquidation: bool = False) -> float:
        """Process a fill for an agent. Returns the fee charged."""
        account = self.get_or_create_account(agent_id)
        spec = self.contract_specs.get(symbol)
        margin_mode = spec.margin_mode if spec else MarginMode.NORMAL

        if leverage is None:
            leverage = spec.default_leverage if spec else 10

        # Force isolated if margin mode requires it
        if margin_mode in (MarginMode.NO_CROSS, MarginMode.STRICT_ISOLATED):
            is_isolated = True

        # Compute fee
        notional = fill_qty * fill_price
        if is_liquidation:
            fee = 0.0
        elif is_taker:
            fee = notional * self.fee_schedule.taker_fee_rate
        else:
            fee = notional * self.fee_schedule.maker_fee_rate

        # Split fees
        deployer_fee = fee * self.fee_schedule.deployer_share
        protocol_fee = fee * self.fee_schedule.protocol_share
        self.deployer_fees_collected += deployer_fee
        self.protocol_fees_collected += protocol_fee

        # Track OI change: compute old position, apply fill, compute new position
        old_pos = account.get_position(symbol)
        old_abs_size = abs(old_pos.size) if old_pos.size != 0 else 0.0

        account.apply_fill(symbol, fill_qty, fill_price, is_buy, leverage,
                           fee, is_isolated, margin_mode)

        new_pos = account.get_position(symbol)
        new_abs_size = abs(new_pos.size)

        # Update aggregate OI
        oi_delta = new_abs_size - old_abs_size
        self.open_interest_size[symbol] = self.open_interest_size.get(symbol, 0.0) + oi_delta

        return fee

    # ── OI cap check ────────────────────────────────────────────────────

    def check_oi_cap(self, symbol: str, additional_size: float, mark_price: float) -> bool:
        """Check if an order would breach OI caps. Returns True if allowed."""
        spec = self.contract_specs.get(symbol)
        if spec is None:
            return True

        new_oi_size = self.open_interest_size.get(symbol, 0.0) + additional_size
        if new_oi_size > spec.oi_cap_size:
            return False

        new_oi_notional = new_oi_size * mark_price
        if new_oi_notional > spec.oi_cap_notional:
            return False

        return True

    def recalculate_oi(self, symbol: str, mark_price: float):
        """Recalculate OI from all accounts (for consistency checks)."""
        total_long = 0.0
        total_short = 0.0
        for account in self.accounts.values():
            pos = account.get_position(symbol)
            if pos.size > 0:
                total_long += pos.size
            elif pos.size < 0:
                total_short += abs(pos.size)
        self.open_interest_size[symbol] = max(total_long, total_short)
        self.open_interest_notional[symbol] = self.open_interest_size[symbol] * mark_price

    # ── Liquidation scanning ────────────────────────────────────────────

    def get_liquidatable_accounts(self, mark_prices: Dict[str, float]) -> List[Tuple[int, str, str]]:
        """Scan all accounts for liquidatable positions.
        
        Returns list of (agent_id, symbol, liquidation_type) where
        liquidation_type is 'market' or 'backstop'.
        """
        margin_tables = self.get_margin_tables()
        results = []

        for agent_id, account in self.accounts.items():
            # Check cross positions
            cross_value = account.cross_account_value(mark_prices)
            cross_mm = account.cross_maintenance_margin(mark_prices, margin_tables)

            if cross_mm > 0 and cross_value < cross_mm:
                liq_type = 'backstop' if cross_value < cross_mm * (2.0 / 3.0) else 'market'
                for sym, pos in account.positions.items():
                    if not pos.is_isolated and pos.size != 0:
                        results.append((agent_id, sym, liq_type))

            # Check isolated positions
            for sym, pos in account.positions.items():
                if not pos.is_isolated or pos.size == 0:
                    continue
                mp = mark_prices.get(sym, pos.entry_price)
                mt = margin_tables.get(sym)
                iso_value = account.isolated_account_value(sym, mp)
                iso_mm = account.isolated_maintenance_margin(sym, mp, mt)

                if iso_mm > 0 and iso_value < iso_mm:
                    liq_type = 'backstop' if iso_value < iso_mm * (2.0 / 3.0) else 'market'
                    results.append((agent_id, sym, liq_type))

        return results

    # ── Settlement ──────────────────────────────────────────────────────

    def settle_all(self, symbol: str, mark_price: float) -> Dict[int, float]:
        """Settle all positions in a symbol at mark price. Returns dict of agent_id -> realized PnL."""
        results = {}
        for agent_id, account in self.accounts.items():
            pnl = account.settle_position(symbol, mark_price)
            if pnl != 0:
                results[agent_id] = pnl
        self.open_interest_size[symbol] = 0.0
        self.open_interest_notional[symbol] = 0.0
        return results

    # ── Dynamic deployer updates ────────────────────────────────────────

    def update_margin_table(self, symbol: str, margin_table: MarginTable):
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.margin_table = margin_table

    def update_oi_caps(self, symbol: str, notional_cap: float, size_cap: float):
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.oi_cap_notional = notional_cap
            spec.oi_cap_size = size_cap

    def update_funding_multiplier(self, symbol: str, multiplier: float):
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.funding_multiplier = max(0.0, min(10.0, multiplier))

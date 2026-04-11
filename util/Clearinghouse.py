"""Canonical clearinghouse for the HIP-3 perp exchange."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from util.ContractSpec import ContractSpec, MarginMode, MarginTable, MarginType, PerpDexConfig
from util.FeeEngine import FeeEngine, FeeProfile
from util.PerpAccount import OrderHold, PerpAccount, Position


@dataclass
class HoldPreview:
    leverage: int
    margin_type: MarginType
    remaining_qty: float
    cross_reserved: float
    isolated_reserved: float
    opening_qty_reserved: float
    placement_price: float


class Clearinghouse:
    def __init__(
        self,
        dex_config: PerpDexConfig,
        starting_balances: Optional[Dict[int, float]] = None,
        fee_profiles: Optional[Dict[int, FeeProfile]] = None,
    ):
        self.accounts: Dict[int, PerpAccount] = {}
        self.dex_config = dex_config
        self.contract_specs = dex_config.assets
        self.margin_tables = dict(dex_config.margin_tables)
        self.deployer_fees_collected = 0.0
        self.protocol_fees_collected = 0.0
        self.open_interest_size: Dict[str, float] = {symbol: 0.0 for symbol in self.contract_specs}
        self.open_interest_notional: Dict[str, float] = {symbol: 0.0 for symbol in self.contract_specs}
        self.total_dex_open_interest_notional = 0.0
        self.fee_engine = FeeEngine()
        self.fee_engine.bootstrap_profiles(fee_profiles or {})
        self.current_day = None
        self._margin_tables_by_symbol: Dict[str, MarginTable] = {
            symbol: self._resolve_margin_table(symbol) for symbol in self.contract_specs
        }

        if starting_balances:
            self.bootstrap_accounts(starting_balances)

    def bootstrap_accounts(self, starting_balances: Dict[int, float]) -> None:
        for agent_id, balance in starting_balances.items():
            self.accounts[agent_id] = PerpAccount(starting_balance=balance)
            self.fee_engine.get_or_create_profile(agent_id)

    def bootstrap_fee_profiles(self, fee_profiles: Dict[int, FeeProfile]) -> None:
        self.fee_engine.bootstrap_profiles(fee_profiles)

    def sync_time(self, current_time: pd.Timestamp) -> None:
        current_day = current_time.date()
        if self.current_day is None or current_day > self.current_day:
            self.current_day = current_day
            self.fee_engine.roll_profiles_to_day(current_day)

    def get_or_create_account(self, agent_id: int, starting_balance: float = 0.0) -> PerpAccount:
        if agent_id not in self.accounts:
            self.accounts[agent_id] = PerpAccount(starting_balance=starting_balance)
        self.fee_engine.get_or_create_profile(agent_id)
        return self.accounts[agent_id]

    def get_account(self, agent_id: int) -> Optional[PerpAccount]:
        return self.accounts.get(agent_id)

    def get_margin_table(self, symbol: str) -> Optional[MarginTable]:
        return self._margin_tables_by_symbol.get(symbol)

    def _resolve_margin_table(self, symbol: str) -> Optional[MarginTable]:
        spec = self.contract_specs.get(symbol)
        if spec is None:
            return None
        if spec.margin_table_id in self.margin_tables:
            return self.margin_tables[spec.margin_table_id]
        return spec.margin_table

    def get_margin_tables(self) -> Dict[str, MarginTable]:
        return self._margin_tables_by_symbol

    def _refresh_margin_table_cache(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._margin_tables_by_symbol = {
                name: self._resolve_margin_table(name) for name in self.contract_specs
            }
            return
        self._margin_tables_by_symbol[symbol] = self._resolve_margin_table(symbol)

    def get_margin_mode(self, symbol: str) -> MarginMode:
        spec = self.contract_specs.get(symbol)
        return spec.margin_mode if spec else MarginMode.NORMAL

    def get_or_create_position(self, agent_id: int, symbol: str, leverage: int, margin_type: MarginType) -> Position:
        account = self.get_or_create_account(agent_id)
        return account.ensure_position(symbol, leverage=leverage, margin_type=margin_type)

    def _resolve_margin_type(self, symbol: str, requested: MarginType, current_position: Position) -> MarginType:
        margin_mode = self.get_margin_mode(symbol)
        if margin_mode in (MarginMode.NO_CROSS, MarginMode.STRICT_ISOLATED):
            return MarginType.ISOLATED
        if requested == MarginType.INHERIT:
            if current_position.size != 0:
                return current_position.margin_type
            return MarginType.CROSS
        return requested

    def preview_hold(
        self,
        agent_id: int,
        order,
        mark_prices: Dict[str, float],
        exclude_order_id: Optional[int] = None,
    ) -> Optional[HoldPreview]:
        account = self.get_or_create_account(agent_id)
        return self._preview_hold_values(
            account=account,
            symbol=order.symbol,
            quantity=order.quantity,
            is_buy_order=order.is_buy_order,
            limit_price=order.limit_price,
            is_market_order=order.is_market_order,
            reduce_only=order.reduce_only,
            requested_leverage=order.requested_leverage,
            requested_margin_type=order.margin_type,
            mark_prices=mark_prices,
            exclude_order_id=exclude_order_id,
        )

    def _preview_hold_values(
        self,
        account: PerpAccount,
        symbol: str,
        quantity: float,
        is_buy_order: bool,
        limit_price: float,
        is_market_order: bool,
        reduce_only: bool,
        requested_leverage,
        requested_margin_type: MarginType,
        mark_prices: Dict[str, float],
        exclude_order_id: Optional[int] = None,
    ) -> Optional[HoldPreview]:
        spec = self.contract_specs.get(symbol)
        if spec is None:
            return None

        position = account.get_position(symbol)
        reference_mark_price = mark_prices.get(symbol, spec.initial_oracle_px)
        leverage = int(requested_leverage or (position.leverage if position.size != 0 else spec.default_leverage))
        margin_table = self.get_margin_table(symbol)
        leverage = max(
            1,
            min(leverage, margin_table.get_max_leverage(quantity * max(limit_price, reference_mark_price))),
        )
        margin_type = self._resolve_margin_type(symbol, requested_margin_type, position)
        opening_qty = account.estimate_opening_qty(symbol, quantity, is_buy_order)
        if reduce_only or opening_qty <= 0:
            return HoldPreview(
                leverage=leverage,
                margin_type=margin_type,
                remaining_qty=quantity,
                cross_reserved=0.0,
                isolated_reserved=0.0,
                opening_qty_reserved=0.0,
                placement_price=limit_price,
            )

        placement_reference = reference_mark_price if is_market_order else limit_price
        initial_margin = opening_qty * placement_reference / leverage

        if margin_type == MarginType.CROSS:
            available = account.cross_available_margin(
                mark_prices,
                self._margin_tables_by_symbol,
                exclude_order_id=exclude_order_id,
            )
            if available < initial_margin - 1e-12:
                return None
            return HoldPreview(
                leverage=leverage,
                margin_type=margin_type,
                remaining_qty=quantity,
                cross_reserved=initial_margin,
                isolated_reserved=0.0,
                opening_qty_reserved=opening_qty,
                placement_price=placement_reference,
            )

        available_cash = account.available_cash(exclude_order_id=exclude_order_id)
        if available_cash < initial_margin - 1e-12:
            return None
        return HoldPreview(
            leverage=leverage,
            margin_type=margin_type,
            remaining_qty=quantity,
            cross_reserved=0.0,
            isolated_reserved=initial_margin,
            opening_qty_reserved=opening_qty,
            placement_price=placement_reference,
        )

    def place_hold(self, agent_id: int, order, mark_prices: Dict[str, float]) -> Tuple[bool, Optional[str], Optional[HoldPreview]]:
        preview = self.preview_hold(agent_id, order, mark_prices)
        if preview is None:
            return False, "INSUFFICIENT_MARGIN", None

        if preview.opening_qty_reserved <= 0:
            return True, None, preview

        account = self.get_or_create_account(agent_id)
        account.add_hold(
            OrderHold(
                order_id=order.order_id,
                symbol=order.symbol,
                is_buy_order=order.is_buy_order,
                leverage=preview.leverage,
                margin_type=preview.margin_type,
                remaining_qty=preview.remaining_qty,
                cross_reserved=preview.cross_reserved,
                isolated_reserved=preview.isolated_reserved,
                opening_qty_reserved=preview.opening_qty_reserved,
                placement_price=preview.placement_price,
            )
        )
        return True, None, preview

    def replace_hold(self, agent_id: int, order, mark_prices: Dict[str, float]) -> Tuple[bool, Optional[str], Optional[HoldPreview]]:
        preview = self.preview_hold(agent_id, order, mark_prices, exclude_order_id=order.order_id)
        if preview is None:
            return False, "INSUFFICIENT_MARGIN", None
        account = self.get_or_create_account(agent_id)
        if order.order_id in account.order_holds:
            account.resize_hold(
                order.order_id,
                remaining_qty=preview.remaining_qty,
                cross_reserved=preview.cross_reserved,
                isolated_reserved=preview.isolated_reserved,
                opening_qty_reserved=preview.opening_qty_reserved,
            )
        elif preview.opening_qty_reserved > 0:
            account.add_hold(
                OrderHold(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    is_buy_order=order.is_buy_order,
                    leverage=preview.leverage,
                    margin_type=preview.margin_type,
                    remaining_qty=preview.remaining_qty,
                    cross_reserved=preview.cross_reserved,
                    isolated_reserved=preview.isolated_reserved,
                    opening_qty_reserved=preview.opening_qty_reserved,
                    placement_price=preview.placement_price,
                )
            )
        return True, None, preview

    def resize_hold_incremental(
        self,
        agent_id: int,
        order_id: int,
        new_quantity: float,
        limit_price: float,
        mark_prices: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """Resize an existing hold in place for a same-price amend.

        For size increases, checks that incremental margin is available.
        For size decreases, releases the excess margin.
        Returns (success, error_reason).
        """
        account = self.get_or_create_account(agent_id)
        hold = account.order_holds.get(order_id)
        if hold is None:
            return True, None

        position = account.get_position(hold.symbol)
        spec = self.contract_specs.get(hold.symbol)
        if spec is None:
            return False, "UNKNOWN_SYMBOL"

        new_opening_qty = account.estimate_opening_qty(hold.symbol, new_quantity, hold.is_buy_order)
        if new_opening_qty <= 0:
            account.resize_hold(order_id, remaining_qty=new_quantity, cross_reserved=0.0, isolated_reserved=0.0, opening_qty_reserved=0.0)
            return True, None

        placement_reference = limit_price
        new_margin = new_opening_qty * placement_reference / hold.leverage

        if hold.margin_type == MarginType.CROSS:
            current_reserved = hold.cross_reserved
            if new_margin > current_reserved + 1e-12:
                available = account.cross_available_margin(mark_prices, self._margin_tables_by_symbol, exclude_order_id=order_id)
                if available < new_margin - current_reserved - 1e-12:
                    return False, "INSUFFICIENT_MARGIN"
            account.resize_hold(order_id, remaining_qty=new_quantity, cross_reserved=new_margin, isolated_reserved=0.0, opening_qty_reserved=new_opening_qty)
        else:
            current_reserved = hold.isolated_reserved
            if new_margin > current_reserved + 1e-12:
                available = account.available_cash(exclude_order_id=order_id)
                if available < new_margin - current_reserved - 1e-12:
                    return False, "INSUFFICIENT_MARGIN"
            account.resize_hold(order_id, remaining_qty=new_quantity, cross_reserved=0.0, isolated_reserved=new_margin, opening_qty_reserved=new_opening_qty)

        return True, None

    def release_hold(self, agent_id: int, order_id: int) -> Optional[OrderHold]:
        account = self.get_or_create_account(agent_id)
        return account.release_hold(order_id)

    def ensure_fillable(self, order, fill_qty: float, mark_prices: Dict[str, float]) -> bool:
        if order.reduce_only or fill_qty <= 0:
            return True
        account = self.get_or_create_account(order.agent_id)
        spec = self.contract_specs.get(order.symbol)
        if spec is None:
            return False

        hold = account.order_holds.get(order.order_id)
        position = account.get_position(order.symbol)
        leverage = int(order.requested_leverage or (hold.leverage if hold else (position.leverage if position.size != 0 else spec.default_leverage)))
        margin_type = hold.margin_type if hold else self._resolve_margin_type(order.symbol, order.margin_type, position)
        preview = self._preview_hold_values(
            account=account,
            symbol=order.symbol,
            quantity=fill_qty,
            is_buy_order=order.is_buy_order,
            limit_price=mark_prices.get(order.symbol, spec.initial_oracle_px),
            is_market_order=True,
            reduce_only=order.reduce_only,
            requested_leverage=leverage,
            requested_margin_type=margin_type,
            mark_prices=mark_prices,
            exclude_order_id=order.order_id,
        )
        if preview is None:
            return False

        if hold is None:
            return True

        hold.leverage = preview.leverage
        hold.margin_type = preview.margin_type
        hold.cross_reserved = max(hold.cross_reserved, preview.cross_reserved)
        hold.isolated_reserved = max(hold.isolated_reserved, preview.isolated_reserved)
        hold.placement_price = preview.placement_price
        return True

    def check_oi_cap(self, symbol: str, additional_size: float, mark_price: float) -> bool:
        spec = self.contract_specs.get(symbol)
        if spec is None:
            return True

        new_oi_size = self.open_interest_size.get(symbol, 0.0) + additional_size
        if new_oi_size > spec.oi_cap_size + 1e-12:
            return False

        new_oi_notional = new_oi_size * mark_price
        if new_oi_notional > spec.oi_cap_notional + 1e-12:
            return False

        new_total_dex_oi = self.total_dex_open_interest_notional - self.open_interest_notional.get(symbol, 0.0) + new_oi_notional
        if new_total_dex_oi > self.dex_config.dex_open_interest_cap_notional + 1e-12:
            return False
        return True

    def process_fill(
        self,
        agent_id: int,
        order,
        fill_qty: float,
        fill_price: float,
        is_taker: bool,
        current_time: pd.Timestamp,
        fee_model: str = "hyperliquid",
    ) -> float:
        account = self.get_or_create_account(agent_id)
        spec = self.contract_specs.get(order.symbol)
        self.sync_time(current_time)

        position = account.get_position(order.symbol)
        hold = account.order_holds.get(order.order_id)
        leverage = hold.leverage if hold else int(order.requested_leverage or (position.leverage if position.size != 0 else spec.default_leverage))
        margin_type = hold.margin_type if hold else self._resolve_margin_type(order.symbol, order.margin_type, position)

        notional = fill_qty * fill_price
        if order.is_liquidation:
            fee = 0.0
        elif fee_model == "flat":
            fee = notional * (self.dex_config.fee_schedule.taker_fee_rate if is_taker else self.dex_config.fee_schedule.maker_fee_rate)
        else:
            fee_rate = self.fee_engine.get_fee_rate(
                agent_id,
                is_maker=not is_taker,
                deployer_fee_scale=self.dex_config.fee_scale,
                growth_mode=spec.growth_mode,
                is_aligned_quote=spec.aligned_quote,
            )
            fee = max(0.0, notional * fee_rate) if fee_rate >= 0 else notional * fee_rate

        splits = self.fee_engine.compute_fee_split(max(fee, 0.0), self.dex_config.fee_scale)
        self.deployer_fees_collected += splits["deployer"]
        self.protocol_fees_collected += splits["protocol"]
        self.fee_engine.record_trade(agent_id, notional, is_maker=not is_taker, day=current_time.date(), growth_mode=spec.growth_mode)

        account.apply_fill(
            order.symbol,
            fill_qty=fill_qty,
            fill_price=fill_price,
            is_buy=order.is_buy_order,
            leverage=leverage,
            fee=fee,
            margin_type=margin_type,
            order_id=order.order_id,
            margin_mode=spec.margin_mode,
        )

        return fee

    def recalculate_symbol_oi(self, symbol: str, mark_price: Optional[float] = None) -> None:
        total_long = 0.0
        total_short = 0.0

        for account in self.accounts.values():
            pos = account.positions.get(symbol)
            if pos is None:
                continue
            if pos.size > 0:
                total_long += pos.size
            elif pos.size < 0:
                total_short += abs(pos.size)

        oi_size = max(total_long, total_short)
        if oi_size <= 1e-12:
            oi_size = 0.0

        if mark_price is None:
            previous_size = self.open_interest_size.get(symbol, 0.0)
            if previous_size > 1e-12:
                mark_price = self.open_interest_notional.get(symbol, 0.0) / previous_size
            else:
                mark_price = self.contract_specs[symbol].initial_oracle_px

        previous_notional = self.open_interest_notional.get(symbol, 0.0)
        oi_notional = oi_size * mark_price
        self.open_interest_size[symbol] = oi_size
        self.open_interest_notional[symbol] = oi_notional
        self.total_dex_open_interest_notional += oi_notional - previous_notional

    def refresh_open_interest_notional(self, symbol: str, mark_price: float) -> None:
        previous_notional = self.open_interest_notional.get(symbol, 0.0)
        oi_notional = self.open_interest_size.get(symbol, 0.0) * mark_price
        self.open_interest_notional[symbol] = oi_notional
        self.total_dex_open_interest_notional += oi_notional - previous_notional

    def recalculate_oi(self, mark_prices: Dict[str, float]) -> None:
        total_dex_notional = 0.0
        for symbol in self.contract_specs:
            total_long = 0.0
            total_short = 0.0
            for account in self.accounts.values():
                pos = account.positions.get(symbol)
                if pos is None:
                    continue
                if pos.size > 0:
                    total_long += pos.size
                elif pos.size < 0:
                    total_short += abs(pos.size)
            oi_size = max(total_long, total_short)
            oi_notional = oi_size * mark_prices.get(symbol, self.contract_specs[symbol].initial_oracle_px)
            self.open_interest_size[symbol] = oi_size
            self.open_interest_notional[symbol] = oi_notional
            total_dex_notional += oi_notional
        self.total_dex_open_interest_notional = total_dex_notional

    def get_liquidatable_accounts(self, mark_prices: Dict[str, float]) -> List[Tuple[int, str, str]]:
        results = []
        for agent_id, account in self.accounts.items():
            if not account.positions:
                continue

            cross_value = account.balance
            cross_mm = 0.0
            cross_symbols = []

            for symbol, pos in account.positions.items():
                if pos.size == 0:
                    continue

                spec = self.contract_specs.get(symbol)
                if spec is None:
                    continue

                mark_price = mark_prices.get(symbol, pos.entry_price)
                margin_table = self.margin_tables.get(spec.margin_table_id, spec.margin_table)
                maintenance_margin = (
                    margin_table.get_maintenance_margin(pos.notional_value(mark_price))
                    if margin_table is not None
                    else pos.notional_value(mark_price) * 0.05
                )

                if pos.margin_type == MarginType.CROSS:
                    cross_value += pos.unrealized_pnl(mark_price)
                    cross_mm += maintenance_margin
                    cross_symbols.append(symbol)
                    continue

                iso_value = pos.isolated_margin + pos.unrealized_pnl(mark_price)
                if maintenance_margin > 0 and iso_value < maintenance_margin:
                    liq_type = "backstop" if iso_value < maintenance_margin * (2.0 / 3.0) else "market"
                    results.append((agent_id, symbol, liq_type))

            if cross_mm > 0 and cross_value < cross_mm:
                liq_type = "backstop" if cross_value < cross_mm * (2.0 / 3.0) else "market"
                for symbol in cross_symbols:
                    results.append((agent_id, symbol, liq_type))
        return results

    def settle_all(self, symbol: str, mark_price: float) -> Dict[int, float]:
        results = {}
        for agent_id, account in self.accounts.items():
            pnl = account.settle_position(symbol, mark_price)
            if pnl != 0:
                results[agent_id] = pnl
            for hold_id in [hold.order_id for hold in account.order_holds.values() if hold.symbol == symbol]:
                account.release_hold(hold_id)
        self.open_interest_size[symbol] = 0.0
        self.open_interest_notional[symbol] = 0.0
        self.total_dex_open_interest_notional = sum(self.open_interest_notional.values())
        return results

    def update_margin_table(self, symbol: str, margin_table: MarginTable):
        self.margin_tables[margin_table.table_id] = margin_table
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.margin_table = margin_table
            spec.margin_table_id = margin_table.table_id
            self._refresh_margin_table_cache(symbol)

    def insert_margin_table(self, margin_table: MarginTable) -> None:
        self.margin_tables[margin_table.table_id] = margin_table

    def update_margin_table_id(self, symbol: str, margin_table_id: int) -> None:
        spec = self.contract_specs.get(symbol)
        if spec and margin_table_id in self.margin_tables:
            spec.margin_table_id = margin_table_id
            spec.margin_table = self.margin_tables[margin_table_id]
            self._refresh_margin_table_cache(symbol)

    def update_oi_caps(self, symbol: str, notional_cap: float, size_cap: float):
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.oi_cap_notional = notional_cap
            spec.oi_cap_size = size_cap

    def update_funding_multiplier(self, symbol: str, multiplier: float):
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.funding_multiplier = max(0.0, min(10.0, multiplier))

    def update_funding_interest_rate(self, symbol: str, interest_rate_8h: float):
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.funding_interest_rate_8h = max(-0.01, min(0.01, interest_rate_8h))

    def update_fee_scale(self, fee_scale: float) -> None:
        self.dex_config.fee_scale = max(0.0, min(3.0, fee_scale))

    def update_growth_mode(self, symbol: str, growth_mode: bool) -> None:
        spec = self.contract_specs.get(symbol)
        if spec:
            spec.growth_mode = bool(growth_mode)

    def transfer_to_backstop(self, agent_id: int, symbol: str, liq_type: str) -> dict:
        account = self.get_or_create_account(agent_id)
        position = account.get_position(symbol)
        if position.size == 0:
            return {}

        payload = {
            "symbol": symbol,
            "size": position.size,
            "entry_price": position.entry_price,
            "margin_type": position.margin_type,
            "isolated_margin": position.isolated_margin,
            "balance": account.balance,
        }

        if position.margin_type == MarginType.CROSS:
            for pos_symbol, pos in list(account.positions.items()):
                if pos.margin_type == MarginType.CROSS:
                    pos.size = 0.0
                    pos.entry_price = 0.0
                    pos.isolated_margin = 0.0
            account.balance = 0.0
        else:
            account.positions[symbol] = Position()

        for hold_id in [hold.order_id for hold in account.order_holds.values() if hold.symbol == symbol]:
            account.release_hold(hold_id)

        return payload

    def apply_adl(self, underwater_agent_id: int, symbol: str, mark_price: float, previous_mark_price: float) -> List[dict]:
        underwater_account = self.get_or_create_account(underwater_agent_id)
        underwater_pos = underwater_account.get_position(symbol)
        if underwater_pos.size == 0:
            return []

        needed_equity = -underwater_account.total_equity({symbol: mark_price})
        if needed_equity <= 0:
            return []

        candidates = []
        for agent_id, account in self.accounts.items():
            if agent_id == underwater_agent_id:
                continue
            pos = account.get_position(symbol)
            if pos.size == 0 or pos.size * underwater_pos.size >= 0:
                continue
            account_value = max(account.total_equity({symbol: mark_price}), 1e-9)
            upnl = pos.unrealized_pnl(mark_price)
            if upnl <= 0:
                continue
            notional = pos.notional_value(mark_price)
            adl_index = (mark_price / max(pos.entry_price, 1e-9)) * (notional / account_value)
            candidates.append((adl_index, agent_id, pos))

        candidates.sort(reverse=True)
        actions = []

        for _, agent_id, pos in candidates:
            if needed_equity <= 0:
                break
            close_qty = min(abs(pos.size), abs(underwater_pos.size))
            if close_qty <= 0:
                continue

            maker_side_buy = pos.size < 0
            close_price = previous_mark_price
            self.get_or_create_account(agent_id).apply_fill(
                symbol,
                fill_qty=close_qty,
                fill_price=close_price,
                is_buy=maker_side_buy,
                leverage=pos.leverage,
                fee=0.0,
                margin_type=pos.margin_type,
                order_id=None,
            )
            underwater_account.apply_fill(
                symbol,
                fill_qty=close_qty,
                fill_price=close_price,
                is_buy=(underwater_pos.size < 0),
                leverage=underwater_pos.leverage,
                fee=0.0,
                margin_type=underwater_pos.margin_type,
                order_id=None,
            )
            needed_equity = max(0.0, -underwater_account.total_equity({symbol: mark_price}))
            actions.append({"counterparty": agent_id, "quantity": close_qty, "price": close_price})

        # No-bad-debt invariant: if the underwater account still has negative equity
        # after exhausting all ADL candidates, zero out any remaining position and
        # absorb the loss (set balance to 0).
        underwater_pos = underwater_account.get_position(symbol)
        if underwater_pos.size != 0 and underwater_account.total_equity({symbol: mark_price}) < -1e-12:
            underwater_account.settle_position(symbol, mark_price)
        if underwater_account.balance < -1e-12:
            underwater_account.balance = 0.0

        return actions

"""Hyperliquid liquidation engine.

- Triggered on every mark price update.
- Positions below maintenance margin are sent as market liquidation orders to the book.
- Partial liquidation (20%) for positions > 100k USDC notional, with 30s cooldown.
- Backstop at 2/3 maintenance margin: transfer to liquidator vault.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd

PARTIAL_LIQUIDATION_THRESHOLD = 100_000.0  # 100k USDC
PARTIAL_LIQUIDATION_FRACTION = 0.20
COOLDOWN_SECONDS = 30


class LiquidationEngine:

    def __init__(self):
        self.cooldowns: Dict[int, pd.Timestamp] = {}  # agent_id -> cooldown_expiry

    def get_liquidation_orders(self, liquidatable: List[Tuple[int, str, str]],
                                positions: Dict[int, Dict[str, float]],
                                mark_prices: Dict[str, float],
                                current_time: pd.Timestamp) -> List[dict]:
        """Determine liquidation orders to send to the book.
        
        Args:
            liquidatable: List of (agent_id, symbol, liq_type) from clearinghouse scan.
            positions: Dict of agent_id -> {symbol: position_size}.
            mark_prices: Current mark prices per symbol.
            current_time: Current simulation time.
            
        Returns:
            List of dicts with keys: agent_id, symbol, quantity, is_buy, liq_type.
        """
        orders = []
        seen = set()

        for agent_id, symbol, liq_type in liquidatable:
            key = (agent_id, symbol)
            if key in seen:
                continue
            seen.add(key)

            agent_positions = positions.get(agent_id, {})
            pos_size = agent_positions.get(symbol, 0.0)
            if pos_size == 0:
                continue

            mp = mark_prices.get(symbol, 0.0)
            notional = abs(pos_size) * mp

            if liq_type == 'backstop':
                # Backstop: full position transfer (handled differently)
                orders.append({
                    'agent_id': agent_id,
                    'symbol': symbol,
                    'quantity': abs(pos_size),
                    'is_buy': pos_size < 0,  # close a short = buy, close a long = sell
                    'liq_type': 'backstop',
                })
                continue

            # Market liquidation
            # Check cooldown
            in_cooldown = False
            if agent_id in self.cooldowns:
                if current_time < self.cooldowns[agent_id]:
                    in_cooldown = True

            if notional > PARTIAL_LIQUIDATION_THRESHOLD and not in_cooldown:
                liq_qty = abs(pos_size) * PARTIAL_LIQUIDATION_FRACTION
                self.cooldowns[agent_id] = current_time + pd.Timedelta(seconds=30)
            else:
                liq_qty = abs(pos_size)
                if agent_id in self.cooldowns:
                    del self.cooldowns[agent_id]

            orders.append({
                'agent_id': agent_id,
                'symbol': symbol,
                'quantity': liq_qty,
                'is_buy': pos_size < 0,
                'liq_type': 'market',
            })

        return orders

    def clear_cooldown(self, agent_id: int):
        self.cooldowns.pop(agent_id, None)

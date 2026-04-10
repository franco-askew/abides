"""Hyperliquid liquidation helpers."""

from typing import Dict, List, Tuple

import pandas as pd


PARTIAL_LIQUIDATION_THRESHOLD = 100_000.0
PARTIAL_LIQUIDATION_FRACTION = 0.20
COOLDOWN_SECONDS = 30


class LiquidationEngine:
    def __init__(self):
        self.cooldowns: Dict[int, pd.Timestamp] = {}

    def get_liquidation_orders(
        self,
        liquidatable: List[Tuple[int, str, str]],
        positions: Dict[int, Dict[str, float]],
        mark_prices: Dict[str, float],
        current_time: pd.Timestamp,
    ) -> List[dict]:
        orders = []
        seen = set()
        for agent_id, symbol, liq_type in liquidatable:
            key = (agent_id, symbol)
            if key in seen:
                continue
            seen.add(key)
            pos_size = positions.get(agent_id, {}).get(symbol, 0.0)
            if pos_size == 0:
                continue

            notional = abs(pos_size) * mark_prices.get(symbol, 0.0)
            if liq_type == "backstop":
                orders.append(
                    {
                        "agent_id": agent_id,
                        "symbol": symbol,
                        "quantity": abs(pos_size),
                        "is_buy": pos_size < 0,
                        "liq_type": "backstop",
                    }
                )
                continue

            in_cooldown = agent_id in self.cooldowns and current_time < self.cooldowns[agent_id]
            if notional > PARTIAL_LIQUIDATION_THRESHOLD and not in_cooldown:
                liq_qty = abs(pos_size) * PARTIAL_LIQUIDATION_FRACTION
                self.cooldowns[agent_id] = current_time + pd.Timedelta(seconds=COOLDOWN_SECONDS)
            else:
                liq_qty = abs(pos_size)
                self.cooldowns.pop(agent_id, None)

            orders.append(
                {
                    "agent_id": agent_id,
                    "symbol": symbol,
                    "quantity": liq_qty,
                    "is_buy": pos_size < 0,
                    "liq_type": "market",
                }
            )
        return orders

"""Hyperliquid funding rate engine."""

from collections import defaultdict
from typing import Dict, List


MAX_FUNDING_PER_HOUR = 0.04


class FundingEngine:
    def __init__(self):
        self.premium_samples: Dict[str, List[float]] = defaultdict(list)
        self.last_funding_rates: Dict[str, float] = {}

    def sample_premium(self, symbol: str, oracle_price: float, impact_bid_px: float, impact_ask_px: float):
        if oracle_price <= 0:
            return
        impact_diff = 0.0
        if impact_bid_px is not None and impact_ask_px is not None:
            impact_diff = max(impact_bid_px - oracle_price, 0.0) - max(oracle_price - impact_ask_px, 0.0)
        premium = impact_diff / oracle_price
        self.premium_samples[symbol].append(premium)

    def compute_hourly_rate(self, symbol: str, funding_multiplier: float = 1.0, interest_rate_8h: float = 0.0001) -> float:
        samples = self.premium_samples.get(symbol, [])
        avg_premium = sum(samples) / len(samples) if samples else 0.0
        self.premium_samples[symbol] = []
        clamped = max(-0.0005, min(0.0005, interest_rate_8h - avg_premium))
        eight_hour_rate = (avg_premium + clamped) * funding_multiplier
        hourly_rate = max(-MAX_FUNDING_PER_HOUR, min(MAX_FUNDING_PER_HOUR, eight_hour_rate / 8.0))
        self.last_funding_rates[symbol] = hourly_rate
        return hourly_rate

    def compute_funding_payments(self, symbol: str, hourly_rate: float, oracle_price: float, positions: Dict[int, float]) -> Dict[int, float]:
        payments = {}
        for agent_id, size in positions.items():
            if size == 0:
                continue
            payments[agent_id] = size * oracle_price * hourly_rate
        return payments

    def get_last_rate(self, symbol: str) -> float:
        return self.last_funding_rates.get(symbol, 0.0)

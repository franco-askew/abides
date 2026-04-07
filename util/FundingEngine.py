"""Hyperliquid funding rate engine.

Exact replication of HL's funding formula:
  F = (avg(premium_samples) + clamp(0.0001 - avg(premium_samples), -0.0005, 0.0005)) * funding_multiplier
  premium = impact_price_diff / oracle_price
  impact_price_diff = max(impact_bid - oracle, 0) - max(oracle - impact_ask, 0)

- Premium sampled every 5 seconds, averaged over the hour.
- Settled hourly at 1/8 of the computed 8-hour rate.
- Payment = position_size * oracle_price * hourly_rate.
- Capped at 4% per hour.
"""

from typing import Dict, List, Tuple
from collections import defaultdict

INTEREST_RATE = 0.0001  # 0.01% per 8 hours
MAX_FUNDING_PER_HOUR = 0.04  # 4% cap


class FundingEngine:

    def __init__(self):
        self.premium_samples: Dict[str, List[float]] = defaultdict(list)
        self.last_funding_rates: Dict[str, float] = {}

    def sample_premium(self, symbol: str, oracle_price: float,
                       impact_bid_px: float, impact_ask_px: float):
        """Record one premium sample (called every 5 simulated seconds)."""
        if oracle_price <= 0:
            return

        impact_diff = max(impact_bid_px - oracle_price, 0) - max(oracle_price - impact_ask_px, 0) \
            if impact_bid_px is not None and impact_ask_px is not None else 0.0

        premium = impact_diff / oracle_price
        self.premium_samples[symbol].append(premium)

    def compute_hourly_rate(self, symbol: str, funding_multiplier: float = 1.0) -> float:
        """Compute the hourly funding rate from accumulated premium samples.
        
        Returns the hourly rate (1/8 of the 8-hour rate).
        """
        samples = self.premium_samples.get(symbol, [])
        if not samples:
            avg_premium = 0.0
        else:
            avg_premium = sum(samples) / len(samples)

        # Clear samples for next hour
        self.premium_samples[symbol] = []

        # 8-hour funding rate formula
        clamped = max(-0.0005, min(0.0005, INTEREST_RATE - avg_premium))
        eight_hour_rate = (avg_premium + clamped) * funding_multiplier

        # Hourly rate is 1/8 of the 8-hour rate
        hourly_rate = eight_hour_rate / 8.0

        # Apply cap
        hourly_rate = max(-MAX_FUNDING_PER_HOUR, min(MAX_FUNDING_PER_HOUR, hourly_rate))

        self.last_funding_rates[symbol] = hourly_rate
        return hourly_rate

    def compute_funding_payments(self, symbol: str, hourly_rate: float,
                                  oracle_price: float,
                                  positions: Dict[int, float]) -> Dict[int, float]:
        """Compute funding payments for all agents with open positions.
        
        Args:
            symbol: The trading symbol.
            hourly_rate: The computed hourly funding rate.
            oracle_price: Current oracle price (used for notional calculation).
            positions: Dict of agent_id -> position_size (signed).
            
        Returns:
            Dict of agent_id -> payment amount (positive = agent pays, negative = agent receives).
        """
        payments = {}
        for agent_id, size in positions.items():
            if size == 0:
                continue
            # payment = position_size * oracle_price * funding_rate
            # Positive rate + long position => long pays (positive payment)
            # Positive rate + short position => short receives (negative payment)
            payment = size * oracle_price * hourly_rate
            payments[agent_id] = payment

        return payments

    def get_last_rate(self, symbol: str) -> float:
        return self.last_funding_rates.get(symbol, 0.0)

"""Hyperliquid fee model utilities — doc-faithful implementation.

Implements the full Hyperliquid perpetual futures fee schedule:
- 14-day rolling volume with daily UTC tier recomputation
- Staking discounts (6 tiers, up to 40%)
- Maker rebate tiers (based on maker share of volume)
- Referral discounts with 25M volume cap
- Aligned quote adjustment (maker 1.5x rebate, taker 0.8x)
- Growth mode (10% of fees)
- HIP-3 deployer/protocol fee split with fee_scale interaction

Fee scale semantics (HIP-3):
    fee_scale < 1.0:
        total_fee_multiplier = (fee_scale + 1.0)
        deployer_share = fee_scale / (1.0 + fee_scale)
    fee_scale == 1.0:
        total_fee_multiplier = 2.0 (standard 50/50)
        deployer_share = 0.5
    fee_scale > 1.0:
        total_fee_multiplier = fee_scale * 2.0
        deployer_share = 0.5
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Deque, Dict, Optional, Tuple


PERP_FEE_TIERS = [
    {"min_volume": 0.0, "taker": 0.00045, "maker": 0.00015},
    {"min_volume": 5_000_000.0, "taker": 0.00040, "maker": 0.00012},
    {"min_volume": 25_000_000.0, "taker": 0.00035, "maker": 0.00008},
    {"min_volume": 100_000_000.0, "taker": 0.00030, "maker": 0.00004},
    {"min_volume": 500_000_000.0, "taker": 0.00028, "maker": 0.0},
    {"min_volume": 2_000_000_000.0, "taker": 0.00026, "maker": 0.0},
    {"min_volume": 7_000_000_000.0, "taker": 0.00024, "maker": 0.0},
]

STAKING_DISCOUNTS = [
    (500_000.0, 0.40),
    (100_000.0, 0.30),
    (10_000.0, 0.20),
    (1_000.0, 0.15),
    (100.0, 0.10),
    (10.0, 0.05),
]

MAKER_REBATE_TIERS = [
    (0.03, -0.00003),
    (0.015, -0.00002),
    (0.005, -0.00001),
]


@dataclass
class DailyVolumeBucket:
    day: date
    total_volume: float = 0.0
    maker_volume: float = 0.0


@dataclass
class FeeProfile:
    staked_hype: float = 0.0
    referral_discount: float = 0.0
    referred: bool = False
    aligned_quote: bool = False
    current_day: Optional[date] = None
    current_total_volume: float = 0.0
    current_maker_volume: float = 0.0
    historical: Deque[DailyVolumeBucket] = field(default_factory=lambda: deque(maxlen=14))
    cumulative_referral_volume: float = 0.0

    def roll_to_day(self, day: date) -> None:
        if self.current_day is None:
            self.current_day = day
            return

        while self.current_day < day:
            self.historical.append(
                DailyVolumeBucket(
                    day=self.current_day,
                    total_volume=self.current_total_volume,
                    maker_volume=self.current_maker_volume,
                )
            )
            self.current_day = self.current_day + timedelta(days=1)
            self.current_total_volume = 0.0
            self.current_maker_volume = 0.0

    def record_trade(self, notional: float, is_maker: bool, day: date) -> None:
        self.roll_to_day(day)
        self.current_total_volume += notional
        if is_maker:
            self.current_maker_volume += notional
        if self.referred and self.cumulative_referral_volume < 25_000_000.0:
            self.cumulative_referral_volume += notional
            if self.cumulative_referral_volume >= 25_000_000.0:
                self.referral_discount = 0.0

    def rolling_total_volume(self) -> float:
        return self.current_total_volume + sum(bucket.total_volume for bucket in self.historical)

    def rolling_maker_share(self) -> float:
        total = self.rolling_total_volume()
        if total <= 0:
            return 0.0
        maker_total = self.current_maker_volume + sum(bucket.maker_volume for bucket in self.historical)
        return maker_total / total


class FeeEngine:
    def __init__(self):
        self.user_profiles: Dict[int, FeeProfile] = {}

    def bootstrap_profiles(self, profiles: Dict[int, FeeProfile]) -> None:
        self.user_profiles = dict(profiles)

    def get_or_create_profile(self, agent_id: int) -> FeeProfile:
        if agent_id not in self.user_profiles:
            self.user_profiles[agent_id] = FeeProfile()
        return self.user_profiles[agent_id]

    def roll_profiles_to_day(self, day: date) -> None:
        for profile in self.user_profiles.values():
            profile.roll_to_day(day)

    def record_trade(self, agent_id: int, notional: float, is_maker: bool, day: date, growth_mode: bool = False) -> None:
        volume_contribution = notional * 0.1 if growth_mode else notional
        self.get_or_create_profile(agent_id).record_trade(volume_contribution, is_maker=is_maker, day=day)

    @staticmethod
    def _lookup_tier(rolling_volume: float) -> dict:
        tier = PERP_FEE_TIERS[0]
        for candidate in PERP_FEE_TIERS:
            if rolling_volume >= candidate["min_volume"]:
                tier = candidate
        return tier

    @staticmethod
    def _staking_discount(staked_hype: float) -> float:
        for threshold, discount in STAKING_DISCOUNTS:
            if staked_hype >= threshold:
                return discount
        return 0.0

    @staticmethod
    def _maker_rebate(maker_share: float) -> float:
        for threshold, rebate in MAKER_REBATE_TIERS:
            if maker_share >= threshold:
                return rebate
        return 0.0

    @staticmethod
    def _fee_scale_params(deployer_fee_scale: float) -> Tuple[float, float]:
        """Return (total_fee_multiplier, deployer_share_fraction).

        HIP-3 fee scale semantics:
            fee_scale < 1: multiplier = (scale + 1), deployer = scale / (1 + scale)
            fee_scale == 1: multiplier = 2.0, deployer = 0.5
            fee_scale > 1: multiplier = scale * 2.0, deployer = 0.5
        """
        if deployer_fee_scale <= 0:
            return 1.0, 0.0
        if deployer_fee_scale < 1.0:
            multiplier = deployer_fee_scale + 1.0
            deployer_frac = deployer_fee_scale / (1.0 + deployer_fee_scale)
        else:
            multiplier = deployer_fee_scale * 2.0
            deployer_frac = 0.5
        return multiplier, deployer_frac

    def get_fee_rate(
        self,
        agent_id: int,
        is_maker: bool,
        deployer_fee_scale: float = 1.0,
        growth_mode: bool = False,
        is_aligned_quote: bool = False,
    ) -> float:
        """Compute the effective fee rate for a trade.

        Steps (matching Hyperliquid docs):
        1. Look up base tier rate from 14-day rolling volume.
        2. Apply staking discount to the base rate.
        3. For makers, add rebate based on maker share of volume.
        4. Apply HIP-3 fee scale multiplier.
        5. Apply referral discount (taker only; maker rebates are not reduced).
        6. Apply growth mode scaling (10% of fees if enabled).
        7. Apply aligned quote adjustment if applicable.
        """
        profile = self.get_or_create_profile(agent_id)

        # Step 1: Tier lookup
        rolling_volume = profile.rolling_total_volume()
        tier = self._lookup_tier(rolling_volume)
        base_rate = tier["maker"] if is_maker else tier["taker"]

        # Step 2: Staking discount
        staking_disc = self._staking_discount(profile.staked_hype)
        base_rate *= (1.0 - staking_disc)

        # Step 3: Maker rebate
        if is_maker:
            maker_share = profile.rolling_maker_share()
            rebate = self._maker_rebate(maker_share)
            base_rate += rebate  # rebate is negative, so this reduces or inverts

        # Step 4: HIP-3 fee scale
        fee_multiplier, deployer_frac = self._fee_scale_params(deployer_fee_scale)

        # Step 5-7: Combine adjustments
        referral_discount = max(0.0, min(1.0, profile.referral_discount))
        growth_scale = 0.1 if growth_mode else 1.0
        uses_aligned = is_aligned_quote or profile.aligned_quote

        if is_maker:
            if base_rate >= 0:
                # Positive maker fee: apply scale, growth, referral
                rate = base_rate * fee_multiplier * growth_scale * (1.0 - referral_discount)
            else:
                # Maker rebate (negative rate): apply aligned quote scaling
                # Rebate is paid from protocol share and optionally enhanced
                if uses_aligned:
                    # Aligned quote: protocol portion gives 1.5x rebate, deployer portion unchanged
                    protocol_frac = 1.0 - deployer_frac
                    aligned_maker_scale = protocol_frac * 1.5 + deployer_frac
                    rate = base_rate * aligned_maker_scale * growth_scale
                else:
                    rate = base_rate * fee_multiplier * growth_scale
        else:
            # Taker fee: always positive
            rate = base_rate * fee_multiplier * growth_scale * (1.0 - referral_discount)
            if uses_aligned:
                # Aligned quote: taker gets 0.8x on the protocol portion
                protocol_frac = 1.0 - deployer_frac
                aligned_taker_scale = protocol_frac * 0.8 + deployer_frac
                rate *= aligned_taker_scale

        return rate

    @staticmethod
    def compute_fee_split(total_fee: float, deployer_fee_scale: float) -> Dict[str, float]:
        """Split a collected fee between deployer and protocol.

        Only splits positive fees (maker rebates are paid from protocol share).
        """
        if total_fee <= 0:
            return {"deployer": 0.0, "protocol": 0.0}

        _, deployer_frac = FeeEngine._fee_scale_params(deployer_fee_scale)
        return {
            "deployer": total_fee * deployer_frac,
            "protocol": total_fee * (1.0 - deployer_frac),
        }

"""Hyperliquid fee model utilities."""

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Deque, Dict


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
    day: object
    total_volume: float = 0.0
    maker_volume: float = 0.0


@dataclass
class FeeProfile:
    staked_hype: float = 0.0
    referral_discount: float = 0.0
    referred: bool = False
    aligned_quote: bool = False
    current_day: object = None
    current_total_volume: float = 0.0
    current_maker_volume: float = 0.0
    historical: Deque[DailyVolumeBucket] = field(default_factory=lambda: deque(maxlen=14))
    cumulative_referral_volume: float = 0.0

    def roll_to_day(self, day) -> None:
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

    def record_trade(self, notional: float, is_maker: bool, day) -> None:
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

    def roll_profiles_to_day(self, day) -> None:
        for profile in self.user_profiles.values():
            profile.roll_to_day(day)

    def record_trade(self, agent_id: int, notional: float, is_maker: bool, day) -> None:
        self.get_or_create_profile(agent_id).record_trade(notional, is_maker=is_maker, day=day)

    def get_fee_rate(
        self,
        agent_id: int,
        is_maker: bool,
        deployer_fee_scale: float = 1.0,
        growth_mode: bool = False,
        is_aligned_quote: bool = False,
    ) -> float:
        profile = self.get_or_create_profile(agent_id)
        rolling_volume = profile.rolling_total_volume()
        tier = PERP_FEE_TIERS[0]
        for candidate in PERP_FEE_TIERS:
            if rolling_volume >= candidate["min_volume"]:
                tier = candidate

        base_rate = tier["maker"] if is_maker else tier["taker"]
        staking_discount = 0.0
        for threshold, discount in STAKING_DISCOUNTS:
            if profile.staked_hype >= threshold:
                staking_discount = discount
                break
        base_rate *= (1.0 - staking_discount)

        maker_share = profile.rolling_maker_share()
        if is_maker:
            for threshold, rebate in MAKER_REBATE_TIERS:
                if maker_share >= threshold:
                    base_rate += rebate
                    break

        referral_discount = max(0.0, min(1.0, profile.referral_discount))
        growth_mode_scale = 0.1 if growth_mode else 1.0

        if deployer_fee_scale < 1.0:
            scale_if_hip3 = deployer_fee_scale + 1.0
            deployer_share = deployer_fee_scale / (1.0 + deployer_fee_scale) if deployer_fee_scale > 0 else 0.0
        else:
            scale_if_hip3 = deployer_fee_scale * 2.0
            deployer_share = 0.5

        if is_maker:
            rate = base_rate * growth_mode_scale
            if rate > 0:
                rate *= scale_if_hip3 * (1.0 - referral_discount)
            elif is_aligned_quote or profile.aligned_quote:
                maker_scale = (1.0 - deployer_share) * 1.5 + deployer_share
                rate *= maker_scale
        else:
            rate = base_rate * scale_if_hip3 * growth_mode_scale * (1.0 - referral_discount)
            if is_aligned_quote or profile.aligned_quote:
                taker_scale = (1.0 - deployer_share) * 0.8 + deployer_share
                rate *= taker_scale

        return rate

    @staticmethod
    def compute_fee_split(total_fee: float, deployer_fee_scale: float) -> Dict[str, float]:
        if total_fee <= 0:
            return {"deployer": 0.0, "protocol": 0.0}

        if deployer_fee_scale < 1.0:
            deployer_share = deployer_fee_scale / (1.0 + deployer_fee_scale) if deployer_fee_scale > 0 else 0.0
            protocol_share = 1.0 - deployer_share
        else:
            deployer_share = 0.5
            protocol_share = 0.5

        return {
            "deployer": total_fee * deployer_share,
            "protocol": total_fee * protocol_share,
        }

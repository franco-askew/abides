"""HIP-3 contract specifications, margin tables, and deployer configuration.

All HIP-3 deployer actions are represented as configuration loaded from a JSON file.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class MarginMode(Enum):
    NORMAL = "normal"           # cross or isolated per agent choice
    NO_CROSS = "noCross"        # isolated only, margin removal allowed
    STRICT_ISOLATED = "strictIsolated"  # isolated only, no manual margin removal


class TimeInForce(Enum):
    GTC = "GTC"   # good til cancel (default)
    IOC = "IOC"   # immediate or cancel
    ALO = "ALO"   # add liquidity only (post-only)


@dataclass
class MarginTier:
    lower_bound_notional: float
    max_leverage: int

    @property
    def initial_margin_rate(self):
        return 1.0 / self.max_leverage

    @property
    def maintenance_margin_rate(self):
        return self.initial_margin_rate / 2.0


@dataclass
class MarginTable:
    tiers: List[MarginTier]

    def __post_init__(self):
        self.tiers = sorted(self.tiers, key=lambda t: t.lower_bound_notional)
        self._compute_deductions()

    def _compute_deductions(self):
        """Compute maintenance deduction for each tier for continuity."""
        self.deductions = [0.0]
        for n in range(1, len(self.tiers)):
            prev_deduction = self.deductions[n - 1]
            boundary = self.tiers[n].lower_bound_notional
            rate_diff = self.tiers[n].maintenance_margin_rate - self.tiers[n - 1].maintenance_margin_rate
            self.deductions.append(prev_deduction + boundary * rate_diff)

    def get_tier_index(self, notional_value: float) -> int:
        idx = 0
        for i, tier in enumerate(self.tiers):
            if notional_value >= tier.lower_bound_notional:
                idx = i
        return idx

    def get_maintenance_margin(self, notional_value: float) -> float:
        idx = self.get_tier_index(notional_value)
        mm_rate = self.tiers[idx].maintenance_margin_rate
        deduction = self.deductions[idx]
        return notional_value * mm_rate - deduction

    def get_max_leverage(self, notional_value: float) -> int:
        idx = self.get_tier_index(notional_value)
        return self.tiers[idx].max_leverage


@dataclass
class ContractSpec:
    coin: str
    sz_decimals: int = 2
    initial_oracle_px: float = 100.0
    margin_mode: MarginMode = MarginMode.NORMAL
    margin_table: MarginTable = None
    funding_impact_notional: float = 6000.0
    funding_multiplier: float = 1.0
    oi_cap_notional: float = 50_000_000.0
    oi_cap_size: float = 1_000_000_000.0
    max_market_order_value: float = 500_000.0

    def __post_init__(self):
        if self.margin_table is None:
            self.margin_table = MarginTable(tiers=[MarginTier(0, 10)])
        self.max_limit_order_value = self.max_market_order_value * 10

    @property
    def tick_size(self):
        return 10 ** (-self.sz_decimals)


@dataclass
class FeeSchedule:
    maker_fee_bps: float = 2.0     # basis points (0.02%)
    taker_fee_bps: float = 7.0     # basis points (0.07%)
    deployer_share: float = 0.5
    protocol_share: float = 0.5

    @property
    def maker_fee_rate(self):
        return self.maker_fee_bps / 10000.0

    @property
    def taker_fee_rate(self):
        return self.taker_fee_bps / 10000.0


@dataclass
class PerpDexConfig:
    """Full HIP-3 DEX configuration loaded from deployer_config.json."""
    dex_name: str = "SIM_DEX"
    collateral_token: str = "USDC"
    fee_schedule: FeeSchedule = None
    assets: Dict[str, ContractSpec] = field(default_factory=dict)
    oracle_update_interval_s: float = 3.0
    deployer_mark_px_mode: str = "none"      # "none", "oracle_based", "custom"
    external_perp_px_mode: str = "ema_of_mark"  # "ema_of_mark", "none"

    def __post_init__(self):
        if self.fee_schedule is None:
            self.fee_schedule = FeeSchedule()


def load_deployer_config(config_path: str) -> PerpDexConfig:
    """Load a HIP-3 deployer configuration from a JSON file."""
    with open(config_path, 'r') as f:
        raw = json.load(f)

    dex_raw = raw.get('dex', {})
    fee_raw = dex_raw.get('fee_schedule', {})
    fee_schedule = FeeSchedule(
        maker_fee_bps=fee_raw.get('maker_fee_bps', 2.0),
        taker_fee_bps=fee_raw.get('taker_fee_bps', 7.0),
        deployer_share=fee_raw.get('deployer_share', 0.5),
        protocol_share=fee_raw.get('protocol_share', 0.5),
    )

    assets = {}
    for asset_raw in raw.get('assets', []):
        tiers = []
        mt_raw = asset_raw.get('margin_table', {})
        for tier_raw in mt_raw.get('tiers', [{'lower_bound_notional': 0, 'max_leverage': 10}]):
            tiers.append(MarginTier(
                lower_bound_notional=tier_raw['lower_bound_notional'],
                max_leverage=tier_raw['max_leverage'],
            ))
        margin_table = MarginTable(tiers=tiers)

        mode_str = asset_raw.get('margin_mode', 'normal')
        margin_mode = MarginMode(mode_str)

        spec = ContractSpec(
            coin=asset_raw['coin'],
            sz_decimals=asset_raw.get('sz_decimals', 2),
            initial_oracle_px=asset_raw.get('initial_oracle_px', 100.0),
            margin_mode=margin_mode,
            margin_table=margin_table,
            funding_impact_notional=asset_raw.get('funding_impact_notional', 6000.0),
            funding_multiplier=asset_raw.get('funding_multiplier', 1.0),
            oi_cap_notional=asset_raw.get('oi_cap_notional', 50_000_000.0),
            oi_cap_size=asset_raw.get('oi_cap_size', 1_000_000_000.0),
            max_market_order_value=asset_raw.get('max_market_order_value', 500_000.0),
        )
        assets[spec.coin] = spec

    config = PerpDexConfig(
        dex_name=dex_raw.get('name', 'SIM_DEX'),
        collateral_token=dex_raw.get('collateral_token', 'USDC'),
        fee_schedule=fee_schedule,
        assets=assets,
        oracle_update_interval_s=raw.get('oracle_update_interval_s', 3.0),
        deployer_mark_px_mode=raw.get('deployer_mark_px_mode', 'none'),
        external_perp_px_mode=raw.get('external_perp_px_mode', 'ema_of_mark'),
    )
    return config

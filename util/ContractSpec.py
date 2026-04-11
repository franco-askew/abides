"""HIP-3 contract specifications and deployer configuration helpers."""

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, List, Optional


MAX_PERP_PRICE_DECIMALS = 6
MAX_PERP_SIGNIFICANT_FIGURES = 5


class MarginMode(Enum):
    NORMAL = "normal"
    NO_CROSS = "noCross"
    STRICT_ISOLATED = "strictIsolated"


class MarginType(Enum):
    CROSS = "cross"
    ISOLATED = "isolated"
    INHERIT = "inherit"


class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"
    ALO = "ALO"


@dataclass
class MarginTier:
    lower_bound_notional: float
    max_leverage: int

    @property
    def initial_margin_rate(self) -> float:
        return 1.0 / self.max_leverage

    @property
    def maintenance_margin_rate(self) -> float:
        return self.initial_margin_rate / 2.0


@dataclass
class MarginTable:
    tiers: List[MarginTier]
    description: str = ""
    table_id: int = 0

    def __post_init__(self) -> None:
        self.tiers = sorted(self.tiers, key=lambda tier: tier.lower_bound_notional)
        self._compute_deductions()

    def _compute_deductions(self) -> None:
        self.deductions = [0.0]
        for idx in range(1, len(self.tiers)):
            prev = self.tiers[idx - 1]
            cur = self.tiers[idx]
            prev_deduction = self.deductions[idx - 1]
            boundary = cur.lower_bound_notional
            rate_diff = cur.maintenance_margin_rate - prev.maintenance_margin_rate
            self.deductions.append(prev_deduction + boundary * rate_diff)

    def get_tier_index(self, notional_value: float) -> int:
        tier_idx = 0
        for idx, tier in enumerate(self.tiers):
            if notional_value >= tier.lower_bound_notional:
                tier_idx = idx
        return tier_idx

    def get_maintenance_margin(self, notional_value: float) -> float:
        idx = self.get_tier_index(notional_value)
        return notional_value * self.tiers[idx].maintenance_margin_rate - self.deductions[idx]

    def get_initial_margin(self, notional_value: float) -> float:
        idx = self.get_tier_index(notional_value)
        return notional_value * self.tiers[idx].initial_margin_rate

    def get_max_leverage(self, notional_value: float) -> int:
        idx = self.get_tier_index(notional_value)
        return self.tiers[idx].max_leverage


@dataclass
class ContractSpec:
    coin: str
    sz_decimals: int = 2
    initial_oracle_px: float = 100.0
    oracle_csv: Optional[str] = None
    margin_mode: MarginMode = MarginMode.NORMAL
    margin_table: MarginTable = None
    margin_table_id: int = 0
    funding_impact_notional: float = 6000.0
    funding_multiplier: float = 1.0
    funding_interest_rate_8h: float = 0.0001
    oi_cap_notional: float = 50_000_000.0
    oi_cap_size: float = 1_000_000_000.0
    growth_mode: bool = False
    aligned_quote: bool = False
    max_market_order_value: Optional[float] = None
    max_limit_order_value: Optional[float] = None
    min_order_value: float = 11.0
    default_leverage: int = 10

    def __post_init__(self) -> None:
        if self.margin_table is None:
            self.margin_table = MarginTable(tiers=[MarginTier(0, max(1, self.default_leverage))])
        if self.max_market_order_value is None:
            self.max_market_order_value = self._default_max_market_value()
        if self.max_limit_order_value is None:
            self.max_limit_order_value = 10 * self.max_market_order_value

    def _default_max_market_value(self) -> float:
        max_lev = self.margin_table.get_max_leverage(0.0)
        if max_lev >= 25:
            return 30_000_000.0
        if max_lev >= 20:
            return 5_000_000.0
        if max_lev >= 10:
            return 2_000_000.0
        return 500_000.0

    @property
    def tick_size(self) -> float:
        return 10 ** (-self.sz_decimals)

    @property
    def max_price_decimals(self) -> int:
        return max(0, MAX_PERP_PRICE_DECIMALS - self.sz_decimals)

    def round_size(self, quantity: float) -> float:
        return round(float(quantity), self.sz_decimals)

    def is_valid_price(self, price: float) -> bool:
        try:
            decimal_price = Decimal(str(price))
        except (InvalidOperation, ValueError):
            return False

        if decimal_price <= 0:
            return False

        normalized = decimal_price.normalize()
        digits = normalized.as_tuple().digits
        exponent = normalized.as_tuple().exponent
        decimals = max(0, -exponent)
        significant_figures = len(digits)

        if decimal_price == decimal_price.to_integral():
            return True

        return (
            significant_figures <= MAX_PERP_SIGNIFICANT_FIGURES
            and decimals <= self.max_price_decimals
        )

    def to_trading_rules(self) -> Dict[str, Optional[float]]:
        return {
            "sz_decimals": int(self.sz_decimals),
            "max_price_decimals": int(self.max_price_decimals),
            "max_significant_figures": int(MAX_PERP_SIGNIFICANT_FIGURES),
            "min_order_value": float(self.min_order_value),
            "max_limit_order_value": (
                float(self.max_limit_order_value)
                if self.max_limit_order_value is not None
                else None
            ),
            "max_market_order_value": (
                float(self.max_market_order_value)
                if self.max_market_order_value is not None
                else None
            ),
        }


@dataclass
class DeployerPermission:
    variants: List[str] = field(default_factory=list)

    def allows(self, action_name: str) -> bool:
        return "*" in self.variants or action_name in self.variants


@dataclass
class FeeSchedule:
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 7.0
    deployer_share: float = 0.5
    protocol_share: float = 0.5

    @property
    def maker_fee_rate(self) -> float:
        return self.maker_fee_bps / 10000.0

    @property
    def taker_fee_rate(self) -> float:
        return self.taker_fee_bps / 10000.0


@dataclass
class PerpDexConfig:
    dex_name: str = "SIM_DEX"
    collateral_token: str = "USDC"
    fee_schedule: FeeSchedule = None
    fee_scale: float = 1.0
    fee_model: str = "hyperliquid"
    execution_mode: str = "hypercore_blocked"
    block_interval_ms: int = 100
    dex_open_interest_cap_notional: float = float("inf")
    assets: Dict[str, ContractSpec] = field(default_factory=dict)
    margin_tables: Dict[int, MarginTable] = field(default_factory=dict)
    oracle_update_interval_s: float = 3.0
    deployer_mark_px_mode: str = "none"
    external_perp_px_mode: str = "ema_of_mark"
    deployer_permissions: Dict[int, DeployerPermission] = field(default_factory=dict)
    default_sub_deployer_permissions: Dict[int, DeployerPermission] = field(default_factory=dict)
    perp_annotations: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fee_schedule is None:
            self.fee_schedule = FeeSchedule()
        self.fee_scale = max(0.0, min(3.0, self.fee_scale))


def _build_margin_table(raw: dict, default_id: int = 0) -> MarginTable:
    tiers = []
    for tier_raw in raw.get("tiers", raw.get("marginTiers", [{"lower_bound_notional": 0, "max_leverage": 10}])):
        lower = tier_raw.get("lower_bound_notional", tier_raw.get("lowerBound", 0))
        max_leverage = tier_raw.get("max_leverage", tier_raw.get("maxLeverage", 10))
        tiers.append(MarginTier(lower_bound_notional=float(lower), max_leverage=int(max_leverage)))
    return MarginTable(
        tiers=tiers,
        description=raw.get("description", ""),
        table_id=int(raw.get("id", default_id)),
    )


def _parse_permissions(raw: dict) -> Dict[int, DeployerPermission]:
    permissions = {}
    for agent_id_str, variants in raw.items():
        if isinstance(variants, dict):
            allowed = [name for name, flag in variants.items() if flag]
        else:
            allowed = list(variants)
        permissions[int(agent_id_str)] = DeployerPermission(variants=allowed)
    return permissions


def load_deployer_config(config_path: str) -> PerpDexConfig:
    with open(config_path, "r") as handle:
        raw = json.load(handle)

    dex_raw = raw.get("dex", {})
    fee_raw = dex_raw.get("fee_schedule", {})
    fee_schedule = FeeSchedule(
        maker_fee_bps=float(fee_raw.get("maker_fee_bps", 2.0)),
        taker_fee_bps=float(fee_raw.get("taker_fee_bps", 7.0)),
        deployer_share=float(fee_raw.get("deployer_share", 0.5)),
        protocol_share=float(fee_raw.get("protocol_share", 0.5)),
    )

    margin_tables: Dict[int, MarginTable] = {}
    for table_raw in raw.get("margin_tables", []):
        table = _build_margin_table(table_raw, default_id=int(table_raw.get("id", 0)))
        margin_tables[table.table_id] = table

    assets = {}
    for idx, asset_raw in enumerate(raw.get("assets", []), start=1):
        margin_table_id = int(asset_raw.get("margin_table_id", asset_raw.get("marginTableId", 0)))
        if "margin_table" in asset_raw:
            margin_table = _build_margin_table(asset_raw["margin_table"], default_id=margin_table_id or idx)
            margin_table_id = margin_table.table_id or margin_table_id or idx
            margin_table.table_id = margin_table_id
            margin_tables[margin_table_id] = margin_table
        else:
            margin_table = margin_tables.get(margin_table_id)

        if margin_table is None:
            margin_table = MarginTable(tiers=[MarginTier(0, int(asset_raw.get("default_leverage", 10)))], table_id=margin_table_id)
            margin_tables[margin_table.table_id] = margin_table

        margin_mode = MarginMode(asset_raw.get("margin_mode", asset_raw.get("marginMode", "normal")))

        spec = ContractSpec(
            coin=asset_raw["coin"],
            sz_decimals=int(asset_raw.get("sz_decimals", asset_raw.get("szDecimals", 2))),
            initial_oracle_px=float(asset_raw.get("initial_oracle_px", asset_raw.get("oraclePx", 100.0))),
            oracle_csv=asset_raw.get("oracle_csv"),
            margin_mode=margin_mode,
            margin_table=margin_table,
            margin_table_id=margin_table_id,
            funding_impact_notional=float(asset_raw.get("funding_impact_notional", 6000.0)),
            funding_multiplier=float(asset_raw.get("funding_multiplier", 1.0)),
            funding_interest_rate_8h=float(asset_raw.get("funding_interest_rate_8h", 0.0001)),
            oi_cap_notional=float(asset_raw.get("oi_cap_notional", 50_000_000.0)),
            oi_cap_size=float(asset_raw.get("oi_cap_size", 1_000_000_000.0)),
            growth_mode=bool(asset_raw.get("growth_mode", False)),
            aligned_quote=bool(asset_raw.get("aligned_quote", False)),
            max_market_order_value=(
                float(asset_raw["max_market_order_value"])
                if asset_raw.get("max_market_order_value") is not None
                else None
            ),
            max_limit_order_value=(
                float(asset_raw["max_limit_order_value"])
                if asset_raw.get("max_limit_order_value") is not None
                else None
            ),
            min_order_value=float(asset_raw.get("min_order_value", 11.0)),
            default_leverage=int(asset_raw.get("default_leverage", 10)),
        )
        assets[spec.coin] = spec

    config = PerpDexConfig(
        dex_name=dex_raw.get("name", "SIM_DEX"),
        collateral_token=dex_raw.get("collateral_token", "USDC"),
        fee_schedule=fee_schedule,
        fee_scale=float(raw.get("fee_scale", dex_raw.get("fee_scale", 1.0))),
        fee_model=str(raw.get("fee_model", dex_raw.get("fee_model", "hyperliquid"))),
        execution_mode=str(raw.get("execution_mode", raw.get("executionMode", "hypercore_blocked"))),
        block_interval_ms=int(raw.get("block_interval_ms", raw.get("blockIntervalMs", 100))),
        dex_open_interest_cap_notional=float(raw.get("dex_open_interest_cap_notional", float("inf"))),
        assets=assets,
        margin_tables=margin_tables,
        oracle_update_interval_s=float(raw.get("oracle_update_interval_s", 3.0)),
        deployer_mark_px_mode=raw.get("deployer_mark_px_mode", "none"),
        external_perp_px_mode=raw.get("external_perp_px_mode", "ema_of_mark"),
        deployer_permissions=_parse_permissions(raw.get("deployer_permissions", {})),
        default_sub_deployer_permissions=_parse_permissions(raw.get("sub_deployer_permissions", {})),
        perp_annotations=raw.get("perp_annotations", {}),
    )
    return config

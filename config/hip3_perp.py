"""HIP-3 perpetual futures simulation configuration."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from Kernel import Kernel
from agent.ChiarellaAgent import ChiarellaAgent
from agent.OracleDeployerAgent import OracleDeployerAgent
from agent.PerpExchangeAgent import PerpExchangeAgent
from agent.PerpMomentumAgent import PerpMomentumAgent
from agent.PerpNoiseAgent import PerpNoiseAgent
from agent.PerpValueAgent import PerpValueAgent
from util.ContractSpec import DeployerPermission, load_deployer_config
from util.oracle.MultiCsvOracle import MultiCsvOracle
import util.util as util


parser = argparse.ArgumentParser(description="HIP-3 Perp Simulation")
parser.add_argument("-c", "--config", required=False, default="hip3_perp")
parser.add_argument(
    "--deployer-config",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "deployer_config.json"),
    help="Path to deployer_config.json",
)
parser.add_argument("--oracle-csv", type=str, default=None, help="Path to oracle CSV override for the selected symbol")
parser.add_argument("--symbol", type=str, default=None, help="Primary symbol for strategy agents")
parser.add_argument("--start-time", type=str, default="2025-01-01 00:00:00")
parser.add_argument("--end-time", type=str, default="2025-01-01 01:00:00")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-agents", type=int, default=0, help="Shortcut for --num-noise")
parser.add_argument("--num-noise", type=int, default=None)
parser.add_argument("--num-momentum", type=int, default=0)
parser.add_argument("--num-value", type=int, default=0)
parser.add_argument("--num-fundamentalists", type=int, default=0)
parser.add_argument("--num-chartists", type=int, default=0)
parser.add_argument("--num-chiarella-noise", type=int, default=0)
parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
parser.add_argument("--order-size", type=float, default=1.0)
parser.add_argument("--sigma-e", type=float, default=0.05)
parser.add_argument("--k-max", type=float, default=None)
parser.add_argument("--l-min", type=int, default=None)
parser.add_argument("--l-max", type=int, default=None)
parser.add_argument("--bias", type=float, default=0.5)
parser.add_argument("--exit-prob", type=float, default=None)
parser.add_argument("--wake-interval", type=float, default=60.0)
parser.add_argument("--noise-wake-freq", type=str, default="30s")
parser.add_argument("--noise-trade-prob", type=float, default=None)
parser.add_argument("--noise-max-position-size", type=float, default=None)
parser.add_argument("--noise-max-live-orders", type=int, default=None)
parser.add_argument("--noise-opening-order-cooldown-s", type=float, default=600.0)
parser.add_argument("--noise-max-take-distance-bps", type=float, default=50.0)
parser.add_argument("--noise-max-passive-distance-bps", type=float, default=None)
parser.add_argument("--momentum-wake-freq", type=str, default="300s")
parser.add_argument("--momentum-crossover-deadband-bps", type=float, default=None)
parser.add_argument("--momentum-max-position-size", type=float, default=None)
parser.add_argument("--momentum-max-live-orders", type=int, default=None)
parser.add_argument("--momentum-opening-order-cooldown-s", type=float, default=600.0)
parser.add_argument("--momentum-max-take-distance-bps", type=float, default=50.0)
parser.add_argument("--momentum-max-passive-distance-bps", type=float, default=None)
parser.add_argument("--value-mean-wake-interval-s", type=float, default=300.0)
parser.add_argument("--value-mispricing-deadband-bps", type=float, default=None)
parser.add_argument("--value-aggressive-cross-prob", type=float, default=None)
parser.add_argument("--value-max-position-size", type=float, default=None)
parser.add_argument("--value-max-live-orders", type=int, default=None)
parser.add_argument("--value-opening-order-cooldown-s", type=float, default=600.0)
parser.add_argument("--value-max-take-distance-bps", type=float, default=50.0)
parser.add_argument("--value-max-passive-distance-bps", type=float, default=None)
parser.add_argument("--forecast-deadband-bps", type=float, default=None)
parser.add_argument("--chiarella-max-position-size", type=float, default=None)
parser.add_argument("--chiarella-max-live-orders", type=int, default=None)
parser.add_argument("--chiarella-opening-order-cooldown-s", type=float, default=600.0)
parser.add_argument("--chiarella-max-take-distance-bps", type=float, default=50.0)
parser.add_argument("--chiarella-max-passive-distance-bps", type=float, default=None)
parser.set_defaults(momentum_trade_on_signal_change_only=None)
parser.add_argument(
    "--momentum-trade-on-signal-change-only",
    dest="momentum_trade_on_signal_change_only",
    action="store_true",
)
parser.add_argument(
    "--momentum-trade-every-wake",
    dest="momentum_trade_on_signal_change_only",
    action="store_false",
)
parser.add_argument("--log-orders", action="store_true", default=False)
parser.add_argument("--log-l1", action="store_true", default=False, help="Record L1 (top-of-book) snapshots to L1.csv")
parser.add_argument("--log-l2", action="store_true", default=False, help="Record L2 (full depth) snapshots to L2.csv")
parser.add_argument("--log-dir", type=str, default=None, help="Run artifact directory under ./log")
parser.add_argument("--block-interval-ms", type=int, default=None)
parser.add_argument("--execution-mode", type=str, default=None)
parser.add_argument("--fee-model", type=str, default=None)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-o", "--override", action="append", default=[], help="Key=value overrides")

args, remaining = parser.parse_known_args()
util.silent_mode = not args.verbose


def _load_raw_config(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _parse_overrides(raw_overrides):
    parsed = {}
    for override in raw_overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            parsed[key] = value
    return parsed


def _non_null_kwargs(**kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


overrides = _parse_overrides(args.override)
dex_config = load_deployer_config(args.deployer_config)
raw_config = _load_raw_config(args.deployer_config)

if args.execution_mode:
    dex_config.execution_mode = args.execution_mode
if args.fee_model:
    dex_config.fee_model = args.fee_model
if args.block_interval_ms is not None:
    dex_config.block_interval_ms = args.block_interval_ms

symbols = list(dex_config.assets.keys())
primary_symbol = overrides.get("symbol", args.symbol) or symbols[0]
if primary_symbol not in dex_config.assets:
    print(f"ERROR: symbol {primary_symbol} not found in deployer config")
    sys.exit(1)
primary_spec = dex_config.assets[primary_symbol]
trading_rules_by_symbol = {
    symbol: spec.to_trading_rules()
    for symbol, spec in dex_config.assets.items()
}

oracle_map = {}
for asset_raw in raw_config.get("assets", []):
    symbol = asset_raw["coin"]
    csv_path = asset_raw.get("oracle_csv")
    if symbol == primary_symbol and (overrides.get("oracle_csv") or args.oracle_csv):
        csv_path = overrides.get("oracle_csv", args.oracle_csv)
    if csv_path:
        oracle_map[symbol] = csv_path

if primary_symbol not in oracle_map:
    print("ERROR: No oracle CSV specified for the primary symbol. Use --oracle-csv or set oracle_csv in deployer_config.json")
    sys.exit(1)

oracle = MultiCsvOracle(oracle_map)

start_time = pd.Timestamp(args.start_time)
end_time = pd.Timestamp(args.end_time)
kernel_start = start_time - pd.Timedelta("1min")
kernel_stop = end_time + pd.Timedelta("1min")

seed = args.seed
np.random.seed(seed)
run_log_dir = args.log_dir or f"hip3_perp_{seed}"

print(f"Run log directory: log/{run_log_dir}")

agent_count = 0
agents = []

exchange = PerpExchangeAgent(
    id=agent_count,
    name="PERP_EXCHANGE",
    type="PerpExchangeAgent",
    dex_config=dex_config,
    pipeline_delay=40000,
    computation_delay=1,
    stream_history=10,
    log_orders=args.log_orders,
    log_l1=args.log_l1,
    log_l2=args.log_l2,
    starting_balances={},
    block_interval_ms=dex_config.block_interval_ms,
    execution_mode=dex_config.execution_mode,
    fee_model=dex_config.fee_model,
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
)
agents.append(exchange)
exchange_id = agent_count
agent_count += 1

deployer = OracleDeployerAgent(
    id=agent_count,
    name="ORACLE_DEPLOYER",
    type="OracleDeployerAgent",
    symbols=symbols,
    oracle_update_interval_s=dex_config.oracle_update_interval_s,
    deployer_mark_px_mode=dex_config.deployer_mark_px_mode,
    external_perp_px_mode=dex_config.external_perp_px_mode,
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
)
agents.append(deployer)
deployer_id = agent_count
agent_count += 1

num_noise = args.num_noise if args.num_noise is not None else args.num_agents

for i in range(num_noise):
    agents.append(
        PerpNoiseAgent(
            id=agent_count,
            name=f"Noise_{i}",
            type="PerpNoiseAgent",
            symbol=primary_symbol,
            starting_cash=args.starting_cash,
            wake_up_freq=args.noise_wake_freq,
            log_orders=args.log_orders,
            trading_rules_by_symbol=trading_rules_by_symbol,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
            **_non_null_kwargs(
                trade_probability=args.noise_trade_prob,
                max_position_size=args.noise_max_position_size,
                max_live_orders_per_symbol=args.noise_max_live_orders,
                opening_order_cooldown_after_unfunded_s=args.noise_opening_order_cooldown_s,
                max_take_distance_bps_from_mark=args.noise_max_take_distance_bps,
                max_passive_distance_bps_from_mark=args.noise_max_passive_distance_bps,
            ),
        )
    )
    agent_count += 1

for i in range(args.num_momentum):
    agents.append(
        PerpMomentumAgent(
            id=agent_count,
            name=f"Momentum_{i}",
            type="PerpMomentumAgent",
            symbol=primary_symbol,
            starting_cash=args.starting_cash,
            wake_up_freq=args.momentum_wake_freq,
            log_orders=args.log_orders,
            trading_rules_by_symbol=trading_rules_by_symbol,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
            **_non_null_kwargs(
                trade_on_signal_change_only=args.momentum_trade_on_signal_change_only,
                crossover_deadband_bps=args.momentum_crossover_deadband_bps,
                max_position_size=args.momentum_max_position_size,
                max_live_orders_per_symbol=args.momentum_max_live_orders,
                opening_order_cooldown_after_unfunded_s=args.momentum_opening_order_cooldown_s,
                max_take_distance_bps_from_mark=args.momentum_max_take_distance_bps,
                max_passive_distance_bps_from_mark=args.momentum_max_passive_distance_bps,
            ),
        )
    )
    agent_count += 1

for i in range(args.num_value):
    agents.append(
        PerpValueAgent(
            id=agent_count,
            name=f"Value_{i}",
            type="PerpValueAgent",
            symbol=primary_symbol,
            starting_cash=args.starting_cash,
            r_bar=dex_config.assets[primary_symbol].initial_oracle_px,
            mean_wake_interval_s=args.value_mean_wake_interval_s,
            log_orders=args.log_orders,
            trading_rules_by_symbol=trading_rules_by_symbol,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
            **_non_null_kwargs(
                mispricing_deadband_bps=args.value_mispricing_deadband_bps,
                aggressive_cross_prob=args.value_aggressive_cross_prob,
                max_position_size=args.value_max_position_size,
                max_live_orders_per_symbol=args.value_max_live_orders,
                opening_order_cooldown_after_unfunded_s=args.value_opening_order_cooldown_s,
                max_take_distance_bps_from_mark=args.value_max_take_distance_bps,
                max_passive_distance_bps_from_mark=args.value_max_passive_distance_bps,
            ),
        )
    )
    agent_count += 1

chiarella_configs = [
    ("Fundamentalist", args.num_fundamentalists, 10.0, 0.0, 0.0),
    ("Chartist", args.num_chartists, 0.0, 10.0, 0.0),
    ("Noise", args.num_chiarella_noise, 0.0, 0.0, 10.0),
]

for label, count, sigma_f, sigma_c, sigma_n in chiarella_configs:
    for i in range(count):
        agents.append(
            ChiarellaAgent(
                id=agent_count,
                name=f"{label}_{i}",
                type="ChiarellaAgent",
                symbol=primary_symbol,
                sigma_f=sigma_f,
                sigma_c=sigma_c,
                sigma_n=sigma_n,
                sigma_e=args.sigma_e,
                bias=args.bias,
                order_size=args.order_size,
                wake_interval_s=args.wake_interval,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
                starting_cash=args.starting_cash,
                log_orders=args.log_orders,
                trading_rules_by_symbol=trading_rules_by_symbol,
                **_non_null_kwargs(
                    k_max=args.k_max,
                    l_min=args.l_min,
                    l_max=args.l_max,
                    exit_prob=args.exit_prob,
                    forecast_deadband_bps=args.forecast_deadband_bps,
                    max_position_size=args.chiarella_max_position_size,
                    max_live_orders_per_symbol=args.chiarella_max_live_orders,
                    opening_order_cooldown_after_unfunded_s=args.chiarella_opening_order_cooldown_s,
                    max_take_distance_bps_from_mark=args.chiarella_max_take_distance_bps,
                    max_passive_distance_bps_from_mark=args.chiarella_max_passive_distance_bps,
                ),
            )
        )
        agent_count += 1

starting_balances = {}
for agent in agents:
    if hasattr(agent, "starting_cash"):
        starting_balances[agent.id] = float(agent.starting_cash)
exchange.starting_balances = starting_balances
exchange.clearinghouse.bootstrap_accounts(starting_balances)
exchange.deployer_permissions[deployer_id] = DeployerPermission(variants=["*"])

kernel = Kernel(
    "HIP3 Perp Simulation",
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
)

latency = [[1_000_000] * len(agents) for _ in range(len(agents))]

# Oracle access modes: deployer and exchange get unrestricted, all others get strict
oracle_access_modes = {
    exchange_id: {"mode": "unrestricted"},
    deployer_id: {"mode": "unrestricted"},
}
# All other agents default to "strict" (enforced by Kernel)

kernel.runner(
    agents=agents,
    startTime=kernel_start,
    stopTime=kernel_stop,
    agentLatency=latency,
    defaultComputationDelay=1,
    seed=seed,
    oracle=oracle,
    log_dir=run_log_dir,
    oracle_access_modes=oracle_access_modes,
)

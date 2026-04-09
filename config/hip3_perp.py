"""HIP-3 Perpetual Futures Simulation Configuration.

Usage:
    python -u abides.py -c hip3_perp -o oracle_csv=data/oracle.csv -o symbol=ASSET-USD

    # Run with Chiarella agents (Rao 2025):
    python abides.py -c hip3_perp --oracle-csv data/oracles/trump.csv \\
        --num-fundamentalists 50 --num-chartists 100 --num-noise 50 \\
        --start-time "2025-12-10 00:00:00" --end-time "2025-12-11 00:00:00"

This config:
  1. Loads deployer_config.json for all HIP-3 parameters.
  2. Builds a CsvOracle from the specified CSV file.
  3. Instantiates the PerpExchangeAgent, OracleDeployerAgent.
  4. Optionally adds Chiarella et al. (2002) agents: Fundamentalist, Chartist, Noise.
  5. Supports user-defined agent injection via the `agents` list.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from Kernel import Kernel
from agent.PerpExchangeAgent import PerpExchangeAgent
from agent.PerpTradingAgent import PerpTradingAgent
from agent.PerpNoiseAgent import PerpNoiseAgent
from agent.PerpMomentumAgent import PerpMomentumAgent
from agent.PerpValueAgent import PerpValueAgent
from agent.OracleDeployerAgent import OracleDeployerAgent
from agent.ChiarellaAgent import ChiarellaAgent
from util.oracle.CsvOracle import CsvOracle
from util.ContractSpec import load_deployer_config
import util.util as util

# ── Argument parsing ────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='HIP-3 Perp Simulation')
parser.add_argument('-c', '--config', required=False, default='hip3_perp')
parser.add_argument('--deployer-config', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'deployer_config.json'),
                    help='Path to deployer_config.json')
parser.add_argument('--oracle-csv', type=str, default=None,
                    help='Path to oracle CSV (timestamp, price)')
parser.add_argument('--symbol', type=str, default=None,
                    help='Symbol to trade (must match deployer config)')
parser.add_argument('--start-time', type=str, default='2025-01-01 00:00:00',
                    help='Simulation start time (ISO format)')
parser.add_argument('--end-time', type=str, default='2025-01-01 01:00:00',
                    help='Simulation end time (ISO format)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num-agents', type=int, default=0,
                    help='Number of PerpNoiseAgents (shortcut for --num-noise)')
parser.add_argument('--num-noise', type=int, default=None,
                    help='Number of PerpNoiseAgents')
parser.add_argument('--num-momentum', type=int, default=0,
                    help='Number of PerpMomentumAgents')
parser.add_argument('--num-value', type=int, default=0,
                    help='Number of PerpValueAgents')
parser.add_argument('--num-fundamentalists', type=int, default=0,
                    help='Number of Fundamentalist agents (Chiarella, mean-reversion to oracle)')
parser.add_argument('--num-chartists', type=int, default=0,
                    help='Number of Chartist agents (Chiarella, momentum/moving-average)')
parser.add_argument('--num-chiarella-noise', type=int, default=0,
                    help='Number of Chiarella Noise agents (random forecast)')
parser.add_argument('--starting-cash', type=float, default=1_000_000.0,
                    help='Starting cash for each agent')
parser.add_argument('--order-size', type=float, default=1.0,
                    help='Order size per agent per wakeup')
parser.add_argument('--sigma-e', type=float, default=0.05,
                    help='Noise volatility parameter (sigma_epsilon)')
parser.add_argument('--k-max', type=float, default=0.05,
                    help='Max bid/ask spread factor (k_max)')
parser.add_argument('--l-min', type=int, default=1,
                    help='Chartist min lookback horizon')
parser.add_argument('--l-max', type=int, default=5,
                    help='Chartist max lookback horizon')
parser.add_argument('--bias', type=float, default=0.5,
                    help='Positional vs basis trading bias (0=longs positional, 0.5=equal)')
parser.add_argument('--exit-prob', type=float, default=0.05,
                    help='Probability of closing position each wakeup')
parser.add_argument('--wake-interval', type=float, default=60.0,
                    help='Seconds between agent wakeups')
parser.add_argument('--log-orders', action='store_true', default=False)
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Print kernel/agent debug messages (very slow with many agents)')
parser.add_argument('-o', '--override', action='append', default=[],
                    help='Key=value overrides (e.g. -o oracle_csv=data/oracle.csv)')

args, remaining = parser.parse_known_args()

# Silent mode (default) suppresses all log_print output for speed
util.silent_mode = not args.verbose

# Process -o overrides
overrides = {}
for ov in args.override:
    if '=' in ov:
        k, v = ov.split('=', 1)
        overrides[k] = v

oracle_csv = overrides.get('oracle_csv', args.oracle_csv)
symbol_override = overrides.get('symbol', args.symbol)

# ── Load deployer configuration ─────────────────────────────────────────

dex_config = load_deployer_config(args.deployer_config)

# Override symbol if specified
symbols = list(dex_config.assets.keys())
if symbol_override and symbol_override in dex_config.assets:
    symbols = [symbol_override]

# If oracle_csv is provided, update the config
# The first asset's oracle_csv is used as the default
if oracle_csv is None:
    first_asset_raw = None
    with open(args.deployer_config, 'r') as f:
        raw = json.load(f)
    for asset_raw in raw.get('assets', []):
        if asset_raw.get('coin') in symbols:
            oracle_csv = asset_raw.get('oracle_csv')
            break

if oracle_csv is None:
    print("ERROR: No oracle CSV specified. Use --oracle-csv or set oracle_csv in deployer_config.json")
    sys.exit(1)

# ── Timing ──────────────────────────────────────────────────────────────

start_time = pd.Timestamp(args.start_time)
end_time = pd.Timestamp(args.end_time)
kernel_start = start_time - pd.Timedelta('1min')
kernel_stop = end_time + pd.Timedelta('1min')

# ── Random state ────────────────────────────────────────────────────────

seed = args.seed
np.random.seed(seed)

# ── Oracle ──────────────────────────────────────────────────────────────

primary_symbol = symbols[0]
oracle = CsvOracle(primary_symbol, oracle_csv)

# ── Build agents ────────────────────────────────────────────────────────

agent_count = 0
agents = []

# 0: PerpExchangeAgent
exchange = PerpExchangeAgent(
    id=agent_count, name="PERP_EXCHANGE", type="PerpExchangeAgent",
    dex_config=dex_config,
    pipeline_delay=40000, computation_delay=1,
    stream_history=10, log_orders=args.log_orders,
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
)
agents.append(exchange)
agent_count += 1

# 1: OracleDeployerAgent
deployer = OracleDeployerAgent(
    id=agent_count, name="ORACLE_DEPLOYER", type="OracleDeployerAgent",
    symbols=symbols,
    oracle_update_interval_s=dex_config.oracle_update_interval_s,
    deployer_mark_px_mode=dex_config.deployer_mark_px_mode,
    external_perp_px_mode=dex_config.external_perp_px_mode,
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
)
agents.append(deployer)
agent_count += 1

# Determine agent counts (--num-agents is a shortcut for --num-noise)
num_noise = args.num_noise if args.num_noise is not None else args.num_agents
num_momentum = args.num_momentum
num_value = args.num_value

for i in range(num_noise):
    agent = PerpNoiseAgent(
        id=agent_count,
        name="Noise_{}".format(i),
        type="PerpNoiseAgent",
        symbol=primary_symbol,
        starting_cash=args.starting_cash,
        log_orders=args.log_orders,
        random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
    )
    agents.append(agent)
    agent_count += 1

for i in range(num_momentum):
    agent = PerpMomentumAgent(
        id=agent_count,
        name="Momentum_{}".format(i),
        type="PerpMomentumAgent",
        symbol=primary_symbol,
        starting_cash=args.starting_cash,
        log_orders=args.log_orders,
        random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
    )
    agents.append(agent)
    agent_count += 1

for i in range(num_value):
    agent = PerpValueAgent(
        id=agent_count,
        name="Value_{}".format(i),
        type="PerpValueAgent",
        symbol=primary_symbol,
        starting_cash=args.starting_cash,
        r_bar=dex_config.assets[primary_symbol].initial_oracle_px,
        log_orders=args.log_orders,
        random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
    )
    agents.append(agent)
    agent_count += 1

# ── Chiarella et al. (2002) agents ──────────────────────────────────────
# Pure-type agents: each type has only its own weight enabled.
# For composite agents, use all three weights together via a custom config.

chiarella_configs = [
    ("Fundamentalist", args.num_fundamentalists, 10.0, 0.0, 0.0),
    ("Chartist",       args.num_chartists,       0.0, 10.0, 0.0),
    ("Noise",          args.num_chiarella_noise,  0.0, 0.0, 10.0),
]

for label, count, sf, sc, sn in chiarella_configs:
    for i in range(count):
        agent = ChiarellaAgent(
            id=agent_count,
            name="{}_{}".format(label, i),
            type="ChiarellaAgent",
            symbol=primary_symbol,
            sigma_f=sf, sigma_c=sc, sigma_n=sn,
            sigma_e=args.sigma_e,
            k_max=args.k_max,
            l_min=args.l_min, l_max=args.l_max,
            bias=args.bias, exit_prob=args.exit_prob,
            order_size=args.order_size,
            wake_interval_s=args.wake_interval,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)),
            starting_cash=args.starting_cash,
            log_orders=args.log_orders,
        )
        agents.append(agent)
        agent_count += 1

# ── Kernel ──────────────────────────────────────────────────────────────

kernel = Kernel("HIP3 Perp Simulation", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2**31 - 1)))

# Simple latency: 1ms between all agents
latency = [[1_000_000] * len(agents)] * len(agents)

kernel.runner(
    agents=agents,
    startTime=kernel_start,
    stopTime=kernel_stop,
    agentLatency=latency,
    defaultComputationDelay=1,
    oracle=oracle,
    log_dir="hip3_perp_{}".format(seed),
)

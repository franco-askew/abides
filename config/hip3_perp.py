"""HIP-3 Perpetual Futures Simulation Configuration.

Usage:
    python -u abides.py -c hip3_perp -o oracle_csv=data/oracle.csv -o symbol=ASSET-USD

This config:
  1. Loads deployer_config.json for all HIP-3 parameters.
  2. Builds a CsvOracle from the specified CSV file.
  3. Instantiates the PerpExchangeAgent, OracleDeployerAgent.
  4. Supports user-defined agent injection via the `agents` list.

To add your own agents, either:
  - Edit the agents list below directly.
  - Or import this config from your own script and append to the agents list.
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
from agent.OracleDeployerAgent import OracleDeployerAgent
from util.oracle.CsvOracle import CsvOracle
from util.ContractSpec import load_deployer_config

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
                    help='Number of example noise agents to add (for testing)')
parser.add_argument('--starting-cash', type=float, default=1_000_000.0,
                    help='Starting cash for each agent')
parser.add_argument('--log-orders', action='store_true', default=False)
parser.add_argument('-o', '--override', action='append', default=[],
                    help='Key=value overrides (e.g. -o oracle_csv=data/oracle.csv)')

args, remaining = parser.parse_known_args()

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

# Optional: add example noise agents for testing
for i in range(args.num_agents):
    agent = PerpTradingAgent(
        id=agent_count,
        name="TestAgent_{}".format(i),
        type="PerpTradingAgent",
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

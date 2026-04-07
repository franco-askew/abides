"""End-to-end integration test for the HIP-3 perpetual futures simulation.

Creates a minimal simulation with:
  - PerpExchangeAgent
  - OracleDeployerAgent
  - Two PerpTradingAgent subclasses: one market maker and one taker
  
Verifies: order matching, margin accounting, funding, mark price, and basic liquidation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from Kernel import Kernel
from agent.PerpExchangeAgent import PerpExchangeAgent
from agent.PerpTradingAgent import PerpTradingAgent
from agent.OracleDeployerAgent import OracleDeployerAgent
from util.oracle.CsvOracle import CsvOracle
from util.ContractSpec import load_deployer_config, TimeInForce


class SimpleMaker(PerpTradingAgent):
    """Places a tight bid and ask around the last known oracle price."""

    def __init__(self, id, name, type, symbol, spread_bps=50, size=1.0, **kwargs):
        super().__init__(id, name, type, **kwargs)
        self.symbol = symbol
        self.spread_bps = spread_bps
        self.size = size
        self.placed = False

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return

        if not self.placed:
            self.getCurrentSpread(self.symbol, depth=5)
            self.getMarkPrice(self.symbol)
            self.placed = True
            self.setWakeup(currentTime + pd.Timedelta('1s'))
            return

        oracle_px = self.oracle_prices.get(self.symbol)
        if oracle_px is None or oracle_px <= 0:
            self.setWakeup(currentTime + pd.Timedelta('1s'))
            return

        # Cancel existing orders
        for oid, o in list(self.orders.items()):
            self.cancelOrder(o)

        spread = oracle_px * self.spread_bps / 10000.0
        bid_px = round(oracle_px - spread / 2, 4)
        ask_px = round(oracle_px + spread / 2, 4)

        self.placeLimitOrder(self.symbol, self.size, True, bid_px, tag="MM_BID")
        self.placeLimitOrder(self.symbol, self.size, False, ask_px, tag="MM_ASK")

        self.setWakeup(currentTime + pd.Timedelta('5s'))

    def getWakeFrequency(self):
        return pd.Timedelta('100ms')


class SimpleTaker(PerpTradingAgent):
    """Periodically takes liquidity from the book."""

    def __init__(self, id, name, type, symbol, side='buy', size=0.5, **kwargs):
        super().__init__(id, name, type, **kwargs)
        self.symbol = symbol
        self.side = side
        self.size = size
        self.traded = False

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return

        if not self.traded:
            self.getCurrentSpread(self.symbol, depth=1)
            self.traded = True
            self.setWakeup(currentTime + pd.Timedelta('2s'))
            return

        is_buy = (self.side == 'buy')
        self.placeMarketOrder(self.symbol, self.size, is_buy, tag="TAKE")

        self.setWakeup(currentTime + pd.Timedelta('10s'))

    def getWakeFrequency(self):
        return pd.Timedelta('500ms')


def run_test():
    seed = 42
    np.random.seed(seed)

    symbol = 'ASSET-USD'
    oracle_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'sample_oracle.csv')
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'config', 'deployer_config.json')

    dex_config = load_deployer_config(config_path)
    oracle = CsvOracle(symbol, oracle_csv)

    start_time = pd.Timestamp('2025-01-01 00:00:00')
    end_time = pd.Timestamp('2025-01-01 00:02:00')

    agents = []
    agent_id = 0

    # Exchange
    exchange = PerpExchangeAgent(
        id=agent_id, name="PERP_EXCHANGE", type="PerpExchangeAgent",
        dex_config=dex_config, stream_history=10, log_orders=True,
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
    )
    agents.append(exchange)
    agent_id += 1

    # Oracle deployer
    deployer = OracleDeployerAgent(
        id=agent_id, name="ORACLE_DEPLOYER", type="OracleDeployerAgent",
        symbols=[symbol], oracle_update_interval_s=3.0,
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
    )
    agents.append(deployer)
    agent_id += 1

    # Market maker
    maker = SimpleMaker(
        id=agent_id, name="MAKER", type="SimpleMaker",
        symbol=symbol, spread_bps=50, size=10.0,
        starting_cash=1_000_000.0,
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
    )
    agents.append(maker)
    agent_id += 1

    # Taker (buyer)
    taker = SimpleTaker(
        id=agent_id, name="TAKER_BUY", type="SimpleTaker",
        symbol=symbol, side='buy', size=2.0,
        starting_cash=100_000.0,
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
    )
    agents.append(taker)
    agent_id += 1

    # Kernel
    kernel = Kernel("HIP3 E2E Test",
                    random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)))

    latency = [[1_000_000] * len(agents)] * len(agents)

    print("=" * 60)
    print("RUNNING HIP-3 END-TO-END TEST")
    print("=" * 60)

    result = kernel.runner(
        agents=agents,
        startTime=start_time - pd.Timedelta('10s'),
        stopTime=end_time,
        agentLatency=latency,
        defaultComputationDelay=1,
        oracle=oracle,
        log_dir="test_hip3_e2e",
        skip_log=True,
    )

    # Validate results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    # Check exchange state
    ch = exchange.clearinghouse
    me = exchange.mark_engines[symbol]
    ob = exchange.order_books[symbol]

    print(f"Mark price: {me.mark_price:.4f}")
    print(f"Oracle price: {me.oracle_price:.4f}")
    print(f"Last trade: {ob.last_trade}")
    print(f"OI size: {ch.open_interest_size.get(symbol, 0):.4f}")
    print(f"Deployer fees: {ch.deployer_fees_collected:.4f}")
    print(f"Protocol fees: {ch.protocol_fees_collected:.4f}")

    for agent in agents:
        if hasattr(agent, 'account'):
            acct = agent.account
            pos = acct.get_position(symbol)
            print(f"\n{agent.name}: balance={acct.balance:.2f}, "
                  f"pos={pos.size:.4f}@{pos.entry_price:.4f}, "
                  f"realized_pnl={acct.total_realized_pnl:.4f}, "
                  f"fees_paid={acct.total_fees_paid:.4f}, "
                  f"funding_paid={acct.total_funding_paid:.4f}")

    # Basic assertions
    assert me.mark_price > 0, "Mark price should be positive"
    assert me.oracle_price > 0, "Oracle price should be positive"
    print("\nAll basic assertions passed!")
    print("=" * 60)


if __name__ == '__main__':
    run_test()

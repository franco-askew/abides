# ABIDES-Perp: HIP-3 Perpetual Futures Simulation

Agent-based discrete event simulation of a Hyperliquid HIP-3 builder-deployed perpetual futures market, built on the [ABIDES](https://arxiv.org/abs/1904.12066) framework.

The platform simulates a perpetual contract with an **untradeable underlying** -- the only external input is an oracle price fed via CSV. All market logic (matching, margin, funding, liquidation, mark price, fees) replicates HIP-3 semantics. Agents interact through the ABIDES message-passing kernel with configurable pairwise latencies.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Running a Simulation](#running-a-simulation)
5. [Deployer Configuration Reference](#deployer-configuration-reference)
6. [Oracle CSV Format](#oracle-csv-format)
7. [Writing a Custom Agent](#writing-a-custom-agent)
8. [PerpTradingAgent API Reference](#perptradingagent-api-reference)
9. [HIP-3 Mechanics Reference](#hip-3-mechanics-reference)
10. [Message Protocol](#message-protocol)
11. [Post-Simulation Analysis](#post-simulation-analysis)
12. [End-to-End Test](#end-to-end-test)
13. [Original ABIDES](#original-abides)

---

## Quickstart

```bash
git clone <repo-url>
cd abides
pip install -r requirements.txt

# Run with the sample oracle (5-minute sim, no trading agents)
python abides.py -c hip3_perp -- --oracle-csv data/sample_oracle.csv \
    --start-time "2025-01-01 00:00:00" --end-time "2025-01-01 00:05:00"

# Run with trading agents
python abides.py -c hip3_perp -- --oracle-csv data/sample_oracle.csv \
    --start-time "2025-01-01 00:00:00" --end-time "2025-01-01 00:05:00" \
    --num-agents 5 --log-orders

# Run the end-to-end integration test
python tests/test_perp_e2e.py
```

---

## Architecture Overview

```
                        ┌─────────────────────┐
                        │       Kernel         │
                        │  (priority queue,    │
                        │   latency model,     │
                        │   event dispatch)    │
                        └────────┬────────────┘
                                 │ messages
            ┌────────────────────┼────────────────────┐
            │                    │                     │
   ┌────────▼─────────┐  ┌──────▼───────────┐  ┌─────▼──────────┐
   │OracleDeployerAgent│  │PerpExchangeAgent │  │PerpTradingAgent│
   │                   │  │                  │  │  (your agent)  │
   │ Reads CsvOracle   │  │ ┌──────────────┐ │  │                │
   │ Pushes SET_ORACLE │  │ │PerpOrderBook │ │  │ placeLimitOrder│
   │ every N seconds   │  │ │ (CLOB, float │ │  │ placeMarketOrder
   │                   │  │ │  pricing)    │ │  │ cancelOrder    │
   │ Runtime controls: │  │ ├──────────────┤ │  │ getMarkPrice   │
   │  haltTrading      │  │ │MarkPriceEngine│ │  │ getFundingRate │
   │  setOiCaps        │  │ │ (HIP-3 rules)│ │  │ getPosition    │
   │  setFundingMult   │  │ ├──────────────┤ │  │ ...            │
   │  setMarginTable   │  │ │ Clearinghouse│ │  └────────────────┘
   └───────────────────┘  │ │ (margin, OI, │ │
                          │ │  fees, PnL)  │ │
                          │ ├──────────────┤ │
                          │ │FundingEngine │ │
                          │ │ (5s sampling, │ │
                          │ │  hourly settle)│ │
                          │ ├──────────────┤ │
                          │ │LiquidationEng│ │
                          │ │ (mark-based, │ │
                          │ │  partial/    │ │
                          │ │  backstop)   │ │
                          │ └──────────────┘ │
                          └──────────────────┘
```

**Data flow:**

1. The **Kernel** drives simulation time. It delivers messages from a priority queue ordered by timestamp, applying configurable latency between each agent pair.
2. The **OracleDeployerAgent** wakes every N seconds (default 3s), reads the oracle price from `CsvOracle`, and sends a `SET_ORACLE` message to the exchange.
3. The **PerpExchangeAgent** processes `SET_ORACLE` by updating the mark price engine, then scans for liquidations. It also schedules internal premium sampling (every 5s) and funding settlement (every hour).
4. **PerpTradingAgent** subclasses wake on their own schedule, observe market data, and send orders to the exchange. The exchange matches orders, updates the clearinghouse, and sends execution confirmations back.

**Key design properties:**

- **24/7 trading** -- no market open/close guards on the order flow. The kernel `startTime`/`stopTime` bounds the simulation window.
- **Float pricing** -- all prices are native Python floats, not integer cents.
- **No tradeable spot** -- the only external price is the oracle. The perp market is the sole venue for speculation.
- **Existing equity simulation untouched** -- the perp system is built as new files alongside the original ABIDES equity agents and order book.

---

## Directory Structure

### Perp-specific files (new)

```
abides/
├── config/
│   ├── deployer_config.json      # All HIP-3 deployer-tunable parameters
│   └── hip3_perp.py              # Simulation config / entry point
│
├── data/
│   └── sample_oracle.csv         # Example oracle data (72 points, 1 hour)
│
├── agent/
│   ├── PerpExchangeAgent.py      # Central matching + clearing exchange
│   ├── PerpTradingAgent.py       # Base class for user strategies
│   └── OracleDeployerAgent.py    # HIP-3 deployer (oracle pusher)
│
├── util/
│   ├── ContractSpec.py           # Dataclasses: ContractSpec, MarginTable,
│   │                             #   PerpDexConfig, FeeSchedule, enums
│   ├── PerpOrderBook.py          # CLOB with float pricing, TIF, STP
│   ├── MarkPriceEngine.py        # HIP-3 mark price with clamping rules
│   ├── PerpAccount.py            # Position, cross/isolated margin, PnL
│   ├── Clearinghouse.py          # Multi-account margin, OI, fees
│   ├── FundingEngine.py          # HL funding rate formula
│   ├── LiquidationEngine.py      # Mark-based liquidation with partial/backstop
│   ├── order/
│   │   └── PerpLimitOrder.py     # Float-price order with TIF, reduce-only,
│   │                             #   trigger, liquidation flag
│   └── oracle/
│       └── CsvOracle.py          # CSV (timestamp, price) oracle adapter
│
├── tests/
│   └── test_perp_e2e.py          # End-to-end integration test
│
└── README.md                     # This file
```

### Core ABIDES files (unchanged, reused)

```
abides/
├── abides.py                     # Main launcher: python abides.py -c <config>
├── Kernel.py                     # Discrete event kernel (priority queue)
├── message/Message.py            # Message and MessageType classes
├── agent/
│   ├── Agent.py                  # Base agent class
│   ├── FinancialAgent.py         # Financial agent base
│   ├── TradingAgent.py           # Original equity trading agent
│   └── ExchangeAgent.py          # Original equity exchange agent
├── util/
│   ├── OrderBook.py              # Original equity order book
│   └── oracle/
│       ├── ExternalFileOracle.py
│       ├── MeanRevertingOracle.py
│       └── SparseMeanRevertingOracle.py
└── model/
    └── LatencyModel.py           # Pairwise agent latency model
```

---

## Running a Simulation

### Via the ABIDES launcher

```bash
python abides.py -c hip3_perp -- [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--oracle-csv PATH` | from deployer_config.json | Path to oracle CSV file |
| `--symbol SYMBOL` | first asset in config | Which asset to simulate |
| `--deployer-config PATH` | `config/deployer_config.json` | Path to deployer config |
| `--start-time ISO` | `2025-01-01 00:00:00` | Simulation start |
| `--end-time ISO` | `2025-01-01 01:00:00` | Simulation end |
| `--seed INT` | `42` | Random seed |
| `--num-agents INT` | `0` | Number of placeholder `PerpTradingAgent` instances |
| `--starting-cash FLOAT` | `1000000` | Starting balance per agent |
| `--log-orders` | off | Log all order activity to agent logs |
| `-o key=value` | -- | Override any parameter (e.g. `-o oracle_csv=my.csv`) |

### Programmatic usage

For full control, write your own config script. This is the recommended approach for research:

```python
import numpy as np
import pandas as pd

from Kernel import Kernel
from agent.PerpExchangeAgent import PerpExchangeAgent
from agent.PerpTradingAgent import PerpTradingAgent
from agent.OracleDeployerAgent import OracleDeployerAgent
from util.oracle.CsvOracle import CsvOracle
from util.ContractSpec import load_deployer_config

# Load config
dex_config = load_deployer_config('config/deployer_config.json')
oracle = CsvOracle('ASSET-USD', 'data/my_oracle.csv')

# Build agents
agents = []
agent_id = 0

exchange = PerpExchangeAgent(
    id=agent_id, name="EXCHANGE", type="PerpExchangeAgent",
    dex_config=dex_config,
    random_state=np.random.RandomState(seed=1),
)
agents.append(exchange)
agent_id += 1

deployer = OracleDeployerAgent(
    id=agent_id, name="DEPLOYER", type="OracleDeployerAgent",
    symbols=['ASSET-USD'],
    random_state=np.random.RandomState(seed=2),
)
agents.append(deployer)
agent_id += 1

# Add your custom agents here...
# my_agent = MyStrategy(id=agent_id, ...)
# agents.append(my_agent)
# agent_id += 1

# Run
kernel = Kernel("My Sim", random_state=np.random.RandomState(seed=0))
latency = [[1_000_000] * len(agents)] * len(agents)

kernel.runner(
    agents=agents,
    startTime=pd.Timestamp('2025-01-01') - pd.Timedelta('1min'),
    stopTime=pd.Timestamp('2025-01-01 01:00:00') + pd.Timedelta('1min'),
    agentLatency=latency,
    defaultComputationDelay=1,
    oracle=oracle,
    log_dir="my_sim_output",
)

# Access post-sim state
ch = exchange.clearinghouse
for aid, account in ch.accounts.items():
    print(f"Agent {aid}: balance={account.balance:.2f}")
```

---

## Deployer Configuration Reference

All HIP-3 deployer actions are stored in `config/deployer_config.json`. Every parameter is configurable -- nothing is hardcoded.

```jsonc
{
  "dex": {
    "name": "SIM_DEX",                   // DEX identifier
    "collateral_token": "USDC",          // Collateral denomination
    "fee_schedule": {
      "maker_fee_bps": 2.0,              // Maker fee in basis points
      "taker_fee_bps": 7.0,              // Taker fee in basis points
      "deployer_share": 0.5,             // Fraction of fees to deployer
      "protocol_share": 0.5              // Fraction of fees to protocol
    }
  },
  "assets": [
    {
      "coin": "ASSET-USD",               // Symbol name
      "sz_decimals": 4,                  // Size decimal precision
      "initial_oracle_px": 100.0,        // Opening price
      "oracle_csv": "data/oracle.csv",   // Default oracle CSV path
      "margin_mode": "normal",           // "normal" | "noCross" | "strictIsolated"
      "margin_table": {
        "tiers": [                        // Tiered margin schedule
          {"lower_bound_notional": 0, "max_leverage": 20},
          {"lower_bound_notional": 3000000, "max_leverage": 10}
        ]
      },
      "funding_impact_notional": 6000.0, // Notional for impact price calc
      "funding_multiplier": 1.0,         // 0 to 10, scales funding rate
      "oi_cap_notional": 50000000,       // Max OI in notional terms
      "oi_cap_size": 1000000000,         // Max OI in contract size terms
      "max_market_order_value": 500000   // Max single market order
    }
  ],
  "oracle_update_interval_s": 3.0,       // How often deployer pushes oracle
  "deployer_mark_px_mode": "none",       // "none" | "oracle_based"
  "external_perp_px_mode": "ema_of_mark" // "ema_of_mark" | "none"
}
```

**Margin modes:**

| Mode | Description |
|---|---|
| `normal` | Agent may choose cross or isolated margin per position |
| `noCross` | Isolated only, but agents may manually remove excess margin |
| `strictIsolated` | Isolated only, no manual margin removal |

**Margin table tiers:** Each tier sets the maximum leverage for positions above a given notional threshold. Maintenance margin is always half of initial margin. Tiers are continuous (deductions are auto-computed so there are no jumps at boundaries).

---

## Oracle CSV Format

The oracle CSV must have two columns: a timestamp and a price. A header row is expected by default.

```csv
timestamp,price
1735689600.000,100.00
1735689600.100,100.02
1735689600.200,100.05
```

**Timestamp formats accepted:**

- **Unix epoch float** (seconds since 1970-01-01, sub-second precision via decimals)
- **ISO datetime strings** (e.g. `2025-01-01 00:00:00.100`)

The oracle uses `numpy.searchsorted` for O(log n) lookups, so files with millions of rows perform well. The oracle returns the most recent price at or before the query time (no interpolation, step function).

---

## Writing a Custom Agent

Subclass `PerpTradingAgent` and override `wakeup()` to implement your strategy. The base class handles all message routing, margin accounting, and order tracking automatically.

```python
from agent.PerpTradingAgent import PerpTradingAgent
from util.ContractSpec import TimeInForce
import pandas as pd


class MyStrategy(PerpTradingAgent):

    def __init__(self, id, name, type, symbol, **kwargs):
        super().__init__(id, name, type, **kwargs)
        self.symbol = symbol
        self.initialized = False

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return

        if not self.initialized:
            # Subscribe to live book updates (top 5 levels, every 100ms)
            self.requestDataSubscription(self.symbol, levels=5, freq=100_000_000)
            self.getMarkPrice(self.symbol)
            self.initialized = True
            self.setWakeup(currentTime + pd.Timedelta('1s'))
            return

        # Access cached state
        oracle_px = self.oracle_prices.get(self.symbol)
        mark_px = self.mark_prices.get(self.symbol)
        position = self.getPositionSize(self.symbol)
        balance = self.getBalance()
        best_bid, best_ask = self.getKnownBidAsk(self.symbol)

        # Place orders
        if position == 0 and oracle_px:
            self.placeLimitOrder(
                self.symbol,
                quantity=1.0,
                is_buy_order=True,
                limit_price=oracle_px * 0.999,
                tag="ENTRY",
                time_in_force=TimeInForce.GTC,
            )

        # Close position with reduce-only
        if position > 0 and mark_px and mark_px > oracle_px * 1.005:
            self.placeMarketOrder(
                self.symbol,
                quantity=abs(position),
                is_buy_order=False,
                reduce_only=True,
                tag="TAKE_PROFIT",
            )

        # Schedule next wakeup
        self.setWakeup(currentTime + pd.Timedelta('5s'))

    def getWakeFrequency(self):
        return pd.Timedelta('100ms')
```

**Lifecycle of a custom agent:**

1. The kernel calls `kernelStarting()` -- the agent discovers the exchange.
2. The agent receives `WHEN_MKT_OPEN` and `WHEN_MKT_CLOSE` from the exchange, then schedules its first wakeup via `getWakeFrequency()`.
3. On each `wakeup()`, the agent reads cached state and places/cancels orders.
4. Execution confirmations (`ORDER_EXECUTED`), funding payments (`FUNDING_PAYMENT`), and liquidation notices (`LIQUIDATED`) arrive via `receiveMessage()` and are handled automatically by the base class.
5. At shutdown, `kernelStopping()` logs final positions and equity.

---

## PerpTradingAgent API Reference

### Order Placement

| Method | Description |
|---|---|
| `placeLimitOrder(symbol, quantity, is_buy_order, limit_price, order_id=None, tag=None, time_in_force=GTC, reduce_only=False)` | Place a limit order. `time_in_force` can be `GTC`, `IOC`, or `ALO` (post-only). |
| `placeMarketOrder(symbol, quantity, is_buy_order, order_id=None, tag=None, reduce_only=False)` | Place a market order (internally IOC at extreme price). |
| `cancelOrder(order)` | Cancel a resting order. Pass the order object from `self.orders`. |
| `modifyOrder(order, new_order)` | Replace an existing order with a new one (same order_id). |

### Market Data Queries

All queries are **asynchronous** -- they send a message to the exchange and the response arrives in the next `receiveMessage()` cycle. Results are cached automatically.

| Method | Cached result |
|---|---|
| `getLastTrade(symbol)` | `self.last_trade[symbol]` |
| `getCurrentSpread(symbol, depth=1)` | `self.known_bids[symbol]`, `self.known_asks[symbol]` |
| `getMarkPrice(symbol)` | `self.mark_prices[symbol]`, `self.oracle_prices[symbol]` |
| `getFundingRate(symbol)` | `self.funding_rates[symbol]` |
| `getPosition()` | Syncs `self.account.balance` with exchange |
| `requestDataSubscription(symbol, levels, freq)` | Pushes `MARKET_DATA` to agent on book changes. `freq` is min ns between updates (0 = every change). Results cached in `self.known_bids`, `self.known_asks`, `self.mark_prices`, `self.oracle_prices`. |
| `cancelDataSubscription(symbol)` | Stop receiving `MARKET_DATA` pushes. |

### Convenience Accessors (synchronous, from cache)

| Method | Returns |
|---|---|
| `getPositionSize(symbol)` | Signed position size (float) |
| `getBalance()` | Current USDC balance (float) |
| `getKnownBidAsk(symbol)` | `(best_bid, best_ask)` or `(None, None)` |
| `getKnownMidPrice(symbol)` | Mid-price or `None` |

### Agent State

| Attribute | Description |
|---|---|
| `self.account` | `PerpAccount` instance -- local mirror of positions, balance, PnL |
| `self.orders` | `dict` of `order_id -> PerpLimitOrder` for open orders |
| `self.mark_prices` | `dict` of `symbol -> float` (last known mark price) |
| `self.oracle_prices` | `dict` of `symbol -> float` (last known oracle price) |
| `self.funding_rates` | `dict` of `symbol -> float` (last known hourly funding rate) |
| `self.known_bids[symbol]` | `list` of `(price, qty)` tuples, best first |
| `self.known_asks[symbol]` | `list` of `(price, qty)` tuples, best first |

---

## HIP-3 Mechanics Reference

### Mark Price

The mark price is computed on every `SET_ORACLE` message:

```
local_mark = median(best_bid, best_ask, last_trade)
raw_mark   = median(deployer_mark_pxs + [local_mark])
```

Then clamped:
- **1% per-update**: `raw_mark` is clamped to within 1% of the previous mark.
- **10x daily**: cannot exceed 10x or fall below 1/10x of the start-of-day price.
- **OI guard**: update is rejected if `OI * new_mark > 10 * OI_cap`.
- **Staleness**: if no `SET_ORACLE` for 10 seconds, mark falls back to the local book mark (still 1%-clamped).

### Funding Rate

Exact replication of the Hyperliquid formula:

1. Every **5 simulated seconds**, sample the premium:
   ```
   impact_diff = max(impact_bid - oracle, 0) - max(oracle - impact_ask, 0)
   premium = impact_diff / oracle_price
   ```
   where `impact_bid`/`impact_ask` are the average execution prices to fill `funding_impact_notional` USDC on each side.

2. Every **hour**, compute the funding rate:
   ```
   avg_premium = mean(all premium samples this hour)
   clamped = clamp(0.0001 - avg_premium, -0.0005, 0.0005)
   eight_hour_rate = (avg_premium + clamped) * funding_multiplier
   hourly_rate = eight_hour_rate / 8
   ```
   Capped at 4% per hour in either direction.

3. Payments: `payment = position_size * oracle_price * hourly_rate`. Positive rate means longs pay shorts.

### Margin

- **Initial margin** = `notional / leverage`. Checked on every order that increases a position.
- **Maintenance margin** = `initial_margin / 2`, computed per tier from the margin table with continuous deductions (no cliff at tier boundaries).
- **Cross margin**: one shared balance covers all cross-mode positions. Account value = `balance + sum(unrealized_pnl)`.
- **Isolated margin**: each position has its own locked margin. Account value = `isolated_margin + unrealized_pnl`.
- Margin mode is per-asset, set by the deployer in `deployer_config.json`.

### Liquidation

- Triggered on every mark price update.
- **Market liquidation**: if `account_value < maintenance_margin`, a market IOC order is sent to close the position.
  - For positions > 100k USDC notional: only 20% is liquidated, with a 30-second cooldown before the next chunk.
  - For smaller positions: full liquidation.
- **Backstop liquidation**: if `account_value < (2/3) * maintenance_margin`, the position is forcibly transferred to the backstop vault at mark price (no order book interaction).

### Order Types

| Type | Behavior |
|---|---|
| **GTC** (Good-Til-Cancel) | Rests on book until filled or cancelled. Default. |
| **IOC** (Immediate-or-Cancel) | Fills what it can immediately, cancels the rest. |
| **ALO** (Add-Liquidity-Only / Post-Only) | Rejected if it would immediately match. Guarantees maker fee. |
| **Reduce-Only** | Can only reduce/close an existing position. Quantity auto-clipped to position size. |
| **Trigger orders** (Stop/Take) | Stored by the exchange, activated when mark price crosses the trigger. Supports `STOP_MARKET`, `STOP_LIMIT`, `TAKE_MARKET`, `TAKE_LIMIT`. |
| **Liquidation orders** | Internal IOC orders generated by the liquidation engine. Zero fees. |

### Self-Trade Prevention

If the best resting order on the opposite side belongs to the same agent as the incoming order, the resting order is cancelled (not matched).

### Open Interest Caps

Every order that would increase a position is checked against the deployer-configured OI caps:
- `oi_cap_size`: max total OI in contract units.
- `oi_cap_notional`: max total OI in USDC terms.

If either cap would be breached, the order is rejected with reason `OI_CAP`.

### Fee Schedule

Fees are calculated as `notional * fee_rate` and split between deployer and protocol:
- Default: 0.02% maker, 0.07% taker (2x standard Hyperliquid fees, per HIP-3 spec).
- Liquidation orders pay zero fees.
- Configurable via `deployer_config.json`.

---

## Message Protocol

All communication between agents is via `Message` objects routed through the Kernel. The perp system adds these message types:

### Agent -> Exchange

| Message | Fields | Description |
|---|---|---|
| `LIMIT_ORDER` | `sender`, `order` (PerpLimitOrder) | Place a limit order |
| `MARKET_ORDER` | `sender`, `order` (PerpLimitOrder) | Place a market order |
| `CANCEL_ORDER` | `sender`, `order` | Cancel a resting order |
| `MODIFY_ORDER` | `sender`, `order`, `new_order` | Replace a resting order |
| `QUERY_MARK_PRICE` | `sender`, `symbol` | Request current mark/oracle price |
| `QUERY_FUNDING_RATE` | `sender`, `symbol` | Request last funding rate |
| `QUERY_POSITION` | `sender` | Request authoritative account state |
| `QUERY_SPREAD` | `sender`, `symbol`, `depth` | Request order book depth |
| `QUERY_LAST_TRADE` | `sender`, `symbol` | Request last trade price |

### Exchange -> Agent

| Message | Fields | Description |
|---|---|---|
| `ORDER_EXECUTED` | `order` (with fill_price, quantity) | Partial or full fill notification |
| `ORDER_ACCEPTED` | `order` | Order resting on book |
| `ORDER_CANCELLED` | `order` | Order removed from book |
| `ORDER_REJECTED` | `order`, `reason` | Order rejected (reasons: `INSUFFICIENT_MARGIN`, `OI_CAP`, `POST_ONLY_WOULD_MATCH`, `REDUCE_ONLY_NO_POSITION`, `HALTED`) |
| `FUNDING_PAYMENT` | `symbol`, `payment`, `rate`, `oracle_price` | Hourly funding settlement |
| `LIQUIDATED` | `symbol`, `type`, `quantity` or `settled_pnl` | Liquidation notification |
| `POSITION_SETTLED` | `symbol`, `settled_pnl`, `mark_price` | Position settled by deployer halt |
| `MARKET_DATA` | `symbol`, `bids`, `asks`, `last_transaction`, `mark_price`, `oracle_price`, `exchange_ts` | Subscription push |
| `QUERY_MARK_PRICE` | `symbol`, `mark_price`, `oracle_price` | Response to query |
| `QUERY_FUNDING_RATE` | `symbol`, `funding_rate` | Response to query |
| `QUERY_POSITION` | `positions`, `balance` | Authoritative account state |

### Deployer -> Exchange

| Message | Fields | Description |
|---|---|---|
| `SET_ORACLE` | `oracle_pxs`, `mark_pxs`, `external_perp_pxs` | Push oracle/mark prices |
| `HALT_TRADING` | `symbol`, `is_halted` | Halt or resume trading (settles all positions when halting) |
| `SET_OI_CAPS` | `symbol`, `notional_cap`, `size_cap` | Update OI caps |
| `SET_FUNDING_MULTIPLIERS` | `multipliers` (dict of symbol -> float) | Update funding multipliers |
| `SET_MARGIN_TABLE` | `symbol`, `tiers` | Update margin tiers |

---

## Post-Simulation Analysis

### Accessing exchange state after simulation

After `kernel.runner()` returns, you can inspect the exchange agent directly:

```python
# Mark prices
for symbol, me in exchange.mark_engines.items():
    print(f"{symbol}: mark={me.mark_price:.4f}, oracle={me.oracle_price:.4f}")

# All accounts
ch = exchange.clearinghouse
for agent_id, account in ch.accounts.items():
    equity = account.total_equity({s: me.mark_price for s, me in exchange.mark_engines.items()})
    print(f"Agent {agent_id}: balance={account.balance:.2f}, equity={equity:.2f}")
    for sym, pos in account.positions.items():
        if pos.size != 0:
            print(f"  {sym}: size={pos.size:.4f} entry={pos.entry_price:.4f}")

# Fee totals
print(f"Deployer fees: {ch.deployer_fees_collected:.2f}")
print(f"Protocol fees: {ch.protocol_fees_collected:.2f}")

# Funding rates
for sym, rate in exchange.funding_engine.last_funding_rates.items():
    print(f"{sym} last hourly funding: {rate:.8f}")

# Open interest
for sym, oi in ch.open_interest_size.items():
    print(f"{sym} OI: {oi:.4f} contracts")
```

### Log files

When `skip_log=False` (default), ABIDES writes compressed log files to `log/<log_dir>/`:
- One `.bz2` file per agent with timestamped events.
- A `summary_log.bz2` with key aggregates.
- If `log_orders=True`, every order submission, execution, cancellation, and rejection is logged.

Read logs with:
```python
import pandas as pd
df = pd.read_pickle('log/hip3_perp_42/PERP_EXCHANGE.bz2')
```

---

## End-to-End Test

`tests/test_perp_e2e.py` runs a complete simulation with a market maker and a taker:

```bash
python tests/test_perp_e2e.py
```

The test verifies:
- Oracle deployment and mark price updates
- Order matching and trade execution
- Clearinghouse fee accounting (deployer/protocol split)
- Open interest tracking
- Agent position and balance tracking
- Funding premium sampling
- Mark price clamping behavior

The test defines two example agent subclasses (`SimpleMaker` and `SimpleTaker`) that serve as templates for building your own strategies.

---

## Original ABIDES

The original equity simulation is fully intact and can still be run:

```bash
python abides.py -c rmsc02
```

See the [original ABIDES paper](https://arxiv.org/abs/1904.12066) and [wiki](https://github.com/abides-sim/abides/wiki) for documentation on equity configurations, background agents (ZI, noise, value, momentum, HBL), and stylized fact replication.

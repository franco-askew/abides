# Perpetual Futures Agents

This document describes the six perpetual futures strategy archetypes used in this repository:

1. `PerpNoiseAgent`
2. `PerpMomentumAgent`
3. `PerpValueAgent`
4. Chiarella `Fundamentalist`
5. Chiarella `Chartist`
6. Chiarella `Noise`

The source implementations live in:

- `agent/PerpNoiseAgent.py`
- `agent/PerpMomentumAgent.py`
- `agent/PerpValueAgent.py`
- `agent/ChiarellaAgent.py`
- `config/hip3_perp.py` for the configured Chiarella presets

## Shared execution framework

All six strategies inherit from `PerpTradingAgent`, so they share the same perpetual-futures trading infrastructure even though their signals differ.

Common behavior across all of them:

- They trade against `PerpExchangeAgent` and use the same account, fill, funding, and liquidation plumbing.
- They wait until market open/close times are known before trading.
- They round quantities and prices to the contract's symbol-specific precision rules.
- They keep a local mirror of balances and positions and log final equity at shutdown.
- They can be constrained by:
  - `max_position_size`
  - `max_live_orders_per_symbol`
  - `opening_order_cooldown_after_unfunded_s`
  - `max_take_distance_bps_from_mark`
  - `max_passive_distance_bps_from_mark`
- They use cached market data helpers such as best bid/ask, mid price, mark price, oracle price, and last trade.

Important shared safeguards:

- Opening trades are blocked for a cooldown period after an `UNFUNDED_ACCOUNT` rejection.
- Orders that would grow exposure beyond `max_position_size` are skipped.
- Passive quotes are clipped into a configurable band around the reference price, usually the mark price when available.
- Aggressive orders are blocked if the touch is too far from the reference price.
- Most strategies cancel their own prior resting order on the symbol before replacing it, so they usually maintain at most one active quote per symbol by default.

## 1. `PerpNoiseAgent`

### What it is

`PerpNoiseAgent` is the simplest standalone perpetual-futures trader in the repo. It injects random order flow without trying to estimate fair value or detect trends.

### Core decision process

On each wake:

1. It flips a Bernoulli trial with probability `trade_probability`.
2. If it decides to trade, it chooses buy vs sell uniformly at random.
3. It draws an order size uniformly from `[min_size, max_size]`.
4. It submits a passive limit order on the chosen side.

This means it is random in:

- whether it trades
- which direction it trades
- how large the order is
- the exact passive quote price

### Market data and wake model

- Default wake frequency: `30s`
- Optional delayed start: `wakeup_time`
- On first usable wake it requests a level-1 market-data subscription at its wake frequency
- It then trades from cached top-of-book data rather than querying the spread each cycle

### Order placement style

`PerpNoiseAgent` no longer posts exactly at the touch. Instead, it uses `_strategy_passive_price_from_mid(...)` with a random offset between `5` and `25` bps from the current mid price.

Consequences:

- Buy orders are normally posted below the mid.
- Sell orders are normally posted above the mid.
- The agent is intended to provide passive random flow rather than blindly cross the spread.
- If the quote would accidentally become marketable, it is skipped.

Before placing a new order it:

- checks whether the order would violate position or funding cooldown rules
- cancels existing orders on the symbol
- checks open-order capacity again

### Position and risk behavior

- If `max_position_size` is unset, it uses an effective cap of `5 x current_order_size`
- Default `max_live_orders_per_symbol` is `1`
- It uses the shared take/passive price-band guards from `PerpTradingAgent`

### What role it plays in simulations

This agent is best thought of as background, directionally uninformative order flow. It is useful for:

- adding non-strategic activity
- generating passive liquidity turnover
- stress-testing inventory and order-management logic
- creating a baseline population that is simpler than the Chiarella family

### Key parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `min_size` | `0.1` | Minimum random order size |
| `max_size` | `1.0` | Maximum random order size |
| `wake_up_freq` | `"30s"` | Strategy cadence |
| `wakeup_time` | `None` | Optional delayed activation time |
| `trade_probability` | `0.20` | Chance of attempting a trade on each wake |

## 2. `PerpMomentumAgent`

### What it is

`PerpMomentumAgent` is a trend-following agent based on a moving-average crossover computed from mid prices.

### Core signal

It maintains a history of mid prices and computes:

- a 20-observation moving average
- a 50-observation moving average

Its signal is:

- bullish if `MA20 > MA50`
- bearish if `MA20 < MA50`

In code, the actual decision variable is:

`spread_signal_bps = ((MA20 - MA50) / mid) * 10000`

Trading is suppressed if the absolute spread signal is smaller than `crossover_deadband_bps`.

### Trade gating logic

The momentum agent has two important filters:

- `crossover_deadband_bps`
  - avoids trading on tiny MA differences
- `trade_on_signal_change_only`
  - if enabled, the agent only trades when the sign of the signal changes relative to `last_trade_signal`

That gives it a less chatty profile than a naive crossover system.

### Market data and wake model

It supports two data modes:

- Polling mode, default:
  - wake every `wake_up_freq` (default `300s`)
  - request the current spread
  - trade when the `QUERY_SPREAD` response arrives
- Subscription mode:
  - if `subscribe=True`, request level-1 market data once
  - update and trade from `MARKET_DATA` pushes

### Order placement style

The order size is sampled once at initialization from `[min_size, max_size]` and then reused for every trade.

When it trades:

- bullish signal -> place passive buy order
- bearish signal -> place passive sell order

Like the noise agent, it uses a passive quote `5` to `25` bps away from the mid, not a market order. It also cancels existing symbol orders before replacing them.

### Position and risk behavior

- If `max_position_size` is unset, default cap is `5 x size`
- Default `max_live_orders_per_symbol` is `1`
- It inherits the same unfunded-account cooldown and mark-band protections as the other perp agents

### What role it plays in simulations

This is the cleanest pure trend follower in the perp strategy set. It is useful for:

- creating directional pressure when prices trend
- contrasting momentum flow against value-reversion flow
- testing how longer-horizon signals interact with passive order placement

### Key parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `min_size` | `0.1` | Lower bound for initial size draw |
| `max_size` | `1.0` | Upper bound for initial size draw |
| `wake_up_freq` | `"300s"` | Poll cadence |
| `subscribe` | `False` | Use streaming market data instead of polling |
| `trade_on_signal_change_only` | `True` | Only trade when signal sign flips |
| `crossover_deadband_bps` | `10.0` | Ignore weak crossover signals |

## 3. `PerpValueAgent`

### What it is

`PerpValueAgent` is the standalone fundamental/value trader. It observes the oracle with noise, updates a latent value estimate, projects a terminal value, and trades when the market looks mispriced versus that estimate.

### Fundamental model

Its state variables are:

- `r_t`: current filtered value estimate
- `sigma_t`: uncertainty estimate
- `prev_wake_time`: previous observation time

At each decision point it:

1. Observes the oracle price with observation noise `sigma_n`
2. Mean-reverts the prior estimate toward `r_bar` at speed `kappa`
3. Evolves uncertainty using `sigma_s`
4. Applies a Bayesian-style update using the new noisy observation
5. Projects the estimate forward to market close to get `r_T`

`r_T` is the agent's estimate of terminal fair value.

### Wake model

Unlike the fixed-interval noise and momentum agents, the value agent wakes on an exponential clock.

- `mean_wake_interval_s` is the mean inter-arrival time
- each wake schedules the next wake by sampling `Exp(mean_wake_interval_s)`
- the minimum delay is clamped to `1s`

This makes it behave more like a Poisson-arrival informed trader.

### Trading rule

After estimating `r_T`, the agent compares it to the current mid price:

- if `|r_T - mid| / mid` is below `mispricing_deadband_bps`, it does nothing
- if `r_T > mid`, it wants to buy
- if `r_T < mid`, it wants to sell

### Order placement style

The value agent has a mixed passive/aggressive style:

- Most of the time it posts a passive limit order `5` to `25` bps away from the mid.
- With probability `aggressive_cross_prob`, it submits a market order instead, but only if the touch is within the permitted take band.

As with the other agents, it cancels previous symbol orders before replacing them.

### Position and risk behavior

- The trade size is sampled once at initialization from `[min_size, max_size]`
- If `max_position_size` is unset, default cap is `5 x size`
- It logs final valuation against a zero-noise oracle observation at shutdown

### Parameter aliases and compatibility notes

The class preserves older parameter names:

- `lambda_a` is deprecated in favor of `mean_wake_interval_s`
- `percent_aggr` is treated as an alias for `aggressive_cross_prob`

### What role it plays in simulations

This is the repo's cleanest standalone informed trader. It is useful for:

- mean-reversion experiments
- comparing informed vs uninformed order flow
- testing market response to a trader that sometimes crosses and sometimes quotes passively

### Key parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `sigma_n` | `1.0` | Observation noise on oracle price |
| `r_bar` | `100.0` | Long-run mean value |
| `kappa` | `0.05` | Mean-reversion speed |
| `sigma_s` | `1.0` | State-noise term in value evolution |
| `mean_wake_interval_s` | `300.0` | Mean exponential wake interval |
| `mispricing_deadband_bps` | `15.0` | Minimum value-vs-mid gap needed to trade |
| `aggressive_cross_prob` | `0.02` | Chance of using a market order |
| `min_size` | `0.1` | Lower bound for initial size draw |
| `max_size` | `1.0` | Upper bound for initial size draw |

## 4-6. The Chiarella family

### Important structural note

There is only one `ChiarellaAgent` class in the codebase, but `config/hip3_perp.py` instantiates three distinct perpetual-futures archetypes from it:

- `Fundamentalist`: `sigma_f=10.0`, `sigma_c=0.0`, `sigma_n=0.0`
- `Chartist`: `sigma_f=0.0`, `sigma_c=10.0`, `sigma_n=0.0`
- `Noise`: `sigma_f=0.0`, `sigma_c=0.0`, `sigma_n=10.0`

Because only one sigma is nonzero in each preset, the normalized forecast weights become effectively:

- fundamentalist: `w_f = 1`
- chartist: `w_c = 1`
- noise: `w_n = 1`

So the three configured Chiarella agents behave as pure special cases of the same composite framework.

### Shared Chiarella mechanics

All three Chiarella variants share the following execution pattern.

#### Data collection

- On startup they randomize their first wake within one `wake_interval_s` window so the population is desynchronized.
- They request a level-1 market-data subscription once, then trade from cached `MARKET_DATA`.
- Each decision snapshot uses:
  - oracle price
  - mark price
  - best bid and ask
  - last trade

#### Price series they maintain

They append to three histories on each usable wake:

- `oracle_history`
- `perp_history`
- `premium_history = perp_price - oracle_price`

If best bid and ask are present, the perp price is the mid. Otherwise they fall back to mark price or last trade.

#### Exit behavior

If the agent already has a nonzero position, it may flatten before generating a new entry signal:

- with probability `exit_prob`
- by sending a reduce-only market order
- only if the touch is within the allowed take band

This is a major difference from the standalone noise and momentum agents, which do not have this built-in probabilistic exit rule.

#### Positional vs premium mode

Each wake, the agent samples whether to think in:

- positional mode with probability `bias`
- premium mode with probability `1 - bias`

Positional mode:

- signal series: perpetual price history
- current value: current perp price
- anchor value: oracle price

Premium mode:

- signal series: premium history
- current value: current premium
- anchor value: `0`

If premium history has negative values, it is shifted upward before chartist percentage-return calculations so the denominator stays positive.

#### Forecast-to-order mapping

Every Chiarella agent computes a forecast return and then turns that into a quote:

1. Compute forecast return
2. Skip if the absolute value is below `forecast_deadband_bps`
3. Clamp forecast return to `[-5%, +5%]`
4. Convert it to a candidate price with `perp_px * exp(forecast_return)`
5. Add a favorable slack using `k_max`
   - buy orders are shaded lower
   - sell orders are shaded higher
6. Clip the final price into the passive mark-price band
7. Cancel prior symbol orders and submit a new GTC limit order

`k_max` is implemented as a percentage-scale slack parameter. With the default `k_max=0.15`, the extra shading is sampled uniformly between `0` and `0.15%` of price, which is up to `15` bps.

### 4. Chiarella Fundamentalist

#### What it is

This is the pure Chiarella fundamental-reversion trader created by setting only `sigma_f` nonzero.

#### Forecast component

Its forecast is entirely:

`f_f = fundamentalist_forecast(current_value, anchor_value)`

The implementation is:

- if both values are positive, use `log(anchor / current)`
- otherwise use a scaled linear difference

#### Economic interpretation

In positional mode:

- `current_value = perp price`
- `anchor_value = oracle price`

So it buys when the perpetual is below the oracle and sells when the perpetual is above the oracle.

In premium mode:

- `current_value = premium`
- `anchor_value = 0`

So it trades toward premium mean reversion: negative premium implies buy pressure, positive premium implies sell pressure.

#### How it differs from `PerpValueAgent`

Although both are value-oriented, they are not the same:

- `PerpValueAgent` maintains a dynamic latent-value estimate with explicit noisy observation and time evolution.
- Chiarella Fundamentalist directly converts the current price/oracle or premium gap into a return forecast.
- `PerpValueAgent` sometimes uses market orders; Chiarella Fundamentalist always enters with a GTC limit order.
- Chiarella Fundamentalist includes the Chiarella-specific exit probability and positional-vs-premium mode switch.

#### Key parameters

| Parameter | Default in class | Meaning |
|---|---:|---|
| `sigma_f` | `0.0` | Set to `10.0` in the preset to make this a pure fundamentalist |
| `bias` | `0.5` | Probability of using price/oracle mode instead of premium mode |
| `exit_prob` | `0.20` | Chance to flatten an existing position on a wake |
| `forecast_deadband_bps` | `10.0` | Ignore small forecast returns |
| `order_size` | `1.0` | Fixed order quantity |
| `wake_interval_s` | `60.0` | Decision interval |

### 5. Chiarella Chartist

#### What it is

This is the pure Chiarella trend-following trader created by setting only `sigma_c` nonzero.

#### Forecast component

Its forecast is entirely:

`f_c = chartist_forecast(signal_series)`

The chartist logic:

- requires enough history to cover at least `l_min + 2` observations
- samples a random lookback horizon uniformly from `[l_min, l_max]`
- computes average recent percentage changes over that horizon
- only includes terms whose previous signal value is positive

#### Economic interpretation

In positional mode it follows trends in the perpetual price itself.

In premium mode it follows trends in the basis/premium series rather than outright price. That makes it a trend follower on relative dislocation, not just on direction.

#### How it differs from `PerpMomentumAgent`

Both are momentum-like, but the mechanics are quite different:

- `PerpMomentumAgent` is a deterministic 20/50 moving-average crossover system.
- Chiarella Chartist uses a random lookback between `l_min` and `l_max`.
- `PerpMomentumAgent` always uses price mids.
- Chiarella Chartist may use outright price or premium depending on `bias`.
- `PerpMomentumAgent` suppresses repeat trades with `last_trade_signal`.
- Chiarella Chartist has no direct signal-memory flag like that; it re-evaluates and requotes each wake.

#### Key parameters

| Parameter | Default in class | Meaning |
|---|---:|---|
| `sigma_c` | `10.0` | Chartist weight scale; set as the active preset dimension |
| `l_min` | `1` | Minimum chartist lookback |
| `l_max` | `5` | Maximum chartist lookback |
| `bias` | `0.5` | Price-trend vs premium-trend mode |
| `forecast_deadband_bps` | `10.0` | Ignore weak trends |
| `order_size` | `1.0` | Fixed order quantity |
| `wake_interval_s` | `60.0` | Decision interval |

### 6. Chiarella Noise

#### What it is

This is the pure Chiarella random-shock trader created by setting only `sigma_n` nonzero.

#### Forecast component

Its forecast is entirely:

`f_n = sigma_e * U(-1, 1)`

That means:

- direction is random
- magnitude is random
- `sigma_e` controls how large the shock can be

#### Economic interpretation

This agent is the Chiarella framework's noisy trader. Unlike `PerpNoiseAgent`, its randomness enters as a forecast return rather than as a random side plus random passive quote offset.

Because its forecast is converted through `perp_px * exp(forecast_return)`, it still produces economically interpretable quote prices rather than arbitrary random orders.

#### How it differs from `PerpNoiseAgent`

These two are easy to confuse, but they are meaningfully different:

- `PerpNoiseAgent` randomizes trade/no-trade, side, size, and passive quote offset directly.
- Chiarella Noise randomizes the forecast return and then derives price from the Chiarella quoting framework.
- `PerpNoiseAgent` samples a new size every trade.
- Chiarella Noise uses a fixed `order_size`.
- Chiarella Noise inherits Chiarella-specific `exit_prob`, `bias`, premium/positional mode infrastructure, and GTC quoting behavior.

For the pure noise preset, `bias` does not affect the forecast itself because only the noise component is active, but the rest of the shared Chiarella machinery still runs.

#### Key parameters

| Parameter | Default in class | Meaning |
|---|---:|---|
| `sigma_n` | `10.0` | Noise weight scale; set as the active preset dimension |
| `sigma_e` | `0.05` | Amplitude of random forecast shocks |
| `k_max` | `0.15` | Additional favorable quote shading |
| `exit_prob` | `0.20` | Chance to flatten an existing position |
| `order_size` | `1.0` | Fixed order quantity |
| `wake_interval_s` | `60.0` | Decision interval |

## Quick comparison

| Agent | Main signal | Wake style | Entry style | Size model | Typical role |
|---|---|---|---|---|---|
| `PerpNoiseAgent` | None, purely random | Fixed interval | Passive limit | New random size each trade | Background uninformed flow |
| `PerpMomentumAgent` | 20/50 mid-price MA crossover | Fixed interval or subscription-driven | Passive limit | One size drawn at init | Outright trend follower |
| `PerpValueAgent` | Estimated terminal value vs mid | Exponential random interval | Mostly passive, occasional market cross | One size drawn at init | Informed mean reversion |
| Chiarella `Fundamentalist` | Oracle/perp or premium reversion | Fixed interval with subscription cache | GTC passive limit | Fixed `order_size` | Basis/value reversion inside Chiarella framework |
| Chiarella `Chartist` | Recent trend in price or premium | Fixed interval with subscription cache | GTC passive limit | Fixed `order_size` | Chiarella-style momentum/trend flow |
| Chiarella `Noise` | Random forecast shock | Fixed interval with subscription cache | GTC passive limit | Fixed `order_size` | Random flow with structured Chiarella quoting |

## Practical takeaway

If you want the shortest mental model:

- `PerpNoiseAgent` is random passive flow.
- `PerpMomentumAgent` is clean MA-crossover momentum.
- `PerpValueAgent` is standalone noisy-oracle value trading with occasional aggressive execution.
- The three Chiarella variants are the same execution shell with different forecast engines:
  - `Fundamentalist` for reversion to oracle or zero premium
  - `Chartist` for trend following
  - `Noise` for random shocks

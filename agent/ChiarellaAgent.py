"""Chiarella et al. (2002) composite trader adapted for HIP-3 perpetual futures.

Implements the three canonical strategy components from
"A Simulation Analysis of the Microstructure of Double Auction Markets"
as extended by Rao (2025) for perpetual futures:

  Fundamentalist  --  mean-reversion forecast toward oracle (spot) price
  Chartist        --  moving-average momentum over a random time horizon
  Noise           --  random return scaled by a volatility parameter

Each agent instance carries random weights (w_f, w_c, w_n) that blend
the three forecasts into a composite.  At each wakeup the agent:

  1. Is randomly assigned Long or Short if not already positioned.
  2. Decides positional vs basis trading using the `bias` parameter.
  3. Generates a composite price forecast.
  4. Submits a bid or ask offset by a random spread factor k in (0, k_max).
  5. May randomly exit (close position) with probability `exit_prob`.

To create pure-type agents, set only one weight > 0:
  Fundamentalist:  sigma_f=10, sigma_c=0, sigma_n=0
  Chartist:        sigma_f=0,  sigma_c=10, sigma_n=0
  Noise:           sigma_f=0,  sigma_c=0,  sigma_n=10
"""

from agent.PerpTradingAgent import PerpTradingAgent
from util.ContractSpec import TimeInForce
from util.util import log_print

import math
import pandas as pd


class ChiarellaAgent(PerpTradingAgent):

    def __init__(self, id, name, type, symbol,
                 sigma_f=0.0, sigma_c=10.0, sigma_n=10.0,
                 sigma_e=0.05, k_max=0.5,
                 l_min=1, l_max=5,
                 bias=0.5, exit_prob=0.05,
                 order_size=1.0,
                 wake_interval_s=60.0,
                 **kwargs):
        super().__init__(id, name, type, **kwargs)
        self.symbol = symbol
        self.sigma_e = sigma_e
        self.k_max = k_max
        self.l_min = l_min
        self.l_max = l_max
        self.bias = bias
        self.exit_prob = exit_prob
        self.order_size = order_size
        self.wake_interval_ns = int(wake_interval_s * 1e9)
        self.wake_interval_s = wake_interval_s

        # Draw random weights for this agent (fixed for its lifetime)
        # These are drawn in __init__ so they depend on the agent's own random_state
        self.w_f = sigma_f * self.random_state.uniform() if sigma_f > 0 else 0.0
        self.w_c = sigma_c * self.random_state.uniform() if sigma_c > 0 else 0.0
        self.w_n = sigma_n * self.random_state.uniform() if sigma_n > 0 else 0.0

        total = self.w_f + self.w_c + self.w_n
        if total <= 0:
            self.w_n = 1.0
            total = 1.0
        self.w_f /= total
        self.w_c /= total
        self.w_n /= total

        # Side: None = neutral, 'long', 'short'
        self.side = None

        # Price history buffers (populated from market data)
        self.oracle_history = []
        self.perp_history = []
        self.premium_history = []

        # State
        self.subscribed = False

    def kernelStarting(self, startTime):
        # Replicate PerpTradingAgent setup but skip the base Agent.setWakeup(startTime)
        # which would cause all agents to wake simultaneously and flood the exchange.
        self.logEvent('STARTING_CASH', self.starting_cash, True)
        from agent.PerpExchangeAgent import PerpExchangeAgent
        self.exchangeID = self.kernel.findAgentByType(PerpExchangeAgent)

        # Stagger initial wakeup across the first wake interval to avoid message storm.
        offset_ns = int(self.random_state.uniform() * self.wake_interval_ns)
        self.setWakeup(startTime + pd.Timedelta(offset_ns))

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return

        # On first ready wakeup, just request current mark price (no streaming subscription).
        if not self.subscribed:
            self.getMarkPrice(self.symbol)
            self.subscribed = True
            self.setWakeup(currentTime + pd.Timedelta(self.wake_interval_ns))
            return

        # Request fresh mark price for next wakeup
        self.getMarkPrice(self.symbol)

        # Collect current prices
        oracle_px = self.oracle_prices.get(self.symbol)
        perp_px = self.mark_prices.get(self.symbol)

        if perp_px is None:
            perp_px = self.last_trade.get(self.symbol)
        if oracle_px is None or perp_px is None or oracle_px <= 0 or perp_px <= 0:
            self.setWakeup(currentTime + pd.Timedelta(self.wake_interval_ns))
            return

        premium = perp_px - oracle_px

        # Append to history
        self.oracle_history.append(oracle_px)
        self.perp_history.append(perp_px)
        self.premium_history.append(premium)

        # --- Random exit: close position and become neutral ---
        if self.side is not None and self.random_state.uniform() < self.exit_prob:
            pos_size = self.getPositionSize(self.symbol)
            if pos_size != 0:
                self.placeMarketOrder(
                    self.symbol, abs(pos_size),
                    is_buy_order=(pos_size < 0),
                    reduce_only=True, tag="EXIT")
            self.side = None
            self.setWakeup(currentTime + pd.Timedelta(self.wake_interval_ns))
            return

        # --- Assign side if neutral ---
        if self.side is None:
            self.side = 'long' if self.random_state.uniform() < 0.5 else 'short'

        # --- Decide positional vs basis trading ---
        if self.side == 'long':
            is_positional = self.random_state.uniform() > self.bias
        else:
            is_positional = self.random_state.uniform() < self.bias

        # --- Select signal for forecasting ---
        if is_positional:
            signal = self.oracle_history
            current_price = oracle_px
        else:
            signal = self.premium_history
            current_price = premium
            # Shift premiums to positive domain for the chartist MA (per paper)
            if signal and min(signal) < 0:
                shift = abs(min(signal)) + 1.0
                signal = [s + shift for s in signal]
                current_price = premium + shift

        # --- Generate composite forecast return ---
        forecast_return = self._composite_forecast(signal, current_price, oracle_px)

        # --- Convert return to price forecast ---
        ref_price = oracle_px if is_positional else current_price
        if ref_price <= 0:
            self.setWakeup(currentTime + pd.Timedelta(self.wake_interval_ns))
            return

        price_forecast = ref_price * math.exp(forecast_return)

        # --- Map forecast back to perp price for basis traders ---
        if not is_positional:
            if signal and min(self.premium_history) < 0:
                shift = abs(min(self.premium_history)) + 1.0
                price_forecast = price_forecast - shift
            price_forecast = oracle_px + price_forecast

        # --- Determine order direction (Chiarella et al. 2002) ---
        # Positional: Long buys when forecast > price, Short SELLS (opposite side).
        # Basis: direction is inverted — basis traders trade against the premium.
        # This ensures longs and shorts always take opposite sides of any trade.
        forecast_above = price_forecast > perp_px
        long_side = (self.side == 'long')

        if is_positional:
            is_buy = forecast_above if long_side else not forecast_above
        else:
            is_buy = (not forecast_above) if long_side else forecast_above

        k = self.random_state.uniform(0, self.k_max) if self.k_max > 0 else 0
        if is_buy:
            order_price = price_forecast * (1.0 - k)
        else:
            order_price = price_forecast * (1.0 + k)

        order_price = round(max(order_price, 0.0001), 6)

        # --- Determine if this is a reduce-only order ---
        pos_size = self.getPositionSize(self.symbol)
        reduce_only = False
        if (pos_size > 0 and not is_buy) or (pos_size < 0 and is_buy):
            reduce_only = True

        # --- Cancel existing open orders before placing new one ---
        for oid, o in list(self.orders.items()):
            self.cancelOrder(o)

        # --- Place the order ---
        self.placeLimitOrder(
            self.symbol, self.order_size, is_buy, order_price,
            tag="POS" if not reduce_only else "CLOSE",
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only)

        self.setWakeup(currentTime + pd.Timedelta(self.wake_interval_ns))

    def _composite_forecast(self, signal, current_price, oracle_px):
        """Weighted composite of fundamentalist, chartist, and noise forecast returns."""
        f_f = self._fundamentalist_forecast(current_price, oracle_px)
        f_c = self._chartist_forecast(signal)
        f_n = self._noise_forecast()

        return self.w_f * f_f + self.w_c * f_c + self.w_n * f_n

    def _fundamentalist_forecast(self, current_price, oracle_px):
        """Mean-reversion return toward the oracle (fundamental) price.
        
        r_t = log(oracle / current_price)
        """
        if current_price <= 0 or oracle_px <= 0:
            return 0.0
        return math.log(oracle_px / current_price)

    def _chartist_forecast(self, signal):
        """Moving-average momentum return over a random horizon.
        
        r_t = (1/L) * sum_{j=1}^{L} (p_{t-j} - p_{t-j-1}) / p_{t-j-1}
        """
        if len(signal) < self.l_min + 2:
            return 0.0

        L = self.random_state.randint(self.l_min, min(self.l_max, len(signal) - 1) + 1)
        if L <= 0:
            return 0.0

        total_return = 0.0
        n = len(signal)
        for j in range(1, L + 1):
            idx = n - j
            idx_prev = n - j - 1
            if idx_prev < 0 or idx < 0:
                break
            if signal[idx_prev] > 0:
                total_return += (signal[idx] - signal[idx_prev]) / signal[idx_prev]

        return total_return / L

    def _noise_forecast(self):
        """Random return scaled by sigma_epsilon.
        
        r_t = sigma_e * U(0,1)
        """
        return self.sigma_e * self.random_state.uniform()

    def getWakeFrequency(self):
        return pd.Timedelta('1s')

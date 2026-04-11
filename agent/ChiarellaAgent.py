"""Chiarella et al. (2002) composite trader adapted for HIP-3 perpetual futures.

Implements the three canonical strategy components from
"A Simulation Analysis of the Microstructure of Double Auction Markets"
as extended by Rao (2025) for perpetual futures.
"""

from agent.PerpTradingAgent import PerpTradingAgent
from util.ContractSpec import TimeInForce

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

        self.side = None
        self.oracle_history = []
        self.perp_history = []
        self.premium_history = []
        self.subscription_requested = False
        self.next_strategy_wake = None

    def _schedule_strategy_wake(self, when):
        self.next_strategy_wake = when
        self.setWakeup(when)

    def kernelStarting(self, startTime):
        self.logEvent('STARTING_CASH', self.starting_cash, True)
        from agent.PerpExchangeAgent import PerpExchangeAgent
        self.exchangeID = self.kernel.findAgentByType(PerpExchangeAgent)

        offset_ns = int(self.random_state.uniform() * self.wake_interval_ns)
        self._schedule_strategy_wake(startTime + pd.Timedelta(offset_ns))

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if self.next_strategy_wake is not None and currentTime < self.next_strategy_wake:
            return
        if self.next_strategy_wake is not None and currentTime >= self.next_strategy_wake:
            self.next_strategy_wake = None
        if not ready or self.mkt_closed:
            return
        if not self.subscription_requested:
            self.requestDataSubscription(self.symbol, levels=1, freq=self.wake_interval_ns)
            self.subscription_requested = True
        snapshot = self._snapshot_from_cache()
        if snapshot is None:
            self._schedule_strategy_wake(currentTime + pd.Timedelta("1s"))
            return
        self._trade_from_snapshot(currentTime, snapshot)
        self._schedule_strategy_wake(currentTime + pd.Timedelta(self.wake_interval_ns))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

    def _snapshot_from_cache(self):
        if self.symbol not in self.oracle_prices and self.symbol not in self.mark_prices:
            return None
        return {
            'mark_price': self.mark_prices.get(self.symbol),
            'oracle_price': self.oracle_prices.get(self.symbol),
            'bids': self.known_bids.get(self.symbol, []),
            'asks': self.known_asks.get(self.symbol, []),
            'last_trade': self.last_trade.get(self.symbol),
        }

    def _trade_from_snapshot(self, currentTime, snapshot):
        oracle_px = snapshot.get('oracle_price')
        bids = snapshot.get('bids') or []
        asks = snapshot.get('asks') or []
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        if best_bid is not None and best_ask is not None:
            perp_px = (best_bid + best_ask) / 2.0
        else:
            perp_px = snapshot.get('mark_price') or snapshot.get('last_trade')

        if oracle_px is None or perp_px is None or oracle_px <= 0 or perp_px <= 0:
            return

        premium = perp_px - oracle_px
        self.oracle_history.append(oracle_px)
        self.perp_history.append(perp_px)
        self.premium_history.append(premium)

        if self.side is not None and self.random_state.uniform() < self.exit_prob:
            pos_size = self.getPositionSize(self.symbol)
            if pos_size != 0:
                self.placeMarketOrder(
                    self.symbol, abs(pos_size),
                    is_buy_order=(pos_size < 0),
                    reduce_only=True, tag="EXIT")
            self.side = None
            return

        if self.side is None:
            self.side = 'long' if self.random_state.uniform() < 0.5 else 'short'

        if self.side == 'long':
            is_positional = self.random_state.uniform() > self.bias
        else:
            is_positional = self.random_state.uniform() < self.bias

        if is_positional:
            signal = self.oracle_history
            current_price = oracle_px
        else:
            signal = self.premium_history
            current_price = premium
            if signal and min(signal) < 0:
                shift = abs(min(signal)) + 1.0
                signal = [sample + shift for sample in signal]
                current_price = premium + shift

        forecast_return = self._composite_forecast(signal, current_price, oracle_px)
        ref_price = oracle_px if is_positional else current_price
        if ref_price <= 0:
            return

        forecast_return = max(-10.0, min(10.0, forecast_return))
        price_forecast = ref_price * math.exp(forecast_return)

        if not is_positional:
            if signal and min(self.premium_history) < 0:
                shift = abs(min(self.premium_history)) + 1.0
                price_forecast = price_forecast - shift
            price_forecast = oracle_px + price_forecast

        forecast_above = price_forecast > perp_px
        long_side = (self.side == 'long')
        if is_positional:
            is_buy = forecast_above if long_side else not forecast_above
        else:
            is_buy = (not forecast_above) if long_side else forecast_above

        k = self.random_state.uniform(0, self.k_max) if self.k_max > 0 else 0.0
        order_price = price_forecast * (1.0 - k) if is_buy else price_forecast * (1.0 + k)
        order_price = max(order_price, 0.0001)

        pos_size = self.getPositionSize(self.symbol)
        reduce_only = (pos_size > 0 and not is_buy) or (pos_size < 0 and is_buy)

        for order in list(self.orders.values()):
            self.cancelOrder(order)

        self.placeLimitOrder(
            self.symbol, self.order_size, is_buy, order_price,
            tag="POS" if not reduce_only else "CLOSE",
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only,
        )

    def _composite_forecast(self, signal, current_price, oracle_px):
        f_f = self._fundamentalist_forecast(current_price, oracle_px)
        f_c = self._chartist_forecast(signal)
        f_n = self._noise_forecast()
        return self.w_f * f_f + self.w_c * f_c + self.w_n * f_n

    def _fundamentalist_forecast(self, current_price, oracle_px):
        if current_price <= 0 or oracle_px <= 0:
            return 0.0
        return math.log(oracle_px / current_price)

    def _chartist_forecast(self, signal):
        if len(signal) < self.l_min + 2:
            return 0.0

        horizon = self.random_state.randint(self.l_min, min(self.l_max, len(signal) - 1) + 1)
        if horizon <= 0:
            return 0.0

        total_return = 0.0
        n = len(signal)
        for offset in range(1, horizon + 1):
            idx = n - offset
            idx_prev = n - offset - 1
            if idx_prev < 0 or idx < 0:
                break
            if signal[idx_prev] > 0:
                total_return += (signal[idx] - signal[idx_prev]) / signal[idx_prev]
        return total_return / horizon

    def _noise_forecast(self):
        return self.sigma_e * self.random_state.uniform()

    def getWakeFrequency(self):
        return pd.Timedelta('1s')

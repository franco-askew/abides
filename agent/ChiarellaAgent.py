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
                 sigma_e=0.05, k_max=0.15,
                 l_min=1, l_max=5,
                 bias=0.5, exit_prob=0.20,
                 forecast_deadband_bps=10.0,
                 order_size=1.0,
                 wake_interval_s=60.0,
                 **kwargs):
        kwargs.setdefault("max_live_orders_per_symbol", 1)
        kwargs.setdefault("opening_order_cooldown_after_unfunded_s", 600.0)
        kwargs.setdefault("max_take_distance_bps_from_mark", 50.0)
        kwargs.setdefault("max_passive_distance_bps_from_mark", 100.0)
        super().__init__(id, name, type, **kwargs)
        self.symbol = symbol
        self.sigma_e = sigma_e
        self.k_max = k_max
        self.l_min = l_min
        self.l_max = l_max
        self.bias = bias
        self.exit_prob = exit_prob
        self.forecast_deadband_bps = forecast_deadband_bps
        self.order_size = order_size
        self.wake_interval_ns = int(wake_interval_s * 1e9)
        self.wake_interval_s = wake_interval_s
        if self.max_position_size is None:
            self.max_position_size = 10.0 * self.order_size

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
        mark_px = snapshot.get('mark_price')
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

        pos_size = self.getPositionSize(self.symbol)
        if pos_size != 0 and self.random_state.uniform() < self.exit_prob:
            if self._strategy_touch_within_take_band(self.symbol, pos_size < 0):
                self.placeMarketOrder(
                    self.symbol, abs(pos_size),
                    is_buy_order=(pos_size < 0),
                    reduce_only=True, tag="EXIT")
            return

        is_positional = self.random_state.uniform() < self.bias

        if is_positional:
            signal = self.perp_history
            current_value = perp_px
            anchor_value = oracle_px
        else:
            signal = self.premium_history
            current_value = premium
            anchor_value = 0.0
            if signal and min(signal) < 0:
                shift = abs(min(signal)) + 1.0
                signal = [sample + shift for sample in signal]
                current_value = premium + shift

        forecast_return = self._composite_forecast(signal, current_value, anchor_value)
        deadband = self.forecast_deadband_bps / 10000.0
        if abs(forecast_return) < deadband:
            return

        forecast_return = max(-0.05, min(0.05, forecast_return))
        is_buy = forecast_return > 0
        if not self._strategy_allows_order(self.symbol, self.order_size, is_buy):
            return

        reference_price = mark_px if mark_px is not None and mark_px > 0 else perp_px
        candidate_price = perp_px * math.exp(forecast_return)
        candidate_price = self._strategy_clip_price_to_band(
            self.symbol,
            candidate_price,
            max_bps=self.max_passive_distance_bps_from_mark,
            reference_price=reference_price,
        )
        if candidate_price is None:
            self._record_local_skip(self.symbol, 'NO_REFERENCE_PRICE')
            return

        slack_frac = self.random_state.uniform(0, self.k_max / 100.0) if self.k_max > 0 else 0.0
        order_price = candidate_price * (1.0 - slack_frac) if is_buy else candidate_price * (1.0 + slack_frac)
        order_price = self._strategy_clip_price_to_band(
            self.symbol,
            order_price,
            max_bps=self.max_passive_distance_bps_from_mark,
            reference_price=reference_price,
        )
        if order_price is None or order_price <= 0:
            self._record_local_skip(self.symbol, 'INVALID_PRICE')
            return

        taking = self._strategy_is_taking(self.symbol, is_buy, order_price)
        if taking and not self._strategy_touch_within_take_band(self.symbol, is_buy):
            return

        self._cancel_symbol_orders(self.symbol)
        if not self._strategy_has_open_order_capacity(self.symbol):
            return

        self.placeLimitOrder(
            self.symbol, self.order_size, is_buy, order_price,
            tag="POS",
            time_in_force=TimeInForce.GTC,
            reduce_only=False,
        )

    def _composite_forecast(self, signal, current_value, anchor_value):
        f_f = self._fundamentalist_forecast(current_value, anchor_value)
        f_c = self._chartist_forecast(signal)
        f_n = self._noise_forecast()
        return self.w_f * f_f + self.w_c * f_c + self.w_n * f_n

    def _fundamentalist_forecast(self, current_value, anchor_value):
        if current_value is None:
            return 0.0
        if current_value > 0 and anchor_value > 0:
            return math.log(anchor_value / current_value)
        scale = max(abs(current_value), abs(anchor_value), 1.0)
        return (anchor_value - current_value) / scale

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
        return self.sigma_e * self.random_state.uniform(-1.0, 1.0)

    def getWakeFrequency(self):
        return pd.Timedelta('1s')

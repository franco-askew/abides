"""Perpetual futures value/fundamental agent.

Observes oracle price with noise, applies Bayesian mean-reversion update
to estimate terminal value, then places a limit order based on that estimate
vs the current mid price. Subclasses PerpTradingAgent.
"""

from agent.PerpTradingAgent import PerpTradingAgent
from util.util import log_print

import numpy as np
import pandas as pd
import warnings


class PerpValueAgent(PerpTradingAgent):

    def __init__(self, id, name, type, symbol='ASSET-USD', starting_cash=100000.0,
                 sigma_n=1.0, r_bar=100.0, kappa=0.05, sigma_s=1.0,
                 lambda_a=None, percent_aggr=None, depth_spread=2,
                 mean_wake_interval_s=300.0, mispricing_deadband_bps=15.0,
                 aggressive_cross_prob=0.02,
                 min_size=0.1, max_size=1.0,
                 log_orders=False, log_to_file=True, random_state=None, **kwargs):
        kwargs.setdefault("max_live_orders_per_symbol", 1)
        kwargs.setdefault("opening_order_cooldown_after_unfunded_s", 600.0)
        kwargs.setdefault("max_take_distance_bps_from_mark", 50.0)
        kwargs.setdefault("max_passive_distance_bps_from_mark", 100.0)

        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, log_to_file=log_to_file,
                         random_state=random_state, **kwargs)

        self.symbol = symbol
        self.sigma_n = sigma_n
        self.r_bar = r_bar
        self.kappa = kappa
        self.sigma_s = sigma_s
        if lambda_a is not None:
            warnings.warn(
                "lambda_a is deprecated; use mean_wake_interval_s instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if mean_wake_interval_s is None:
                mean_wake_interval_s = max(1.0, 1.0 / float(lambda_a))
        self.lambda_a = lambda_a
        self.mean_wake_interval_s = float(mean_wake_interval_s if mean_wake_interval_s is not None else 300.0)
        self.mispricing_deadband_bps = float(mispricing_deadband_bps)
        self.aggressive_cross_prob = (
            float(percent_aggr) if percent_aggr is not None else float(aggressive_cross_prob)
        )
        self.depth_spread = depth_spread
        self.min_size = min_size
        self.max_size = max_size

        self.trading = False
        self.state = 'AWAITING_WAKEUP'
        self.r_t = r_bar
        self.sigma_t = 0
        self.prev_wake_time = None
        self.size = self._round_quantity(self.symbol, self.random_state.uniform(self.min_size, self.max_size))
        if self.max_position_size is None:
            self.max_position_size = 5.0 * self.size

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.agent_oracles.get(self.id, self.kernel.oracle)

    def kernelStopping(self):
        super().kernelStopping()
        rT = self.oracle.observePrice(self.symbol, self.currentTime,
                                       sigma_n=0, random_state=self.random_state)
        pos_size = self.getPositionSize(self.symbol)
        equity = self.getBalance() + pos_size * rT
        surplus = (equity - self.starting_cash) / self.starting_cash if self.starting_cash > 0 else 0
        self.logEvent('FINAL_VALUATION', surplus, True)
        log_print("{} final: balance={:.2f}, pos={:.4f}, equity={:.2f}, surplus={:.4f}",
                  self.name, self.getBalance(), pos_size, equity, surplus)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            return

        if not self.trading:
            self.trading = True
            log_print("{} is ready to start trading now.", self.name)

        if self.mkt_closed:
            return

        self.setWakeup(currentTime + self._sample_next_wake_offset())
        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            if self.mkt_closed:
                return
            self.placeOrder()
            self.state = 'AWAITING_WAKEUP'

    def updateEstimates(self):
        obs_t = self.oracle.observePrice(self.symbol, self.currentTime,
                                          sigma_n=self.sigma_n,
                                          random_state=self.random_state)
        log_print("{} observed {:.4f} at {}", self.name, obs_t, self.currentTime)

        if self.prev_wake_time is None:
            self.prev_wake_time = self.mkt_open

        delta = max(0.0, (self.currentTime - self.prev_wake_time) / np.timedelta64(1, 's'))

        r_tprime = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_tprime += ((1 - self.kappa) ** delta) * self.r_t

        sigma_tprime = ((1 - self.kappa) ** (2 * delta)) * self.sigma_t
        sigma_tprime += ((1 - (1 - self.kappa) ** (2 * delta)) / (1 - (1 - self.kappa) ** 2)) * self.sigma_s

        self.r_t = (self.sigma_n / (self.sigma_n + sigma_tprime)) * r_tprime
        self.r_t += (sigma_tprime / (self.sigma_n + sigma_tprime)) * obs_t

        self.sigma_t = (self.sigma_n * self.sigma_t) / (self.sigma_n + self.sigma_t) if (self.sigma_n + self.sigma_t) > 0 else 0

        delta = max(0.0, (self.mkt_close - self.currentTime) / np.timedelta64(1, 's'))

        r_T = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_T += ((1 - self.kappa) ** delta) * self.r_t

        self.prev_wake_time = self.currentTime

        log_print("{} estimates r_T = {:.4f} as of {}", self.name, r_T, self.currentTime)
        return r_T

    def placeOrder(self):
        r_T = self.updateEstimates()
        bb, ba = self.getKnownBidAsk(self.symbol)

        if bb and ba:
            mid = (ba + bb) / 2.0
            mispricing_bps = abs(r_T - mid) / mid * 10000.0 if mid > 0 else 0.0
            if mispricing_bps < self.mispricing_deadband_bps:
                return

            is_buy = r_T > mid
            if not self._strategy_allows_order(self.symbol, self.size, is_buy):
                return

            self._cancel_symbol_orders(self.symbol)
            if not self._strategy_has_open_order_capacity(self.symbol):
                return

            if self.random_state.rand() < self.aggressive_cross_prob and self._strategy_touch_within_take_band(self.symbol, is_buy):
                self.placeMarketOrder(self.symbol, self.size, is_buy)
                return

            p = self._strategy_passive_price_from_mid(self.symbol, is_buy, min_bps=5.0, max_bps=25.0)
        else:
            self._record_local_skip(self.symbol, 'NO_TOUCH')
            return

        if p is not None and p > 0:
            self.placeLimitOrder(self.symbol, self.size, is_buy, p)

    def cancelOrders(self):
        if not self.orders:
            return False
        for oid, order in list(self.orders.items()):
            self.cancelOrder(order)
        return True

    def getWakeFrequency(self):
        return pd.Timedelta(seconds=self.mean_wake_interval_s)

    def _sample_next_wake_offset(self):
        seconds = max(1.0, float(self.random_state.exponential(scale=self.mean_wake_interval_s)))
        return pd.Timedelta(seconds=seconds)

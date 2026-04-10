"""Perpetual futures value/fundamental agent.

Observes oracle price with noise, applies Bayesian mean-reversion update
to estimate terminal value, then places a limit order based on that estimate
vs the current mid price. Subclasses PerpTradingAgent.
"""

from agent.PerpTradingAgent import PerpTradingAgent
from util.util import log_print

import numpy as np
import pandas as pd


class PerpValueAgent(PerpTradingAgent):

    def __init__(self, id, name, type, symbol='ASSET-USD', starting_cash=100000.0,
                 sigma_n=1.0, r_bar=100.0, kappa=0.05, sigma_s=1.0,
                 lambda_a=0.005, percent_aggr=0.1, depth_spread=2,
                 min_size=0.1, max_size=1.0,
                 log_orders=False, log_to_file=True, random_state=None, **kwargs):

        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, log_to_file=log_to_file,
                         random_state=random_state, **kwargs)

        self.symbol = symbol
        self.sigma_n = sigma_n
        self.r_bar = r_bar
        self.kappa = kappa
        self.sigma_s = sigma_s
        self.lambda_a = lambda_a
        self.percent_aggr = percent_aggr
        self.depth_spread = depth_spread
        self.min_size = min_size
        self.max_size = max_size

        self.trading = False
        self.state = 'AWAITING_WAKEUP'
        self.r_t = r_bar
        self.sigma_t = 0
        self.prev_wake_time = None
        self.size = self._round_quantity(self.symbol, self.random_state.uniform(self.min_size, self.max_size))

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle

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

        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(int(round(delta_time)))))

        self.cancelOrders()
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

        delta = (self.currentTime - self.prev_wake_time) / np.timedelta64(1, 'ns')

        r_tprime = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_tprime += ((1 - self.kappa) ** delta) * self.r_t

        sigma_tprime = ((1 - self.kappa) ** (2 * delta)) * self.sigma_t
        sigma_tprime += ((1 - (1 - self.kappa) ** (2 * delta)) / (1 - (1 - self.kappa) ** 2)) * self.sigma_s

        self.r_t = (self.sigma_n / (self.sigma_n + sigma_tprime)) * r_tprime
        self.r_t += (sigma_tprime / (self.sigma_n + sigma_tprime)) * obs_t

        self.sigma_t = (self.sigma_n * self.sigma_t) / (self.sigma_n + self.sigma_t) if (self.sigma_n + self.sigma_t) > 0 else 0

        delta = max(0, (self.mkt_close - self.currentTime) / np.timedelta64(1, 'ns'))

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
            spread = abs(ba - bb)

            if self.random_state.rand() < self.percent_aggr:
                adjust = 0.0
            else:
                adjust = self.random_state.uniform(0, self.depth_spread * spread)

            if r_T < mid:
                is_buy = False
                p = bb + adjust
            else:
                is_buy = True
                p = ba - adjust
        else:
            is_buy = bool(self.random_state.randint(0, 2))
            p = r_T

        p = self._round_price(self.symbol, p)
        if p is not None and p > 0:
            self.placeLimitOrder(self.symbol, self.size, is_buy, p)

    def cancelOrders(self):
        if not self.orders:
            return False
        for oid, order in list(self.orders.items()):
            self.cancelOrder(order)
        return True

    def getWakeFrequency(self):
        delta = self.random_state.exponential(scale=1.0 / self.lambda_a)
        return pd.Timedelta('{}ns'.format(int(round(delta))))

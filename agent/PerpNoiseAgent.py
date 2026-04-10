"""Perpetual futures noise agent.

Places random buy/sell limit orders at the best bid/ask, with configurable
size drawn from a uniform distribution. Subclasses PerpTradingAgent.
"""

from agent.PerpTradingAgent import PerpTradingAgent
from util.util import log_print

import numpy as np
import pandas as pd


class PerpNoiseAgent(PerpTradingAgent):

    def __init__(self, id, name, type, symbol='ASSET-USD', starting_cash=100000.0,
                 min_size=0.1, max_size=1.0, wake_up_freq='5s', wakeup_time=None,
                 log_orders=False, log_to_file=True, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, log_to_file=log_to_file,
                         random_state=random_state)

        self.symbol = symbol
        self.min_size = min_size
        self.max_size = max_size
        self.wake_up_freq = wake_up_freq
        self.wakeup_time = wakeup_time
        self.trading = False
        self.state = 'AWAITING_WAKEUP'

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        super().kernelStopping()
        bb, ba = self.getKnownBidAsk(self.symbol)
        if bb and ba:
            mid = (bb + ba) / 2.0
        else:
            mid = self.last_trade.get(self.symbol, 0)
        pos_size = self.getPositionSize(self.symbol)
        equity = self.getBalance() + pos_size * mid
        surplus = (equity - self.starting_cash) / self.starting_cash if self.starting_cash > 0 else 0
        self.logEvent('FINAL_VALUATION', surplus, True)
        log_print("{} final: balance={:.2f}, pos={:.4f}, equity={:.2f}, surplus={:.4f}",
                  self.name, self.getBalance(), pos_size, equity, surplus)

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # Market hours not yet known; schedule retry after a short delay
            self.setWakeup(currentTime + pd.Timedelta(self.wake_up_freq))
            return

        if not self.trading:
            self.trading = True
            log_print("{} is ready to start trading now.", self.name)

        if self.mkt_closed:
            return

        if self.wakeup_time is not None and self.wakeup_time > currentTime:
            self.setWakeup(self.wakeup_time)
            return

        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            if self.mkt_closed:
                return
            self.placeOrder()
            self.setWakeup(currentTime + pd.Timedelta(self.wake_up_freq))
            self.state = 'AWAITING_WAKEUP'

    def placeOrder(self):
        is_buy = bool(self.random_state.randint(0, 2))
        bb, ba = self.getKnownBidAsk(self.symbol)
        size = round(self.random_state.uniform(self.min_size, self.max_size), 4)

        if is_buy and ba:
            self.placeLimitOrder(self.symbol, size, True, ba)
        elif not is_buy and bb:
            self.placeLimitOrder(self.symbol, size, False, bb)

    def cancelOrders(self):
        if not self.orders:
            return False
        for oid, order in list(self.orders.items()):
            self.cancelOrder(order)
        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

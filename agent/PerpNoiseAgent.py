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
                 min_size=0.1, max_size=1.0, wake_up_freq='30s', wakeup_time=None,
                 trade_probability=0.20,
                 log_orders=False, log_to_file=True, random_state=None, **kwargs):
        kwargs.setdefault("max_live_orders_per_symbol", 1)
        kwargs.setdefault("opening_order_cooldown_after_unfunded_s", 600.0)
        kwargs.setdefault("max_take_distance_bps_from_mark", 50.0)
        kwargs.setdefault("max_passive_distance_bps_from_mark", 100.0)

        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, log_to_file=log_to_file,
                         random_state=random_state, **kwargs)

        self.symbol = symbol
        self.min_size = min_size
        self.max_size = max_size
        self.wake_up_freq = wake_up_freq
        self.wakeup_time = wakeup_time
        self.trade_probability = trade_probability
        self.trading = False
        self.state = 'AWAITING_WAKEUP'
        self.subscription_requested = False
        self.market_data_freq_ns = pd.Timedelta(self.wake_up_freq).value
        self.next_strategy_wake = None

    def _schedule_strategy_wake(self, when):
        self.next_strategy_wake = when
        self.setWakeup(when)

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.agent_oracles.get(self.id, self.kernel.oracle)

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

        if self.next_strategy_wake is not None and currentTime < self.next_strategy_wake:
            return
        if self.next_strategy_wake is not None and currentTime >= self.next_strategy_wake:
            self.next_strategy_wake = None

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # Market hours not yet known; schedule retry after a short delay
            self._schedule_strategy_wake(currentTime + pd.Timedelta(self.wake_up_freq))
            return

        if not self.trading:
            self.trading = True
            log_print("{} is ready to start trading now.", self.name)

        if self.mkt_closed:
            return

        if self.wakeup_time is not None and self.wakeup_time > currentTime:
            self._schedule_strategy_wake(self.wakeup_time)
            return

        if not self.subscription_requested:
            self.requestDataSubscription(self.symbol, levels=1, freq=self.market_data_freq_ns)
            self.subscription_requested = True

        self.placeOrder()
        self._schedule_strategy_wake(currentTime + pd.Timedelta(self.wake_up_freq))
        self.state = 'AWAITING_WAKEUP'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

    def placeOrder(self):
        if self.random_state.uniform() > self.trade_probability:
            return

        is_buy = bool(self.random_state.randint(0, 2))
        bb, ba = self.getKnownBidAsk(self.symbol)
        if bb is None or ba is None:
            self._record_local_skip(self.symbol, 'NO_TOUCH')
            return

        size = self._round_quantity(self.symbol, self.random_state.uniform(self.min_size, self.max_size))
        effective_position_cap = self.max_position_size if self.max_position_size is not None else 5.0 * size
        if not self._strategy_allows_order(
            self.symbol,
            size,
            is_buy,
            max_position_size=effective_position_cap,
        ):
            return

        self._cancel_symbol_orders(self.symbol)
        if not self._strategy_has_open_order_capacity(self.symbol):
            return
        price = self._strategy_passive_price_from_mid(self.symbol, is_buy, min_bps=5.0, max_bps=25.0)
        if price is None:
            return

        self.placeLimitOrder(self.symbol, size, is_buy, price)

    def cancelOrders(self):
        if not self.orders:
            return False
        for oid, order in list(self.orders.items()):
            self.cancelOrder(order)
        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

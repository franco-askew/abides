"""Perpetual futures momentum agent.

Compares a 20-period moving average of mid prices against a 50-period MA.
Buys at ask when 20-MA >= 50-MA, sells at bid otherwise.
Subclasses PerpTradingAgent.
"""

from agent.PerpTradingAgent import PerpTradingAgent

import numpy as np
import pandas as pd


class PerpMomentumAgent(PerpTradingAgent):

    def __init__(self, id, name, type, symbol='ASSET-USD', starting_cash=100000.0,
                 min_size=0.1, max_size=1.0, wake_up_freq='300s',
                 subscribe=False, log_orders=False, random_state=None,
                 trade_on_signal_change_only=True, crossover_deadband_bps=10.0,
                 **kwargs):
        kwargs.setdefault("max_live_orders_per_symbol", 1)
        kwargs.setdefault("opening_order_cooldown_after_unfunded_s", 600.0)
        kwargs.setdefault("max_take_distance_bps_from_mark", 50.0)
        kwargs.setdefault("max_passive_distance_bps_from_mark", 100.0)

        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, random_state=random_state, **kwargs)

        self.symbol = symbol
        self.min_size = min_size
        self.max_size = max_size
        self.size = self._round_quantity(self.symbol, self.random_state.uniform(self.min_size, self.max_size))
        self.wake_up_freq = wake_up_freq
        self.subscribe = subscribe
        self.trade_on_signal_change_only = trade_on_signal_change_only
        self.crossover_deadband_bps = crossover_deadband_bps
        self.subscription_requested = False
        self.mid_list = []
        self.avg_20_list = []
        self.avg_50_list = []
        if self.max_position_size is None:
            self.max_position_size = 5.0 * self.size
        self.last_trade_signal = 0
        self.state = 'AWAITING_WAKEUP'

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if self.subscribe and not self.subscription_requested:
            self.requestDataSubscription(self.symbol, levels=1, freq=int(10e9))
            self.subscription_requested = True
            self.state = 'AWAITING_MARKET_DATA'
        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if not self.subscribe and self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bb, ba = self.getKnownBidAsk(self.symbol)
            self.placeOrders(bb, ba)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'
        elif self.subscribe and self.state == 'AWAITING_MARKET_DATA' and msg.body['msg'] == 'MARKET_DATA':
            bids = self.known_bids.get(self.symbol, [])
            asks = self.known_asks.get(self.symbol, [])
            if bids and asks:
                self.placeOrders(bids[0][0], asks[0][0])
            self.state = 'AWAITING_MARKET_DATA'

    def placeOrders(self, bid, ask):
        if bid and ask:
            mid = (bid + ask) / 2.0
            self.mid_list.append(mid)
            if len(self.mid_list) > 20:
                self.avg_20_list.append(self._ma(self.mid_list, n=20))
            if len(self.mid_list) > 50:
                self.avg_50_list.append(self._ma(self.mid_list, n=50))
            if self.avg_20_list and self.avg_50_list:
                spread_signal_bps = ((self.avg_20_list[-1] - self.avg_50_list[-1]) / mid) * 10000.0 if mid > 0 else 0.0
                if abs(spread_signal_bps) < self.crossover_deadband_bps:
                    return

                signal = 1 if spread_signal_bps > 0 else -1
                if self.trade_on_signal_change_only and signal == self.last_trade_signal:
                    return

                is_buy = signal > 0
                if not self._strategy_allows_order(self.symbol, self.size, is_buy):
                    return

                self._cancel_symbol_orders(self.symbol)
                if not self._strategy_has_open_order_capacity(self.symbol):
                    return

                price = self._strategy_passive_price_from_mid(self.symbol, is_buy, min_bps=5.0, max_bps=25.0)
                if price is None:
                    return

                self.placeLimitOrder(self.symbol, quantity=self.size,
                                     is_buy_order=is_buy, limit_price=price)
                self.last_trade_signal = signal

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    @staticmethod
    def _ma(values, n=20):
        if len(values) < n:
            return sum(values) / len(values)
        return sum(values[-n:]) / n

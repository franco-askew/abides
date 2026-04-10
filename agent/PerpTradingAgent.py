"""Base trading agent for HIP-3 perpetual futures.

Provides margin-aware order placement, position tracking, and message handling.
Users subclass this and implement their own trading strategy by overriding
wakeup() and getWakeFrequency().

This agent does NOT define any trading logic -- it only provides infrastructure.
"""

from agent.FinancialAgent import FinancialAgent
from agent.PerpExchangeAgent import PerpExchangeAgent
from message.Message import Message
from util.order.PerpLimitOrder import PerpLimitOrder
from util.ContractSpec import TimeInForce
from util.PerpAccount import PerpAccount
from util.util import log_print

from copy import deepcopy
import pandas as pd


class PerpTradingAgent(FinancialAgent):

    def __init__(self, id, name, type, random_state=None, starting_cash=100000.0,
                 log_orders=False, log_to_file=True, default_leverage=10):
        super().__init__(id, name, type, random_state, log_to_file)

        self.starting_cash = starting_cash
        self.default_leverage = default_leverage
        self.log_orders = log_orders

        if log_orders is None:
            self.log_orders = False
            self.log_to_file = False

        # Exchange discovery
        self.exchangeID = None
        self.mkt_open = None
        self.mkt_close = None

        # Local account mirror (updated on fills and funding)
        self.account = PerpAccount(starting_cash)

        # Open orders tracking
        self.orders = {}

        # Cached market data
        self.last_trade = {}
        self.mark_prices = {}
        self.oracle_prices = {}
        self.known_bids = {}
        self.known_asks = {}
        self.exchange_ts = {}
        self.stream_history = {}
        self.funding_rates = {}
        self.transacted_volume = {}

        self.first_wake = True
        self.mkt_closed = False

    # ── Lifecycle ───────────────────────────────────────────────────────

    def kernelStarting(self, startTime):
        self.logEvent('STARTING_CASH', self.starting_cash, True)
        self.exchangeID = self.kernel.findAgentByType(PerpExchangeAgent)
        log_print("Agent {} found PerpExchangeAgent ID: {}", self.id, self.exchangeID)
        super().kernelStarting(startTime)

    def kernelStopping(self):
        super().kernelStopping()
        self.logEvent('FINAL_BALANCE', self.account.balance, True)
        mark_px = {s: self.mark_prices.get(s, self.last_trade.get(s, p.entry_price))
                   for s, p in self.account.positions.items()}
        equity = self.account.total_equity(mark_px)
        self.logEvent('ENDING_EQUITY', equity, True)

        positions_str = ", ".join(
            "{}:{:.4f}@{:.2f}".format(s, p.size, p.entry_price)
            for s, p in self.account.positions.items() if p.size != 0)
        print("Final state for {}: balance={:.2f}, positions=[{}], equity={:.2f}".format(
            self.name, self.account.balance, positions_str, equity))

        mytype = self.type
        gain = equity - self.starting_cash
        if mytype in self.kernel.meanResultByAgentType:
            self.kernel.meanResultByAgentType[mytype] += gain
            self.kernel.agentCountByType[mytype] += 1
        else:
            self.kernel.meanResultByAgentType[mytype] = gain
            self.kernel.agentCountByType[mytype] = 1

    # ── Wakeup ──────────────────────────────────────────────────────────

    def wakeup(self, currentTime):
        super().wakeup(currentTime)

        if self.first_wake:
            self.logEvent('ACCOUNT_INIT', {'balance': self.account.balance})
            self.first_wake = False

        if self.mkt_open is None:
            self.sendMessage(self.exchangeID, Message({"msg": "WHEN_MKT_OPEN", "sender": self.id}))
            self.sendMessage(self.exchangeID, Message({"msg": "WHEN_MKT_CLOSE", "sender": self.id}))

        return (self.mkt_open is not None and self.mkt_close is not None) and not self.mkt_closed

    # ── Message handling ────────────────────────────────────────────────

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        had_mkt_hours = self.mkt_open is not None and self.mkt_close is not None
        msg_type = msg.body['msg']

        if msg_type == "WHEN_MKT_OPEN":
            self.mkt_open = msg.body['data']
        elif msg_type == "WHEN_MKT_CLOSE":
            self.mkt_close = msg.body['data']
        elif msg_type == "ORDER_EXECUTED":
            self._on_order_executed(msg.body['order'], msg.body.get('fee', 0.0))
        elif msg_type == "ORDER_ACCEPTED":
            self._on_order_accepted(msg.body['order'])
        elif msg_type == "ORDER_CANCELLED":
            self._on_order_cancelled(msg.body['order'])
        elif msg_type == "ORDER_REJECTED":
            self._on_order_rejected(msg.body['order'], msg.body.get('reason', ''))
        elif msg_type == "ORDER_MODIFIED":
            pass
        elif msg_type == "MKT_CLOSED":
            self.mkt_closed = True
        elif msg_type == "FUNDING_PAYMENT":
            self._on_funding_payment(msg.body)
        elif msg_type == "LIQUIDATED":
            self._on_liquidated(msg.body)
        elif msg_type == "POSITION_SETTLED":
            self._on_position_settled(msg.body)
        elif msg_type == "QUERY_LAST_TRADE":
            if msg.body.get('mkt_closed'):
                self.mkt_closed = True
            self.last_trade[msg.body['symbol']] = msg.body['data']
        elif msg_type == "QUERY_SPREAD":
            if msg.body.get('mkt_closed'):
                self.mkt_closed = True
            sym = msg.body['symbol']
            self.last_trade[sym] = msg.body['data']
            self.known_bids[sym] = msg.body['bids']
            self.known_asks[sym] = msg.body['asks']
        elif msg_type == "QUERY_MARK_PRICE":
            sym = msg.body['symbol']
            self.mark_prices[sym] = msg.body['mark_price']
            self.oracle_prices[sym] = msg.body['oracle_price']
        elif msg_type == "QUERY_FUNDING_RATE":
            self.funding_rates[msg.body['symbol']] = msg.body['funding_rate']
        elif msg_type == "QUERY_POSITION":
            self.account.balance = msg.body.get('balance', self.account.balance)
        elif msg_type == "QUERY_TRANSACTED_VOLUME":
            sym = msg.body['symbol']
            self.transacted_volume[sym] = msg.body['transacted_volume']
        elif msg_type == "QUERY_ORDER_STREAM":
            if msg.body.get('mkt_closed'):
                self.mkt_closed = True
            self.stream_history[msg.body['symbol']] = msg.body['orders']
        elif msg_type == "MARKET_DATA":
            self._on_market_data(msg.body)

        have_mkt_hours = self.mkt_open is not None and self.mkt_close is not None
        if have_mkt_hours and not had_mkt_hours:
            ns_offset = self.getWakeFrequency()
            earliest = max(currentTime, self.mkt_open)
            self.setWakeup(earliest + ns_offset)

    # ── Order placement API ─────────────────────────────────────────────

    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price,
                        order_id=None, tag=None, time_in_force=TimeInForce.GTC,
                        reduce_only=False):
        order = PerpLimitOrder(
            self.id, self.currentTime, symbol, quantity, is_buy_order,
            limit_price, order_id=order_id, tag=tag,
            time_in_force=time_in_force, reduce_only=reduce_only,
        )
        if quantity > 0:
            self.orders[order.order_id] = deepcopy(order)
            self.sendMessage(self.exchangeID,
                             Message({"msg": "LIMIT_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', str(order))

    def placeMarketOrder(self, symbol, quantity, is_buy_order,
                         order_id=None, tag=None, reduce_only=False):
        order = PerpLimitOrder(
            self.id, self.currentTime, symbol, quantity, is_buy_order,
            limit_price=1e18 if is_buy_order else 0.0001,
            order_id=order_id, tag=tag, time_in_force=TimeInForce.IOC,
            reduce_only=reduce_only,
        )
        if quantity > 0:
            self.orders[order.order_id] = deepcopy(order)
            self.sendMessage(self.exchangeID,
                             Message({"msg": "MARKET_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', str(order))

    def placeTriggerOrder(self, symbol, quantity, is_buy_order, limit_price,
                          trigger_price, trigger_type, order_id=None, tag=None,
                          reduce_only=False):
        """Place a trigger (stop/take-profit) order.
        
        trigger_type: one of 'STOP_MARKET', 'STOP_LIMIT', 'TAKE_MARKET', 'TAKE_LIMIT'
        """
        order = PerpLimitOrder(
            self.id, self.currentTime, symbol, quantity, is_buy_order,
            limit_price, order_id=order_id, tag=tag,
            time_in_force=TimeInForce.GTC, reduce_only=reduce_only,
            trigger_price=trigger_price, trigger_type=trigger_type,
        )
        if quantity > 0:
            self.orders[order.order_id] = deepcopy(order)
            self.sendMessage(self.exchangeID,
                             Message({"msg": "TRIGGER_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('TRIGGER_ORDER_SUBMITTED', str(order))

    def cancelOrder(self, order):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "CANCEL_ORDER", "sender": self.id, "order": order}))
        if self.log_orders:
            self.logEvent('CANCEL_SUBMITTED', str(order))

    def modifyOrder(self, order, new_order):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "MODIFY_ORDER", "sender": self.id,
                                  "order": order, "new_order": new_order}))

    # ── Query helpers ───────────────────────────────────────────────────

    def getLastTrade(self, symbol):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "QUERY_LAST_TRADE", "sender": self.id, "symbol": symbol}))

    def getCurrentSpread(self, symbol, depth=1):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "QUERY_SPREAD", "sender": self.id,
                                  "symbol": symbol, "depth": depth}))

    def getMarkPrice(self, symbol):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "QUERY_MARK_PRICE", "sender": self.id, "symbol": symbol}))

    def getFundingRate(self, symbol):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "QUERY_FUNDING_RATE", "sender": self.id, "symbol": symbol}))

    def getPosition(self):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "QUERY_POSITION", "sender": self.id}))

    def get_transacted_volume(self, symbol, lookback_period='10min'):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "QUERY_TRANSACTED_VOLUME", "sender": self.id,
                                  "symbol": symbol, "lookback_period": lookback_period}))

    def requestDataSubscription(self, symbol, levels, freq):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "MARKET_DATA_SUBSCRIPTION_REQUEST",
                                  "sender": self.id, "symbol": symbol,
                                  "levels": levels, "freq": freq}))

    def cancelDataSubscription(self, symbol):
        self.sendMessage(self.exchangeID,
                         Message({"msg": "MARKET_DATA_SUBSCRIPTION_CANCELLATION",
                                  "sender": self.id, "symbol": symbol}))

    # ── Convenience accessors ───────────────────────────────────────────

    def getPositionSize(self, symbol):
        return self.account.get_position_size(symbol)

    def getBalance(self):
        return self.account.balance

    def getKnownBidAsk(self, symbol):
        bid = self.known_bids.get(symbol, [])
        ask = self.known_asks.get(symbol, [])
        best_bid = bid[0][0] if bid else None
        best_ask = ask[0][0] if ask else None
        return best_bid, best_ask

    def getKnownMidPrice(self, symbol):
        bb, ba = self.getKnownBidAsk(symbol)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    # ── Internal handlers ───────────────────────────────────────────────

    def _on_order_executed(self, order, fee=0.0):
        log_print("Execution notification: {}", order)
        if self.log_orders:
            self.logEvent('ORDER_EXECUTED', str(order))

        # Mirror the fill in our local account (fee from clearinghouse)
        self.account.apply_fill(
            order.symbol, order.quantity, order.fill_price,
            order.is_buy_order, self.default_leverage, fee=fee,
        )

        # Update open orders
        if order.order_id in self.orders:
            o = self.orders[order.order_id]
            if order.quantity >= o.quantity:
                del self.orders[order.order_id]
            else:
                o.quantity -= order.quantity

    def _on_order_accepted(self, order):
        log_print("Order accepted: {}", order)
        if self.log_orders:
            self.logEvent('ORDER_ACCEPTED', str(order))

    def _on_order_cancelled(self, order):
        log_print("Order cancelled: {}", order)
        if self.log_orders:
            self.logEvent('ORDER_CANCELLED', str(order))
        if order.order_id in self.orders:
            del self.orders[order.order_id]

    def _on_order_rejected(self, order, reason):
        log_print("Order rejected: {} reason: {}", order, reason)
        if self.log_orders:
            self.logEvent('ORDER_REJECTED', '{} reason={}'.format(order, reason))
        if order.order_id in self.orders:
            del self.orders[order.order_id]

    def _on_funding_payment(self, body):
        symbol = body['symbol']
        payment = body['payment']
        self.account.apply_funding(symbol, payment)
        self.funding_rates[symbol] = body['rate']
        log_print("Funding payment for {}: {:.4f} (rate={:.8f})", symbol, payment, body['rate'])

    def _on_liquidated(self, body):
        symbol = body['symbol']
        liq_type = body['type']
        log_print("LIQUIDATED: {} type={}", symbol, liq_type)
        if self.log_orders:
            self.logEvent('LIQUIDATED', str(body))

    def _on_position_settled(self, body):
        symbol = body['symbol']
        pnl = body['settled_pnl']
        self.account.settle_position(symbol, body['mark_price'])
        log_print("Position settled: {} pnl={:.4f}", symbol, pnl)

    def _on_market_data(self, body):
        symbol = body['symbol']
        self.known_bids[symbol] = body['bids']
        self.known_asks[symbol] = body['asks']
        self.last_trade[symbol] = body['last_transaction']
        self.exchange_ts[symbol] = body['exchange_ts']
        if 'mark_price' in body and body['mark_price'] is not None:
            self.mark_prices[symbol] = body['mark_price']
        if 'oracle_price' in body and body['oracle_price'] is not None:
            self.oracle_prices[symbol] = body['oracle_price']

    # ── To be overridden by subclasses ──────────────────────────────────

    def getWakeFrequency(self):
        """Override in subclass. Return a pd.Timedelta offset from market open."""
        return pd.Timedelta('0ns')

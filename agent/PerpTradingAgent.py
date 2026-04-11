"""Base trading agent for HIP-3 perpetual futures.

Provides margin-aware order placement, position tracking, and message handling.
Users subclass this and implement their own trading strategy by overriding
wakeup() and getWakeFrequency().

This agent does NOT define any trading logic -- it only provides infrastructure.
"""

from collections import defaultdict
from agent.FinancialAgent import FinancialAgent
from agent.PerpExchangeAgent import PerpExchangeAgent
from message.Message import Message
from util.order.PerpLimitOrder import PerpLimitOrder
from util.ContractSpec import MAX_PERP_SIGNIFICANT_FIGURES, MarginType, TimeInForce
from util.PerpAccount import PerpAccount
from util.util import log_print

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP
import pandas as pd


class PerpTradingAgent(FinancialAgent):

    def __init__(self, id, name, type, random_state=None, starting_cash=100000.0,
                 log_orders=False, log_to_file=True, default_leverage=10,
                 print_final_state=False,
                 price_decimals=None, size_decimals=None,
                 trading_rules_by_symbol=None):
        super().__init__(id, name, type, random_state, log_to_file)

        self.starting_cash = starting_cash
        self.default_leverage = default_leverage
        self.log_orders = log_orders
        self.print_final_state = print_final_state
        self.trading_rules_by_symbol = self._normalize_trading_rules(
            trading_rules_by_symbol=trading_rules_by_symbol,
            price_decimals=price_decimals,
            size_decimals=size_decimals,
        )

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
        self.rejection_reasons = {}
        self.rejection_reason_counts_by_symbol = defaultdict(lambda: defaultdict(int))
        self.local_skip_reasons_by_symbol = defaultdict(lambda: defaultdict(int))
        self.activity_counts_by_symbol = defaultdict(lambda: {
            "submitted": 0,
            "accepted": 0,
            "rejected": 0,
            "executed": 0,
            "cancelled": 0,
        })

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

        if self.print_final_state:
            positions_str = ", ".join(
                "{}:{:.4f}@{:.2f}".format(s, p.size, p.entry_price)
                for s, p in self.account.positions.items() if p.size != 0)
            print("Final state for {}: balance={:.2f}, positions=[{}], equity={:.2f}".format(
                self.name, self.account.balance, positions_str, equity))
            if self.rejection_reasons:
                reasons = ", ".join(
                    "{}={}".format(reason, count)
                    for reason, count in sorted(self.rejection_reasons.items())
                )
                print("Rejected orders for {}: {}".format(self.name, reasons))
            if self.activity_counts_by_symbol:
                print("Activity for {}: {}".format(
                    self.name,
                    self._format_activity_summary(),
                ))

        self.logEvent('PERP_ACTIVITY_SUMMARY', self._build_activity_summary(), True)
        self.logEvent('PERP_REJECTION_REASONS', self._stringify_nested_counts(self.rejection_reason_counts_by_symbol), True)
        self.logEvent('PERP_LOCAL_SKIP_REASONS', self._stringify_nested_counts(self.local_skip_reasons_by_symbol), True)
        self.logEvent('PERP_OPEN_ORDERS_AT_STOP', self._open_orders_at_stop(), True)

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
            self._on_order_modified(msg.body['order'])
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
            positions = msg.body.get('positions', {})
            for sym, payload in positions.items():
                pos = self.account.ensure_position(
                    sym,
                    leverage=payload.get('leverage', self.default_leverage),
                    margin_type=MarginType(payload.get('margin_type', MarginType.CROSS.value)),
                )
                pos.size = payload.get('size', pos.size)
                pos.entry_price = payload.get('entry_price', pos.entry_price)
                pos.leverage = payload.get('leverage', pos.leverage)
                pos.isolated_margin = payload.get('isolated_margin', pos.isolated_margin)
        elif msg_type == "QUERY_TRANSACTED_VOLUME":
            sym = msg.body['symbol']
            self.transacted_volume[sym] = msg.body['transacted_volume']
        elif msg_type == "QUERY_ORDER_STREAM":
            if msg.body.get('mkt_closed'):
                self.mkt_closed = True
            self.stream_history[msg.body['symbol']] = msg.body['orders']
        elif msg_type == "MARKET_DATA":
            self._on_market_data(msg.body)
        elif msg_type in ("TWAP_ACCEPTED", "TWAP_CANCELLED", "TWAP_COMPLETE",
                          "SCALE_ACCEPTED", "SCALE_REJECTED"):
            pass  # Informational — subclasses can override to track

        have_mkt_hours = self.mkt_open is not None and self.mkt_close is not None
        if have_mkt_hours and not had_mkt_hours:
            ns_offset = self.getWakeFrequency()
            earliest = max(currentTime, self.mkt_open)
            self.setWakeup(earliest + ns_offset)

    # ── Order placement API ─────────────────────────────────────────────

    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price,
                        order_id=None, tag=None, time_in_force=TimeInForce.GTC,
                        reduce_only=False, requested_leverage=None,
                        margin_type=MarginType.INHERIT, parent_order_id=None,
                        tpsl_group_id=None, tpsl_mode=None,
                        trigger_slippage_bps=None, dynamic_size=False):
        quantity, limit_price = self._prepare_order_submission(
            symbol=symbol,
            quantity=quantity,
            is_buy_order=is_buy_order,
            limit_price=limit_price,
            is_market_order=False,
            reduce_only=reduce_only,
        )
        if quantity is None or limit_price is None:
            return
        order = PerpLimitOrder(
            self.id, self.currentTime, symbol, quantity, is_buy_order,
            limit_price, order_id=order_id, tag=tag,
            time_in_force=time_in_force, reduce_only=reduce_only,
            requested_leverage=requested_leverage,
            margin_type=margin_type,
            parent_order_id=parent_order_id,
            tpsl_group_id=tpsl_group_id,
            tpsl_mode=tpsl_mode,
            trigger_slippage_bps=trigger_slippage_bps,
            dynamic_size=dynamic_size,
        )
        if quantity > 0:
            self.orders[order.order_id] = order.clone()
            self._increment_activity(symbol, 'submitted')
            self.sendMessage(self.exchangeID,
                             Message({"msg": "LIMIT_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', str(order))

    def placeMarketOrder(self, symbol, quantity, is_buy_order,
                         order_id=None, tag=None, reduce_only=False,
                         requested_leverage=None, margin_type=MarginType.INHERIT,
                         parent_order_id=None, tpsl_group_id=None, tpsl_mode=None,
                         trigger_slippage_bps=None, dynamic_size=False):
        quantity, _reference_price = self._prepare_order_submission(
            symbol=symbol,
            quantity=quantity,
            is_buy_order=is_buy_order,
            limit_price=None,
            is_market_order=True,
            reduce_only=reduce_only,
        )
        if quantity is None:
            return
        order = PerpLimitOrder(
            self.id, self.currentTime, symbol, quantity, is_buy_order,
            limit_price=1e18 if is_buy_order else 0.0001,
            order_id=order_id, tag=tag, time_in_force=TimeInForce.IOC,
            reduce_only=reduce_only,
            requested_leverage=requested_leverage,
            margin_type=margin_type,
            parent_order_id=parent_order_id,
            tpsl_group_id=tpsl_group_id,
            tpsl_mode=tpsl_mode,
            trigger_slippage_bps=trigger_slippage_bps,
            dynamic_size=dynamic_size,
            is_market_order=True,
        )
        if quantity > 0:
            self.orders[order.order_id] = order.clone()
            self._increment_activity(symbol, 'submitted')
            self.sendMessage(self.exchangeID,
                             Message({"msg": "MARKET_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', str(order))

    def placeTriggerOrder(self, symbol, quantity, is_buy_order, limit_price,
                          trigger_price, trigger_type, order_id=None, tag=None,
                          reduce_only=False, requested_leverage=None,
                          margin_type=MarginType.INHERIT, parent_order_id=None,
                          tpsl_group_id=None, tpsl_mode=None,
                          trigger_slippage_bps=None, dynamic_size=False):
        """Place a trigger (stop/take-profit) order.
        
        trigger_type: one of 'STOP_MARKET', 'STOP_LIMIT', 'TAKE_MARKET', 'TAKE_LIMIT'
        """
        quantity, limit_price = self._prepare_order_submission(
            symbol=symbol,
            quantity=quantity,
            is_buy_order=is_buy_order,
            limit_price=limit_price,
            is_market_order=False,
            reduce_only=reduce_only,
        )
        trigger_price = self._round_price(symbol, trigger_price)
        if quantity is None or limit_price is None or trigger_price is None:
            self._record_local_skip(symbol, 'INVALID_TRIGGER_PRICE')
            return
        order = PerpLimitOrder(
            self.id, self.currentTime, symbol, quantity, is_buy_order,
            limit_price, order_id=order_id, tag=tag,
            time_in_force=TimeInForce.GTC, reduce_only=reduce_only,
            trigger_price=trigger_price, trigger_type=trigger_type,
            requested_leverage=requested_leverage,
            margin_type=margin_type,
            parent_order_id=parent_order_id,
            tpsl_group_id=tpsl_group_id,
            tpsl_mode=tpsl_mode,
            trigger_slippage_bps=trigger_slippage_bps,
            dynamic_size=dynamic_size,
        )
        if quantity > 0:
            self.orders[order.order_id] = order.clone()
            self._increment_activity(symbol, 'submitted')
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

    def placeBatchOrders(self, orders):
        valid_orders = []
        for order in orders:
            prepared = self._prepare_existing_order(order)
            if prepared is None:
                continue
            valid_orders.append(prepared)
        for order in valid_orders:
            self.orders[order.order_id] = order.clone()
            self._increment_activity(order.symbol, 'submitted')
        if valid_orders:
            self.sendMessage(self.exchangeID, Message({"msg": "PLACE_BATCH", "sender": self.id, "orders": valid_orders}))

    def cancelBatchOrders(self, orders):
        self.sendMessage(self.exchangeID, Message({"msg": "CANCEL_BATCH", "sender": self.id, "orders": orders}))

    def modifyBatchOrders(self, updates):
        self.sendMessage(self.exchangeID, Message({"msg": "MODIFY_BATCH", "sender": self.id, "updates": updates}))

    def updateLeverage(self, symbol, leverage):
        self.sendMessage(self.exchangeID, Message({"msg": "_LEVERAGE_UPDATE", "sender": self.id, "symbol": symbol, "leverage": leverage}))

    def adjustIsolatedMargin(self, symbol, amount):
        self.sendMessage(self.exchangeID, Message({"msg": "_ISOLATED_MARGIN_ADJUST", "sender": self.id, "symbol": symbol, "amount": amount}))

    def depositCollateral(self, amount):
        self.sendMessage(self.exchangeID, Message({"msg": "_COLLATERAL_TRANSFER", "sender": self.id, "direction": "deposit", "amount": amount}))

    def withdrawCollateral(self, amount):
        self.sendMessage(self.exchangeID, Message({"msg": "_COLLATERAL_TRANSFER", "sender": self.id, "direction": "withdraw", "amount": amount}))

    def placeTwapOrder(self, symbol, total_quantity, is_buy, num_slices,
                       interval_ms=1000, reduce_only=False, requested_leverage=None):
        self.sendMessage(self.exchangeID, Message({
            "msg": "TWAP_ORDER",
            "sender": self.id,
            "symbol": symbol,
            "total_quantity": total_quantity,
            "is_buy": is_buy,
            "num_slices": num_slices,
            "interval_ms": interval_ms,
            "reduce_only": reduce_only,
            "requested_leverage": requested_leverage,
        }))

    def cancelTwap(self, twap_id):
        self.sendMessage(self.exchangeID, Message({
            "msg": "CANCEL_TWAP",
            "sender": self.id,
            "twap_id": twap_id,
        }))

    def placeScaleOrder(self, symbol, total_quantity, is_buy, num_orders,
                        price_low, price_high, reduce_only=False,
                        requested_leverage=None, distribution="uniform"):
        self.sendMessage(self.exchangeID, Message({
            "msg": "SCALE_ORDER",
            "sender": self.id,
            "symbol": symbol,
            "total_quantity": total_quantity,
            "is_buy": is_buy,
            "num_orders": num_orders,
            "price_low": price_low,
            "price_high": price_high,
            "reduce_only": reduce_only,
            "requested_leverage": requested_leverage,
            "distribution": distribution,
        }))

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
        self._increment_activity(order.symbol, 'executed')

        # Mirror the fill in our local account (fee from clearinghouse)
        self.account.apply_fill(
            order.symbol, order.quantity, order.fill_price,
            order.is_buy_order, order.requested_leverage or self.default_leverage, fee=fee,
            margin_type=(
                order.margin_type
                if hasattr(order, 'margin_type') and order.margin_type != MarginType.INHERIT
                else self.account.get_position(order.symbol).margin_type
                if self.account.has_position(order.symbol)
                else MarginType.CROSS
            ),
            order_id=order.order_id,
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
        self._increment_activity(order.symbol, 'accepted')

    def _on_order_cancelled(self, order):
        log_print("Order cancelled: {}", order)
        if self.log_orders:
            self.logEvent('ORDER_CANCELLED', str(order))
        self._increment_activity(order.symbol, 'cancelled')
        if order.order_id in self.orders:
            del self.orders[order.order_id]

    def _on_order_modified(self, order):
        log_print("Order modified: {}", order)
        if self.log_orders:
            self.logEvent('ORDER_MODIFIED', str(order))
        self.orders[order.order_id] = order.clone()

    def _on_order_rejected(self, order, reason):
        log_print("Order rejected: {} reason: {}", order, reason)
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        self._increment_activity(order.symbol, 'rejected')
        self.rejection_reason_counts_by_symbol[order.symbol][reason] += 1
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

    def _normalize_trading_rules(self, trading_rules_by_symbol, price_decimals, size_decimals):
        normalized = {}
        for symbol, rules in (trading_rules_by_symbol or {}).items():
            symbol_rules = dict(rules)
            symbol_rules.setdefault('sz_decimals', self._precision_for(symbol, size_decimals, 4))
            symbol_rules.setdefault('max_price_decimals', self._precision_for(symbol, price_decimals, 6))
            symbol_rules.setdefault('max_significant_figures', MAX_PERP_SIGNIFICANT_FIGURES)
            symbol_rules.setdefault('min_order_value', 0.0)
            symbol_rules.setdefault('max_limit_order_value', None)
            symbol_rules.setdefault('max_market_order_value', None)
            normalized[symbol] = symbol_rules

        for configured in (price_decimals, size_decimals):
            if not isinstance(configured, dict):
                continue
            for symbol in configured:
                normalized.setdefault(symbol, {
                    'sz_decimals': self._precision_for(symbol, size_decimals, 4),
                    'max_price_decimals': self._precision_for(symbol, price_decimals, 6),
                    'max_significant_figures': MAX_PERP_SIGNIFICANT_FIGURES,
                    'min_order_value': 0.0,
                    'max_limit_order_value': None,
                    'max_market_order_value': None,
                })
        return normalized

    def _precision_for(self, symbol, configured, default):
        if isinstance(configured, dict):
            return configured.get(symbol, default)
        if configured is None:
            return default
        return configured

    def _get_trading_rules(self, symbol):
        return self.trading_rules_by_symbol.get(symbol, {
            'sz_decimals': 4,
            'max_price_decimals': 6,
            'max_significant_figures': MAX_PERP_SIGNIFICANT_FIGURES,
            'min_order_value': 0.0,
            'max_limit_order_value': None,
            'max_market_order_value': None,
        })

    def _to_decimal(self, value):
        try:
            return Decimal(str(float(value)))
        except (InvalidOperation, ValueError, TypeError):
            return None

    def _round_price(self, symbol, price):
        if price is None:
            return None
        dec_price = self._to_decimal(price)
        if dec_price is None or dec_price <= 0:
            return None
        rules = self._get_trading_rules(symbol)
        max_decimals = int(rules.get('max_price_decimals', 6))
        max_sig_figs = int(rules.get('max_significant_figures', MAX_PERP_SIGNIFICANT_FIGURES))

        if dec_price == dec_price.to_integral():
            return float(dec_price)

        adjusted = dec_price.adjusted()
        decimals_for_sig = max(0, max_sig_figs - adjusted - 1)
        decimals = min(max_decimals, decimals_for_sig)
        rounded = dec_price.quantize(Decimal('1').scaleb(-decimals), rounding=ROUND_HALF_UP)
        if rounded <= 0:
            return None
        return float(rounded)

    def _round_quantity(self, symbol, quantity):
        dec_qty = self._to_decimal(abs(quantity))
        if dec_qty is None:
            return 0.0
        decimals = int(self._get_trading_rules(symbol).get('sz_decimals', 4))
        return float(dec_qty.quantize(Decimal('1').scaleb(-decimals), rounding=ROUND_HALF_UP))

    def _ceil_quantity(self, symbol, quantity):
        dec_qty = self._to_decimal(abs(quantity))
        if dec_qty is None:
            return 0.0
        decimals = int(self._get_trading_rules(symbol).get('sz_decimals', 4))
        return float(dec_qty.quantize(Decimal('1').scaleb(-decimals), rounding=ROUND_CEILING))

    def _reference_price_for_order(self, symbol, is_buy_order, fallback=None):
        best_bid, best_ask = self.getKnownBidAsk(symbol)
        if is_buy_order and best_ask:
            return best_ask
        if (not is_buy_order) and best_bid:
            return best_bid
        mid = self.getKnownMidPrice(symbol)
        if mid:
            return mid
        for candidate in (
            self.mark_prices.get(symbol),
            self.last_trade.get(symbol),
            self.oracle_prices.get(symbol),
            fallback,
        ):
            if candidate is not None and candidate > 0 and candidate < 1e12:
                return candidate
        return None

    def _prepare_order_submission(self, symbol, quantity, is_buy_order, limit_price,
                                  is_market_order, reduce_only):
        quantity = self._round_quantity(symbol, quantity)
        if quantity <= 0:
            self._record_local_skip(symbol, 'ZERO_QUANTITY')
            return None, None

        conformed_price = limit_price
        if not is_market_order:
            conformed_price = self._round_price(symbol, limit_price)
            if conformed_price is None:
                self._record_local_skip(symbol, 'INVALID_PRICE')
                return None, None

        reference_price = self._reference_price_for_order(
            symbol,
            is_buy_order,
            fallback=conformed_price,
        )
        rules = self._get_trading_rules(symbol)
        max_order_value = rules.get('max_market_order_value') if is_market_order else rules.get('max_limit_order_value')
        notional = quantity * reference_price if reference_price else 0.0
        if max_order_value is not None and reference_price and notional > max_order_value + 1e-12:
            self._record_local_skip(symbol, 'ABOVE_MAX_ORDER_VALUE')
            return None, None

        min_order_value = float(rules.get('min_order_value') or 0.0)
        if (not reduce_only) and reference_price and min_order_value > 0 and notional < min_order_value - 1e-12:
            quantity = self._ceil_quantity(symbol, min_order_value / reference_price)
            notional = quantity * reference_price
            if quantity <= 0 or (max_order_value is not None and notional > max_order_value + 1e-12):
                self._record_local_skip(symbol, 'BELOW_MIN_NOTIONAL')
                return None, None

        return quantity, conformed_price

    def _prepare_existing_order(self, order):
        quantity, price = self._prepare_order_submission(
            symbol=order.symbol,
            quantity=order.quantity,
            is_buy_order=order.is_buy_order,
            limit_price=None if getattr(order, 'is_market_order', False) else order.limit_price,
            is_market_order=getattr(order, 'is_market_order', False),
            reduce_only=getattr(order, 'reduce_only', False),
        )
        if quantity is None:
            return None
        prepared = order.clone()
        prepared.quantity = quantity
        if price is not None:
            prepared.limit_price = price
        if getattr(prepared, 'trigger_price', None) is not None:
            prepared.trigger_price = self._round_price(prepared.symbol, prepared.trigger_price)
            if prepared.trigger_price is None:
                self._record_local_skip(prepared.symbol, 'INVALID_TRIGGER_PRICE')
                return None
        return prepared

    def _increment_activity(self, symbol, key, amount=1):
        self.activity_counts_by_symbol[symbol][key] += amount

    def _record_local_skip(self, symbol, reason):
        self.local_skip_reasons_by_symbol[symbol][reason] += 1
        if self.log_orders:
            self.logEvent('ORDER_LOCAL_SKIP', {'symbol': symbol, 'reason': reason})

    def _stringify_nested_counts(self, nested_counts):
        return {
            symbol: dict(sorted(reason_counts.items()))
            for symbol, reason_counts in sorted(nested_counts.items())
        }

    def _open_orders_at_stop(self):
        open_counts = defaultdict(int)
        for order in self.orders.values():
            open_counts[order.symbol] += 1
        return dict(sorted(open_counts.items()))

    def _activity_status_for_symbol(self, symbol, counts, open_orders):
        local_skips = sum(self.local_skip_reasons_by_symbol.get(symbol, {}).values())
        if counts['submitted'] == 0 and counts['accepted'] == 0 and counts['rejected'] == 0 and local_skips == 0:
            return 'no_activity'
        if counts['executed'] > 0:
            return 'filled_and_carrying' if self.getPositionSize(symbol) != 0 else 'filled'
        if open_orders > 0:
            return 'quoted_unfilled'
        if counts['rejected'] > 0 or local_skips > 0:
            return 'rejected_or_skipped'
        if counts['accepted'] > 0 or counts['cancelled'] > 0:
            return 'quoted_unfilled'
        return 'no_activity'

    def _build_activity_summary(self):
        symbols = set(self.activity_counts_by_symbol.keys()) | {order.symbol for order in self.orders.values()}
        symbols |= set(self.rejection_reason_counts_by_symbol.keys()) | set(self.local_skip_reasons_by_symbol.keys())
        summary = {}
        open_orders = self._open_orders_at_stop()
        for symbol in sorted(symbols):
            counts = dict(self.activity_counts_by_symbol.get(symbol, {}))
            if not counts:
                counts = {
                    "submitted": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "executed": 0,
                    "cancelled": 0,
                }
            symbol_open_orders = int(open_orders.get(symbol, 0))
            summary[symbol] = {
                **counts,
                "open_orders_at_stop": symbol_open_orders,
                "rejection_reasons": dict(sorted(self.rejection_reason_counts_by_symbol.get(symbol, {}).items())),
                "local_skip_reasons": dict(sorted(self.local_skip_reasons_by_symbol.get(symbol, {}).items())),
                "status": self._activity_status_for_symbol(symbol, counts, symbol_open_orders),
            }
        return summary

    def _format_activity_summary(self):
        parts = []
        for symbol, payload in self._build_activity_summary().items():
            parts.append(
                "{}: submitted={}, accepted={}, rejected={}, executed={}, cancelled={}, open={}, status={}".format(
                    symbol,
                    payload['submitted'],
                    payload['accepted'],
                    payload['rejected'],
                    payload['executed'],
                    payload['cancelled'],
                    payload['open_orders_at_stop'],
                    payload['status'],
                )
            )
        return "; ".join(parts)

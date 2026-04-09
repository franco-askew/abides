"""HIP-3 Perpetual Futures Exchange Agent.

Owns the order book, clearinghouse, mark price engine, funding engine, and liquidation engine.
Processes orders, oracle updates, funding settlements, and liquidations.
Supports 24/7 trading (no mkt_close guard on orders).
"""

from agent.FinancialAgent import FinancialAgent
from message.Message import Message, MessageType
from util.PerpOrderBook import PerpOrderBook
from util.MarkPriceEngine import MarkPriceEngine
from util.Clearinghouse import Clearinghouse
from util.FundingEngine import FundingEngine
from util.LiquidationEngine import LiquidationEngine
from util.ContractSpec import PerpDexConfig, TimeInForce
from util.order.PerpLimitOrder import PerpLimitOrder
from util.util import log_print

from copy import deepcopy
import pandas as pd


class PerpExchangeAgent(FinancialAgent):

    def __init__(self, id, name, type, dex_config: PerpDexConfig,
                 pipeline_delay=40000, computation_delay=1,
                 stream_history=10, log_orders=False, random_state=None):

        super().__init__(id, name, type, random_state)
        self.reschedule = False
        self.dex_config = dex_config
        self.pipeline_delay = pipeline_delay
        self.computation_delay = computation_delay
        self.stream_history = stream_history
        self.log_orders = log_orders

        # Core components
        self.order_books = {}
        self.mark_engines = {}
        self.clearinghouse = Clearinghouse(dex_config.assets, dex_config.fee_schedule)
        self.funding_engine = FundingEngine()
        self.liquidation_engine = LiquidationEngine()

        # Initialize per-symbol components
        for symbol, spec in dex_config.assets.items():
            self.order_books[symbol] = PerpOrderBook(self, symbol, spec)
            self.mark_engines[symbol] = MarkPriceEngine(spec.initial_oracle_px)
            self.order_books[symbol].last_trade = spec.initial_oracle_px

        # Subscriptions (same pattern as original ExchangeAgent)
        self.subscription_dict = {}

        # Trigger orders storage
        self.trigger_orders = {}  # order_id -> PerpLimitOrder

        # Funding state
        self.premium_sample_interval_ns = 5_000_000_000  # 5 seconds
        self.funding_interval_ns = 3_600_000_000_000     # 1 hour
        self.last_premium_sample_time = None
        self.last_funding_time = None

        # Trading halted symbols
        self.halted_symbols = set()

    def kernelInitializing(self, kernel):
        super().kernelInitializing(kernel)
        self.oracle = self.kernel.oracle

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.last_premium_sample_time = startTime
        self.last_funding_time = startTime
        self._schedule_premium_sample(startTime)
        self._schedule_funding(startTime)

    def kernelTerminating(self):
        super().kernelTerminating()
        if hasattr(self.oracle, 'f_log'):
            for symbol in self.oracle.f_log:
                dfFund = pd.DataFrame(self.oracle.f_log[symbol])
                if not dfFund.empty:
                    dfFund.set_index('FundamentalTime', inplace=True)
                    self.writeLog(dfFund, filename='fundamental_{}'.format(symbol))

    # ── Message dispatch ────────────────────────────────────────────────

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        self.setComputationDelay(self.computation_delay)

        msg_type = msg.body['msg']

        # Log all messages
        if msg_type in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
            if self.log_orders:
                self.logEvent(msg_type, str(msg.body.get('order', '')))
        else:
            self.logEvent(msg_type, msg.body.get('sender', ''))

        # ── Subscription management ────────────────────────────────
        if msg_type in ["MARKET_DATA_SUBSCRIPTION_REQUEST", "MARKET_DATA_SUBSCRIPTION_CANCELLATION"]:
            self._update_subscription(msg, currentTime)
            return

        # ── Immutable queries (no delay) ───────────────────────────
        if msg_type == "WHEN_MKT_OPEN":
            self.setComputationDelay(0)
            self.sendMessage(msg.body['sender'],
                             Message({"msg": "WHEN_MKT_OPEN", "data": self.kernel.startTime}))
            return

        if msg_type == "WHEN_MKT_CLOSE":
            self.setComputationDelay(0)
            self.sendMessage(msg.body['sender'],
                             Message({"msg": "WHEN_MKT_CLOSE", "data": self.kernel.stopTime}))
            return

        # ── Data queries ───────────────────────────────────────────
        if msg_type == "QUERY_LAST_TRADE":
            symbol = msg.body['symbol']
            if symbol in self.order_books:
                self.sendMessage(msg.body['sender'], Message({
                    "msg": "QUERY_LAST_TRADE", "symbol": symbol,
                    "data": self.order_books[symbol].last_trade,
                    "mkt_closed": False,
                }))
            return

        if msg_type == "QUERY_SPREAD":
            symbol = msg.body['symbol']
            depth = msg.body['depth']
            if symbol in self.order_books:
                ob = self.order_books[symbol]
                self.sendMessage(msg.body['sender'], Message({
                    "msg": "QUERY_SPREAD", "symbol": symbol, "depth": depth,
                    "bids": ob.getInsideBids(depth),
                    "asks": ob.getInsideAsks(depth),
                    "data": ob.last_trade,
                    "mkt_closed": False, "book": "",
                }))
            return

        if msg_type == "QUERY_ORDER_STREAM":
            symbol = msg.body['symbol']
            length = msg.body['length']
            if symbol in self.order_books:
                self.sendMessage(msg.body['sender'], Message({
                    "msg": "QUERY_ORDER_STREAM", "symbol": symbol, "length": length,
                    "mkt_closed": False,
                    "orders": self.order_books[symbol].history[1:length + 1],
                }))
            return

        if msg_type == 'QUERY_TRANSACTED_VOLUME':
            symbol = msg.body['symbol']
            lookback = msg.body.get('lookback_period', '10min')
            if symbol in self.order_books:
                vol = self.order_books[symbol].get_transacted_volume(lookback)
                self.sendMessage(msg.body['sender'], Message({
                    "msg": "QUERY_TRANSACTED_VOLUME", "symbol": symbol,
                    "transacted_volume": vol,
                    "mkt_closed": False,
                }))
            return

        if msg_type == "QUERY_MARK_PRICE":
            symbol = msg.body['symbol']
            me = self.mark_engines.get(symbol)
            self.sendMessage(msg.body['sender'], Message({
                "msg": "QUERY_MARK_PRICE", "symbol": symbol,
                "mark_price": me.mark_price if me else None,
                "oracle_price": me.oracle_price if me else None,
            }))
            return

        if msg_type == "QUERY_FUNDING_RATE":
            symbol = msg.body['symbol']
            rate = self.funding_engine.get_last_rate(symbol)
            self.sendMessage(msg.body['sender'], Message({
                "msg": "QUERY_FUNDING_RATE", "symbol": symbol,
                "funding_rate": rate,
            }))
            return

        if msg_type == "QUERY_POSITION":
            agent_id = msg.body['sender']
            account = self.clearinghouse.get_account(agent_id)
            positions = {}
            if account:
                for sym, pos in account.positions.items():
                    positions[sym] = {
                        'size': pos.size, 'entry_price': pos.entry_price,
                        'leverage': pos.leverage, 'unrealized_pnl': pos.unrealized_pnl(
                            self.mark_engines[sym].mark_price if sym in self.mark_engines else pos.entry_price),
                    }
            self.sendMessage(agent_id, Message({
                "msg": "QUERY_POSITION", "positions": positions,
                "balance": account.balance if account else 0,
            }))
            return

        # ── Order handling ─────────────────────────────────────────
        if msg_type == "LIMIT_ORDER":
            order = msg.body['order']
            if order.symbol in self.halted_symbols:
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_REJECTED", "order": order, "reason": "HALTED"}))
                return
            if getattr(order, 'trigger_price', None) is not None:
                self.trigger_orders[order.order_id] = order
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))
                return
            if order.symbol in self.order_books:
                self._handle_limit_order(order)
                self._publish_order_book_data()
            return

        if msg_type == "MARKET_ORDER":
            order = msg.body['order']
            if order.symbol in self.halted_symbols:
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_REJECTED", "order": order, "reason": "HALTED"}))
                return
            if order.symbol in self.order_books:
                self._handle_market_order(order)
                self._publish_order_book_data()
            return

        if msg_type == "TRIGGER_ORDER":
            order = msg.body['order']
            if order.symbol in self.halted_symbols:
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_REJECTED", "order": order, "reason": "HALTED"}))
                return
            if order.trigger_price is not None and order.trigger_type is not None:
                self.trigger_orders[order.order_id] = order
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))
            else:
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_REJECTED", "order": order, "reason": "MISSING_TRIGGER"}))
            return

        if msg_type == "CANCEL_ORDER":
            order = msg.body['order']
            # Check trigger orders first
            if order.order_id in self.trigger_orders:
                cancelled = self.trigger_orders.pop(order.order_id)
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_CANCELLED", "order": cancelled}))
                return
            if order.symbol in self.order_books:
                self.order_books[order.symbol].cancelOrder(deepcopy(order))
                self._publish_order_book_data()
            return

        if msg_type == "MODIFY_ORDER":
            order = msg.body['order']
            new_order = msg.body['new_order']
            if order.symbol in self.order_books:
                self.order_books[order.symbol].modifyOrder(deepcopy(order), deepcopy(new_order))
                self._publish_order_book_data()
            return

        # ── HIP-3 deployer messages ────────────────────────────────
        if msg_type == "SET_ORACLE":
            self._handle_set_oracle(msg.body)
            return

        if msg_type == "HALT_TRADING":
            self._handle_halt_trading(msg.body)
            return

        if msg_type == "SET_OI_CAPS":
            symbol = msg.body['symbol']
            self.clearinghouse.update_oi_caps(
                symbol, msg.body['notional_cap'], msg.body['size_cap'])
            return

        if msg_type == "SET_FUNDING_MULTIPLIERS":
            for symbol, mult in msg.body['multipliers'].items():
                self.clearinghouse.update_funding_multiplier(symbol, mult)
            return

        if msg_type == "SET_MARGIN_TABLE":
            from util.ContractSpec import MarginTable, MarginTier
            symbol = msg.body['symbol']
            tiers = [MarginTier(t['lower_bound_notional'], t['max_leverage'])
                     for t in msg.body['tiers']]
            self.clearinghouse.update_margin_table(symbol, MarginTable(tiers=tiers))
            return

        # ── Internal scheduled events ──────────────────────────────
        if msg_type == "_PREMIUM_SAMPLE":
            self._do_premium_sample(currentTime)
            self._schedule_premium_sample(currentTime)
            return

        if msg_type == "_FUNDING_SETTLE":
            self._do_funding_settlement(currentTime)
            self._schedule_funding(currentTime)
            return

    # ── Order validation ────────────────────────────────────────────────

    def _validate_order(self, order, is_market=False):
        """Validate tick/lot size, min/max order value. Returns (ok, reason)."""
        spec = self.dex_config.assets.get(order.symbol)
        if spec is None:
            return True, None

        order.quantity = round(order.quantity, spec.sz_decimals)
        if order.quantity <= 0:
            return False, "ZERO_QUANTITY"

        if is_market:
            me = self.mark_engines.get(order.symbol)
            ref_price = me.mark_price if me else order.limit_price
        else:
            tick = spec.tick_size
            order.limit_price = round(order.limit_price / tick) * tick
            ref_price = order.limit_price

        notional = order.quantity * ref_price
        if notional < spec.min_order_value:
            return False, "MIN_ORDER_VALUE"

        max_val = spec.max_market_order_value if is_market else spec.max_limit_order_value
        if notional > max_val:
            return False, "MAX_ORDER_VALUE"

        return True, None

    def _check_order_risk(self, order, account, symbol, notional_estimate):
        """Shared margin and OI check for both limit and market orders.
        Returns (ok, reason)."""
        if order.reduce_only or order.is_liquidation:
            return True, None

        mark_price = self.mark_engines[symbol].mark_price
        pos = account.get_position(symbol)
        is_increasing = (pos.size >= 0 and order.is_buy_order) or (pos.size <= 0 and not order.is_buy_order)

        if not is_increasing and pos.size != 0:
            return True, None

        if not self.clearinghouse.check_oi_cap(symbol, order.quantity, mark_price):
            return False, "OI_CAP"

        spec = self.dex_config.assets.get(symbol)
        leverage = pos.leverage if pos.size != 0 else (spec.default_leverage if spec else 10)
        if spec:
            max_lev = spec.margin_table.get_max_leverage(notional_estimate)
            leverage = min(leverage, max_lev)

        mark_prices = {s: me.mark_price for s, me in self.mark_engines.items()}
        margin_tables = self.clearinghouse.get_margin_tables()
        if not account.check_initial_margin(symbol, notional_estimate, leverage,
                                             mark_prices, margin_tables,
                                             pos.is_isolated):
            return False, "INSUFFICIENT_MARGIN"

        return True, None

    # ── Order processing ────────────────────────────────────────────────

    def _handle_limit_order(self, order):
        symbol = order.symbol
        account = self.clearinghouse.get_or_create_account(order.agent_id)

        ok, reason = self._validate_order(order, is_market=False)
        if not ok:
            self.sendMessage(order.agent_id,
                             Message({"msg": "ORDER_REJECTED", "order": order, "reason": reason}))
            return

        notional = order.quantity * order.limit_price
        ok, reason = self._check_order_risk(order, account, symbol, notional)
        if not ok:
            self.sendMessage(order.agent_id,
                             Message({"msg": "ORDER_REJECTED", "order": order, "reason": reason}))
            return

        agent_positions = {s: p.size for s, p in account.positions.items()}

        ob = self.order_books[symbol]
        fills = ob.handleLimitOrder(deepcopy(order), agent_positions)

        self._process_fills(order, symbol, fills)

    def _handle_market_order(self, order):
        symbol = order.symbol
        account = self.clearinghouse.get_or_create_account(order.agent_id)

        ok, reason = self._validate_order(order, is_market=True)
        if not ok:
            self.sendMessage(order.agent_id,
                             Message({"msg": "ORDER_REJECTED", "order": order, "reason": reason}))
            return

        mark_price = self.mark_engines[symbol].mark_price
        notional_estimate = order.quantity * mark_price
        ok, reason = self._check_order_risk(order, account, symbol, notional_estimate)
        if not ok:
            self.sendMessage(order.agent_id,
                             Message({"msg": "ORDER_REJECTED", "order": order, "reason": reason}))
            return

        agent_positions = {s: p.size for s, p in account.positions.items()}

        ob = self.order_books[symbol]
        fills = ob.handleMarketOrder(deepcopy(order), agent_positions)

        self._process_fills(order, symbol, fills)

    def _process_fills(self, order, symbol, fills):
        """Process both taker and maker sides of each fill through the clearinghouse."""
        if not fills:
            return

        spec = self.dex_config.assets.get(symbol)
        default_lev = spec.default_leverage if spec else 10

        for filled_order, matched_order in fills:
            taker_account = self.clearinghouse.get_or_create_account(order.agent_id)
            taker_pos = taker_account.get_position(symbol)
            taker_lev = taker_pos.leverage if taker_pos.size != 0 else default_lev

            taker_fee = self.clearinghouse.process_fill(
                order.agent_id, symbol, filled_order.quantity, filled_order.fill_price,
                order.is_buy_order, is_taker=True,
                leverage=taker_lev,
                is_liquidation=order.is_liquidation,
            )

            maker_account = self.clearinghouse.get_or_create_account(matched_order.agent_id)
            maker_pos = maker_account.get_position(symbol)
            maker_lev = maker_pos.leverage if maker_pos.size != 0 else default_lev

            maker_fee = self.clearinghouse.process_fill(
                matched_order.agent_id, symbol, filled_order.quantity, filled_order.fill_price,
                matched_order.is_buy_order, is_taker=False,
                leverage=maker_lev,
                is_liquidation=getattr(matched_order, 'is_liquidation', False),
            )

            self.sendMessage(filled_order.agent_id,
                             Message({"msg": "ORDER_EXECUTED", "order": filled_order, "fee": taker_fee}))
            self.sendMessage(matched_order.agent_id,
                             Message({"msg": "ORDER_EXECUTED", "order": matched_order, "fee": maker_fee}))

        # Recalculate OI from all accounts to avoid double-counting
        mark_price = self.mark_engines[symbol].mark_price
        self.clearinghouse.recalculate_oi(symbol, mark_price)

    # ── Oracle and mark price ───────────────────────────────────────────

    def _handle_set_oracle(self, body):
        """Process SET_ORACLE from the OracleDeployerAgent."""
        oracle_pxs = body.get('oracle_pxs', {})
        mark_pxs_list = body.get('mark_pxs', [])
        external_perp_pxs = body.get('external_perp_pxs', {})

        for symbol, oracle_px in oracle_pxs.items():
            if symbol not in self.mark_engines:
                continue

            me = self.mark_engines[symbol]
            ob = self.order_books[symbol]

            local_mark = ob.getLocalMarkPrice()
            deployer_marks = mark_pxs_list if isinstance(mark_pxs_list, list) else []

            oi_notional = self.clearinghouse.open_interest_notional.get(symbol, 0.0)
            spec = self.dex_config.assets.get(symbol)
            oi_cap = spec.oi_cap_notional if spec else float('inf')

            ext_perp_px = external_perp_pxs.get(symbol)
            new_mark = me.update(oracle_px, deployer_marks, local_mark,
                                  self.currentTime, oi_notional, oi_cap,
                                  external_perp_px=ext_perp_px)

            if new_mark is not None:
                self.logEvent('MARK_PRICE_UPDATE', '{},{:.6f},{:.6f}'.format(symbol, new_mark, oracle_px))

                # Check trigger orders
                self._check_trigger_orders(symbol, new_mark)

                # Check liquidations
                self._check_liquidations()

    # ── Funding ─────────────────────────────────────────────────────────

    def _schedule_premium_sample(self, from_time):
        next_time = from_time + pd.Timedelta(self.premium_sample_interval_ns)
        self.kernel.messages.put((next_time, (self.id, MessageType.MESSAGE,
                                              Message({"msg": "_PREMIUM_SAMPLE"}))))

    def _schedule_funding(self, from_time):
        next_time = from_time + pd.Timedelta(self.funding_interval_ns)
        self.kernel.messages.put((next_time, (self.id, MessageType.MESSAGE,
                                              Message({"msg": "_FUNDING_SETTLE"}))))

    def _do_premium_sample(self, current_time):
        """Sample premium for all symbols."""
        for symbol, spec in self.dex_config.assets.items():
            me = self.mark_engines.get(symbol)
            ob = self.order_books.get(symbol)
            if me is None or ob is None:
                continue

            oracle_px = me.oracle_price
            impact_bid = ob.getImpactPrice(spec.funding_impact_notional, is_buy=False)
            impact_ask = ob.getImpactPrice(spec.funding_impact_notional, is_buy=True)

            # Fallback: use best bid/ask if impact price unavailable,
            # so thin books still produce a premium signal.
            if impact_bid is None:
                best_bid = ob.getBestBid()
                impact_bid = best_bid if best_bid is not None else oracle_px
            if impact_ask is None:
                best_ask = ob.getBestAsk()
                impact_ask = best_ask if best_ask is not None else oracle_px

            self.funding_engine.sample_premium(symbol, oracle_px, impact_bid, impact_ask)

            # Also check mark price staleness
            local_mark = ob.getLocalMarkPrice()
            me.check_staleness(current_time, local_mark)

    def _do_funding_settlement(self, current_time):
        """Settle funding for all symbols."""
        for symbol, spec in self.dex_config.assets.items():
            me = self.mark_engines.get(symbol)
            if me is None:
                continue

            hourly_rate = self.funding_engine.compute_hourly_rate(
                symbol, spec.funding_multiplier)

            if hourly_rate == 0:
                continue

            oracle_px = me.oracle_price

            # Gather all positions
            positions = {}
            for agent_id, account in self.clearinghouse.accounts.items():
                pos = account.get_position(symbol)
                if pos.size != 0:
                    positions[agent_id] = pos.size

            payments = self.funding_engine.compute_funding_payments(
                symbol, hourly_rate, oracle_px, positions)

            # Apply payments
            for agent_id, payment in payments.items():
                account = self.clearinghouse.get_account(agent_id)
                if account:
                    account.apply_funding(symbol, payment)

                # Notify the agent
                self.sendMessage(agent_id, Message({
                    "msg": "FUNDING_PAYMENT",
                    "symbol": symbol,
                    "payment": payment,
                    "rate": hourly_rate,
                    "oracle_price": oracle_px,
                }))

            self.logEvent('FUNDING_SETTLED', '{},{:.8f}'.format(symbol, hourly_rate))

    # ── Liquidation ─────────────────────────────────────────────────────

    def _check_liquidations(self):
        """Scan for liquidatable positions and process them."""
        mark_prices = {s: me.mark_price for s, me in self.mark_engines.items()}
        liquidatable = self.clearinghouse.get_liquidatable_accounts(mark_prices)

        if not liquidatable:
            return

        # Gather positions
        positions = {}
        for agent_id, account in self.clearinghouse.accounts.items():
            positions[agent_id] = {s: p.size for s, p in account.positions.items()}

        liq_orders = self.liquidation_engine.get_liquidation_orders(
            liquidatable, positions, mark_prices, self.currentTime)

        for liq in liq_orders:
            agent_id = liq['agent_id']
            symbol = liq['symbol']
            quantity = liq['quantity']
            is_buy = liq['is_buy']
            liq_type = liq['liq_type']

            if liq_type == 'backstop':
                # Transfer position to backstop vault
                account = self.clearinghouse.get_account(agent_id)
                if account:
                    pnl = account.settle_position(symbol, mark_prices[symbol])
                    self.sendMessage(agent_id, Message({
                        "msg": "LIQUIDATED",
                        "symbol": symbol,
                        "type": "backstop",
                        "settled_pnl": pnl,
                    }))
                    self.logEvent('BACKSTOP_LIQUIDATION', '{},{},{:.4f}'.format(agent_id, symbol, pnl))
            else:
                # Send market liquidation order to book
                liq_order = PerpLimitOrder(
                    agent_id, self.currentTime, symbol, quantity, is_buy,
                    limit_price=1e18 if is_buy else 0.0001,
                    tag="LIQUIDATION", time_in_force=TimeInForce.IOC,
                    is_liquidation=True,
                )
                self._handle_market_order(liq_order)

                self.sendMessage(agent_id, Message({
                    "msg": "LIQUIDATED",
                    "symbol": symbol,
                    "type": "market",
                    "quantity": quantity,
                }))
                self.logEvent('MARKET_LIQUIDATION', '{},{},{:.4f}'.format(agent_id, symbol, quantity))

    # ── Trigger orders ──────────────────────────────────────────────────

    def _check_trigger_orders(self, symbol, mark_price):
        """Check and activate trigger orders based on mark price."""
        to_remove = []
        for order_id, order in self.trigger_orders.items():
            if order.symbol != symbol or order.trigger_price is None:
                continue

            triggered = False
            if order.trigger_type in ('STOP_MARKET', 'STOP_LIMIT'):
                if order.is_buy_order and mark_price >= order.trigger_price:
                    triggered = True
                elif not order.is_buy_order and mark_price <= order.trigger_price:
                    triggered = True
            elif order.trigger_type in ('TAKE_MARKET', 'TAKE_LIMIT'):
                if order.is_buy_order and mark_price <= order.trigger_price:
                    triggered = True
                elif not order.is_buy_order and mark_price >= order.trigger_price:
                    triggered = True

            if triggered:
                to_remove.append(order_id)
                activated = deepcopy(order)
                activated.trigger_price = None
                activated.trigger_type = None
                if order.trigger_type in ('STOP_MARKET', 'TAKE_MARKET'):
                    self._handle_market_order(activated)
                else:
                    self._handle_limit_order(activated)

        for oid in to_remove:
            del self.trigger_orders[oid]

    # ── HIP-3 deployer controls ─────────────────────────────────────────

    def _handle_halt_trading(self, body):
        symbol = body['symbol']
        is_halted = body['is_halted']

        if is_halted:
            self.halted_symbols.add(symbol)
            # Cancel all resting orders
            if symbol in self.order_books:
                self.order_books[symbol].cancelAllOrders()
            # Settle all positions at mark
            me = self.mark_engines.get(symbol)
            if me:
                settlements = self.clearinghouse.settle_all(symbol, me.mark_price)
                for agent_id, pnl in settlements.items():
                    self.sendMessage(agent_id, Message({
                        "msg": "POSITION_SETTLED",
                        "symbol": symbol,
                        "settled_pnl": pnl,
                        "mark_price": me.mark_price,
                    }))
            self.logEvent('TRADING_HALTED', symbol)
        else:
            self.halted_symbols.discard(symbol)
            self.logEvent('TRADING_RESUMED', symbol)

    # ── Subscriptions ───────────────────────────────────────────────────

    def _update_subscription(self, msg, currentTime):
        if msg.body['msg'] == "MARKET_DATA_SUBSCRIPTION_REQUEST":
            agent_id = msg.body['sender']
            symbol = msg.body['symbol']
            levels = msg.body['levels']
            freq = msg.body['freq']
            self.subscription_dict[agent_id] = {symbol: [levels, freq, currentTime]}
        elif msg.body['msg'] == "MARKET_DATA_SUBSCRIPTION_CANCELLATION":
            agent_id = msg.body['sender']
            symbol = msg.body['symbol']
            if agent_id in self.subscription_dict and symbol in self.subscription_dict[agent_id]:
                del self.subscription_dict[agent_id][symbol]

    def _publish_order_book_data(self):
        for agent_id, params in self.subscription_dict.items():
            for symbol, values in params.items():
                levels, freq, last_update = values[0], values[1], values[2]
                ob = self.order_books.get(symbol)
                if ob is None:
                    continue
                ob_last = ob.last_update_ts
                if ob_last is None:
                    continue
                if freq == 0 or (ob_last > last_update and (ob_last - last_update).value >= freq):
                    me = self.mark_engines.get(symbol)
                    self.sendMessage(agent_id, Message({
                        "msg": "MARKET_DATA",
                        "symbol": symbol,
                        "bids": ob.getInsideBids(levels),
                        "asks": ob.getInsideAsks(levels),
                        "last_transaction": ob.last_trade,
                        "mark_price": me.mark_price if me else None,
                        "oracle_price": me.oracle_price if me else None,
                        "exchange_ts": self.currentTime,
                    }))
                    self.subscription_dict[agent_id][symbol][2] = ob_last

    # ── Message send override ───────────────────────────────────────────

    def sendMessage(self, recipientID, msg):
        if msg.body['msg'] in ['ORDER_ACCEPTED', 'ORDER_CANCELLED', 'ORDER_EXECUTED']:
            super().sendMessage(recipientID, msg, delay=self.pipeline_delay)
            if self.log_orders:
                self.logEvent(msg.body['msg'], str(msg.body.get('order', '')))
        else:
            super().sendMessage(recipientID, msg)

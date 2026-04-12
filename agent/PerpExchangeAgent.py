"""HIP-3 perpetual futures exchange agent."""

from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

import pandas as pd

from agent.FinancialAgent import FinancialAgent
from message.Message import Message, MessageType
from util.Clearinghouse import Clearinghouse
from util.ContractSpec import DeployerPermission, MarginMode, MarginTable, MarginTier, MarginType, PerpDexConfig, TimeInForce
from util.FundingEngine import FundingEngine
from util.LiquidationEngine import LiquidationEngine
from util.MarkPriceEngine import MarkPriceEngine
from util.PerpOrderBook import PerpOrderBook
from util.UserLimitEngine import UserLimitEngine, UserLimits
from util.order.PerpLimitOrder import PerpLimitOrder


@dataclass
class PendingAction:
    category: str
    sender_id: int
    arrival_time: pd.Timestamp
    message_uniq: int
    payload: dict


@dataclass
class TriggerGroup:
    group_id: object
    child_order_ids: List[int]
    parent_order_id: Optional[int] = None


class PerpExchangeAgent(FinancialAgent):
    def __init__(
        self,
        id,
        name,
        type,
        dex_config: PerpDexConfig,
        pipeline_delay=40000,
        computation_delay=1,
        stream_history=10,
        log_orders=False,
        random_state=None,
        starting_balances: Optional[Dict[int, float]] = None,
        block_interval_ms: Optional[int] = None,
        execution_mode: Optional[str] = None,
        fee_model: Optional[str] = None,
        user_limits: Optional[UserLimits] = None,
        user_limits_enabled: bool = False,
        log_l1: bool = False,
        log_l2: bool = False,
        l2_depth: int = 20,
    ):
        super().__init__(id, name, type, random_state)
        self.reschedule = False
        self.dex_config = dex_config
        self.pipeline_delay = pipeline_delay
        self.computation_delay = computation_delay
        self.stream_history = stream_history
        self.log_orders = log_orders

        self.execution_mode = execution_mode or dex_config.execution_mode or "hypercore_blocked"
        self.block_interval_ms = int(block_interval_ms or dex_config.block_interval_ms or 100)
        self.fee_model = fee_model or dex_config.fee_model or "hyperliquid"
        self.starting_balances = dict(starting_balances or {})

        self.order_books: Dict[str, PerpOrderBook] = {}
        self.mark_engines: Dict[str, MarkPriceEngine] = {}
        self.clearinghouse = Clearinghouse(dex_config=dex_config, starting_balances=self.starting_balances)
        self.funding_engine = FundingEngine()
        self.liquidation_engine = LiquidationEngine()
        self.user_limit_engine = UserLimitEngine(limits=user_limits, enabled=user_limits_enabled)

        for symbol, spec in dex_config.assets.items():
            self.order_books[symbol] = PerpOrderBook(self, symbol, spec)
            self.mark_engines[symbol] = MarkPriceEngine(spec.initial_oracle_px)
            self.order_books[symbol].last_trade = spec.initial_oracle_px

        self.subscription_dict = {}
        self.trigger_orders: Dict[int, PerpLimitOrder] = {}
        self.dormant_trigger_orders: Dict[int, PerpLimitOrder] = {}
        self.trigger_groups: Dict[object, TriggerGroup] = {}
        self.child_orders_by_parent: Dict[int, List[int]] = defaultdict(list)
        self.twap_orders: Dict[object, dict] = {}
        self.scale_templates: Dict[object, dict] = {}

        self.halted_symbols = set()
        # Optional L1/L2 order book snapshot logging. Both default off —
        # when disabled, _record_book_snapshot returns on its first line
        # and no lists are allocated.
        self.log_l1 = log_l1
        self.log_l2 = log_l2
        self.l2_depth = l2_depth
        self.l1_log: Optional[List[dict]] = [] if log_l1 else None
        self.l2_log: Optional[List[dict]] = [] if log_l2 else None
        self.backstop_positions: Dict[str, List[dict]] = defaultdict(list)
        self.backstop_vault_balance: float = 0.0
        self.backstop_vault_positions: Dict[str, float] = defaultdict(float)
        self.deployer_permissions = dict(dex_config.deployer_permissions)
        self.sub_deployer_permissions = dict(dex_config.default_sub_deployer_permissions)

        self.pending_actions: List[PendingAction] = []
        self.next_block_time = None
        self.next_premium_sample_time = None
        self.next_funding_time = None
        self.last_funding_marks: Dict[str, float] = {}
        self.last_oracle_update_time: Dict[str, pd.Timestamp] = {}
        self._current_day = None
        self._liquidation_check_pending = False
        self.next_market_data_publish_time = None
        self.exchange_activity = {
            "accepted": 0,
            "rejected": 0,
            "executed": 0,
            "cancelled": 0,
            "trigger_activations": 0,
            "funding_settlements": 0,
            "liquidations": 0,
            "adl_events": 0,
        }
        self.exchange_activity_by_symbol = defaultdict(lambda: {
            "accepted": 0,
            "rejected": 0,
            "executed": 0,
            "cancelled": 0,
            "trigger_activations": 0,
            "funding_settlements": 0,
            "liquidations": 0,
            "adl_events": 0,
        })
        self.exchange_rejection_reasons = defaultdict(int)
        self.exchange_rejection_reasons_by_symbol = defaultdict(lambda: defaultdict(int))

    def kernelInitializing(self, kernel):
        super().kernelInitializing(kernel)
        self.oracle = self.kernel.oracle

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        for symbol, spec in self.dex_config.assets.items():
            try:
                daily_open = self.oracle.getDailyOpenPrice(symbol, startTime)
                if daily_open:
                    self.mark_engines[symbol].start_of_day_price = float(daily_open)
            except Exception:
                self.mark_engines[symbol].start_of_day_price = spec.initial_oracle_px

        self.next_block_time = self._next_aligned_time(startTime, pd.Timedelta(milliseconds=self.block_interval_ms))
        self.next_premium_sample_time = self._next_aligned_time(startTime, pd.Timedelta(seconds=5))
        self.next_funding_time = self._next_aligned_time(startTime, pd.Timedelta(hours=1))
        self._schedule_internal(self.next_block_time, "_PROCESS_BLOCK")
        self._schedule_internal(self.next_premium_sample_time, "_PREMIUM_SAMPLE")
        self._schedule_internal(self.next_funding_time, "_FUNDING_SETTLE")

    def kernelTerminating(self):
        super().kernelTerminating()
        if hasattr(self.oracle, "f_log"):
            for symbol, history in self.oracle.f_log.items():
                df_fund = pd.DataFrame(history)
                if not df_fund.empty:
                    df_fund.set_index("FundamentalTime", inplace=True)
                    self.writeLog(df_fund, filename=f"fundamental_{symbol}")

    def kernelStopping(self):
        super().kernelStopping()
        self.logEvent("PERP_EXCHANGE_ACTIVITY_SUMMARY", self._build_exchange_activity_summary(), True)
        self.logEvent("PERP_EXCHANGE_REJECTION_REASONS", {
            "global": dict(sorted(self.exchange_rejection_reasons.items())),
            "by_symbol": {
                symbol: dict(sorted(reasons.items()))
                for symbol, reasons in sorted(self.exchange_rejection_reasons_by_symbol.items())
            },
        }, True)
        self.logEvent("PERP_EXCHANGE_OPEN_ORDERS_AT_STOP", self._open_orders_at_stop(), True)
        if self.log_l1:
            self._write_l1_csv()
        if self.log_l2:
            self._write_l2_csv()
        print("Exchange activity: {}".format(self._format_exchange_activity_summary()))

    def _record_book_snapshot(self, symbol: str) -> None:
        """Record an L1/L2 snapshot of `symbol`'s book if logging is enabled.

        Fast path: when both flags are off this returns on the first line
        with no allocation. Halted symbols are skipped so administrative
        cleanup during a halt doesn't pollute the microstructure stream.
        """
        if not (self.log_l1 or self.log_l2):
            return
        if symbol in self.halted_symbols:
            return
        ob = self.order_books.get(symbol)
        if ob is None:
            return
        ts = self.currentTime
        if self.log_l1:
            row = ob.l1_snapshot()
            row["Timestamp"] = ts
            row["Symbol"] = symbol
            self.l1_log.append(row)
        if self.log_l2:
            row = ob.snapshot(depth=self.l2_depth)
            row["Timestamp"] = ts
            row["Symbol"] = symbol
            self.l2_log.append(row)

    def _write_l1_csv(self) -> None:
        log_dir = self.kernel._run_log_path()
        os.makedirs(log_dir, exist_ok=True)
        cols = ["Timestamp", "Symbol", "BestBid", "BestAsk", "Mid", "LastTrade", "BestBidQty", "BestAskQty"]
        df = pd.DataFrame(self.l1_log or [], columns=cols)
        df.set_index("Timestamp", inplace=True)
        df.to_csv(os.path.join(log_dir, "L1.csv"))

    def _write_l2_csv(self) -> None:
        log_dir = self.kernel._run_log_path()
        os.makedirs(log_dir, exist_ok=True)
        cols = ["Timestamp", "Symbol"]
        for i in range(1, self.l2_depth + 1):
            cols.append(f"BidPx_{i}")
            cols.append(f"BidQty_{i}")
        for i in range(1, self.l2_depth + 1):
            cols.append(f"AskPx_{i}")
            cols.append(f"AskQty_{i}")
        df = pd.DataFrame(self.l2_log or [], columns=cols)
        df.set_index("Timestamp", inplace=True)
        df.to_csv(os.path.join(log_dir, "L2.csv"))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        self.setComputationDelay(self.computation_delay)
        self.clearinghouse.sync_time(currentTime)

        body = msg.body or {}
        msg_type = body.get("msg")
        if msg_type is None:
            return

        if msg_type in ["MARKET_DATA_SUBSCRIPTION_REQUEST", "MARKET_DATA_SUBSCRIPTION_CANCELLATION"]:
            self._update_subscription(msg, currentTime)
            return

        if msg_type in {"WHEN_MKT_OPEN", "WHEN_MKT_CLOSE"}:
            self._handle_market_hours_query(msg_type, body)
            return

        if msg_type.startswith("QUERY_"):
            self._handle_query(body)
            return

        if msg_type == "_PROCESS_BLOCK":
            self._process_block(currentTime)
            return

        if msg_type == "_PREMIUM_SAMPLE":
            self._do_premium_sample(currentTime)
            self.next_premium_sample_time = currentTime + pd.Timedelta(seconds=5)
            self._schedule_internal(self.next_premium_sample_time, "_PREMIUM_SAMPLE")
            return

        if msg_type == "_FUNDING_SETTLE":
            self._do_funding_settlement(currentTime)
            self.next_funding_time = self._next_aligned_time(currentTime + pd.Timedelta(nanoseconds=1), pd.Timedelta(hours=1))
            self._schedule_internal(self.next_funding_time, "_FUNDING_SETTLE")
            return

        if self._should_queue(body):
            self._queue_action(currentTime, msg, body)
            return

        self._apply_immediate_action(body, msg)

    def sendMessage(self, recipientID, msg):
        if msg.body["msg"] in ["ORDER_ACCEPTED", "ORDER_CANCELLED", "ORDER_EXECUTED", "ORDER_REJECTED", "ORDER_MODIFIED"]:
            self._record_exchange_message_activity(msg.body)
            super().sendMessage(recipientID, msg, delay=self.pipeline_delay)
            if self.log_orders and "order" in msg.body:
                self.logEvent(msg.body["msg"], str(msg.body["order"]))
            return
        super().sendMessage(recipientID, msg)

    def _schedule_internal(self, when: pd.Timestamp, msg_type: str, payload: Optional[dict] = None):
        body = {"msg": msg_type}
        if payload:
            body.update(payload)
        self.kernel.messages.put((when, (self.id, MessageType.MESSAGE, Message(body))))

    def _clone_action_payload(self, value):
        if isinstance(value, PerpLimitOrder):
            return value.clone()
        if isinstance(value, list):
            return [self._clone_action_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_action_payload(item) for item in value)
        if isinstance(value, dict):
            return {key: self._clone_action_payload(item) for key, item in value.items()}
        return value

    def _latest_market_data_update(self, symbol: str) -> Optional[pd.Timestamp]:
        order_book = self.order_books[symbol]
        engine = self.mark_engines[symbol]
        timestamps = [ts for ts in [order_book.last_update_ts, engine.last_update_time] if ts is not None]
        if not timestamps:
            return None
        return max(timestamps)

    def _recompute_next_market_data_publish_time(self):
        next_publish_time = None
        for params in self.subscription_dict.values():
            for _, _, _, next_due in params.values():
                if next_due is None:
                    continue
                if next_publish_time is None or next_due < next_publish_time:
                    next_publish_time = next_due
        self.next_market_data_publish_time = next_publish_time

    def _increment_exchange_activity(self, key: str, symbol: Optional[str] = None, amount: int = 1):
        self.exchange_activity[key] += amount
        if symbol is not None:
            self.exchange_activity_by_symbol[symbol][key] += amount

    def _record_exchange_message_activity(self, body: dict):
        msg_type = body["msg"]
        order = body.get("order")
        symbol = getattr(order, "symbol", None)
        if msg_type == "ORDER_ACCEPTED":
            self._increment_exchange_activity("accepted", symbol)
        elif msg_type == "ORDER_CANCELLED":
            self._increment_exchange_activity("cancelled", symbol)
        elif msg_type == "ORDER_EXECUTED":
            self._increment_exchange_activity("executed", symbol)
        elif msg_type == "ORDER_REJECTED":
            self._increment_exchange_activity("rejected", symbol)
            reason = body.get("reason", "")
            self.exchange_rejection_reasons[reason] += 1
            if symbol is not None:
                self.exchange_rejection_reasons_by_symbol[symbol][reason] += 1

    def _open_orders_at_stop(self):
        open_orders = {}
        for symbol, order_book in self.order_books.items():
            book_count = len(order_book.order_index)
            trigger_count = sum(1 for order in self.trigger_orders.values() if order.symbol == symbol)
            dormant_count = sum(1 for order in self.dormant_trigger_orders.values() if order.symbol == symbol)
            if book_count or trigger_count or dormant_count:
                open_orders[symbol] = {
                    "resting": book_count,
                    "trigger": trigger_count,
                    "dormant_trigger": dormant_count,
                }
        return open_orders

    def _build_exchange_activity_summary(self):
        summary = {"global": dict(self.exchange_activity), "by_symbol": {}}
        for symbol in sorted(self.order_books.keys() | self.exchange_activity_by_symbol.keys()):
            summary["by_symbol"][symbol] = dict(self.exchange_activity_by_symbol[symbol])
        summary["open_orders_at_stop"] = self._open_orders_at_stop()
        return summary

    def _format_exchange_activity_summary(self):
        symbol_parts = []
        for symbol in sorted(self.exchange_activity_by_symbol):
            payload = self.exchange_activity_by_symbol[symbol]
            symbol_parts.append(
                "{}: accepted={}, rejected={}, executed={}, cancelled={}, trigger_activations={}, funding_settlements={}, liquidations={}, adl={}".format(
                    symbol,
                    payload["accepted"],
                    payload["rejected"],
                    payload["executed"],
                    payload["cancelled"],
                    payload["trigger_activations"],
                    payload["funding_settlements"],
                    payload["liquidations"],
                    payload["adl_events"],
                )
            )
        return "global={}{}".format(
            self.exchange_activity,
            "; " + "; ".join(symbol_parts) if symbol_parts else "",
        )

    def _next_aligned_time(self, current_time: pd.Timestamp, delta: pd.Timedelta) -> pd.Timestamp:
        nanos = delta.value
        current_ns = pd.Timestamp(current_time).value
        aligned_ns = ((current_ns + nanos - 1) // nanos) * nanos
        return pd.Timestamp(aligned_ns)

    def _should_queue(self, body: dict) -> bool:
        msg_type = body["msg"]
        if self.execution_mode != "hypercore_blocked":
            return False
        if msg_type.startswith("QUERY_") or msg_type in {"WHEN_MKT_OPEN", "WHEN_MKT_CLOSE"}:
            return False
        if msg_type in {"SET_ORACLE", "HALT_TRADING", "SET_SUB_DEPLOYERS", "_ISOLATED_MARGIN_ADJUST", "_LEVERAGE_UPDATE", "_COLLATERAL_TRANSFER"}:
            return True
        if "ORDER" in msg_type or msg_type.endswith("_BATCH"):
            return True
        return msg_type in {
            "SET_OI_CAPS",
            "SET_FUNDING_MULTIPLIERS",
            "SET_FUNDING_INTEREST_RATES",
            "SET_MARGIN_TABLE",
            "SET_MARGIN_TABLE_IDS",
            "INSERT_MARGIN_TABLE",
            "SET_FEE_SCALE",
            "SET_GROWTH_MODES",
            "SET_PERP_ANNOTATION",
            "SET_FEE_RECIPIENT",
            "SET_MARGIN_MODES",
        }

    def _queue_action(self, current_time: pd.Timestamp, msg: Message, body: dict):
        category = self._categorize_action(body)
        self.pending_actions.append(
            PendingAction(
                category=category,
                sender_id=body.get("sender", -1),
                arrival_time=current_time,
                message_uniq=msg.uniq,
                payload=self._clone_action_payload(body),
            )
        )

    def _categorize_action(self, body: dict) -> str:
        msg_type = body["msg"]
        if msg_type in {
            "SET_ORACLE",
            "HALT_TRADING",
            "SET_OI_CAPS",
            "SET_FUNDING_MULTIPLIERS",
            "SET_FUNDING_INTEREST_RATES",
            "SET_MARGIN_TABLE",
            "SET_MARGIN_TABLE_IDS",
            "INSERT_MARGIN_TABLE",
            "SET_FEE_SCALE",
            "SET_GROWTH_MODES",
            "SET_SUB_DEPLOYERS",
            "SET_PERP_ANNOTATION",
            "SET_FEE_RECIPIENT",
            "SET_MARGIN_MODES",
            "_ISOLATED_MARGIN_ADJUST",
            "_LEVERAGE_UPDATE",
            "_COLLATERAL_TRANSFER",
        }:
            return "non_order"
        if msg_type in {"CANCEL_ORDER", "CANCEL_BATCH"}:
            return "cancel"
        if msg_type == "MODIFY_ORDER":
            new_order = body.get("new_order")
            if new_order and new_order.time_in_force in {TimeInForce.GTC, TimeInForce.IOC}:
                return "place"
            if new_order and new_order.time_in_force == TimeInForce.ALO:
                return "alo_place"
            return "non_order"
        if msg_type == "MODIFY_BATCH":
            has_placeable = False
            all_alo = True
            for update in body.get("updates", []):
                new_order = update.get("new_order")
                if new_order and new_order.time_in_force in {TimeInForce.GTC, TimeInForce.IOC}:
                    has_placeable = True
                    all_alo = False
                elif new_order and new_order.time_in_force == TimeInForce.ALO:
                    has_placeable = True
            if has_placeable:
                return "alo_place" if all_alo else "place"
            return "non_order"
        if msg_type == "LIMIT_ORDER":
            order = body.get("order")
            if order and order.time_in_force == TimeInForce.ALO:
                return "alo_place"
        if msg_type == "PLACE_BATCH":
            orders = body.get("orders", [])
            if orders and all(o.time_in_force == TimeInForce.ALO for o in orders):
                return "alo_place"
        return "place"

    def _handle_market_hours_query(self, msg_type: str, body: dict):
        self.setComputationDelay(0)
        response = self.kernel.startTime if msg_type == "WHEN_MKT_OPEN" else self.kernel.stopTime
        self.sendMessage(body["sender"], Message({"msg": msg_type, "data": response}))

    def _handle_query(self, body: dict):
        msg_type = body["msg"]
        sender = body["sender"]
        if msg_type == "QUERY_LAST_TRADE":
            symbol = body["symbol"]
            self.sendMessage(
                sender,
                Message(
                    {
                        "msg": "QUERY_LAST_TRADE",
                        "symbol": symbol,
                        "data": self.order_books[symbol].last_trade,
                        "mkt_closed": False,
                    }
                ),
            )
            return

        if msg_type == "QUERY_SPREAD":
            symbol = body["symbol"]
            depth = body["depth"]
            order_book = self.order_books[symbol]
            self.sendMessage(
                sender,
                Message(
                    {
                        "msg": "QUERY_SPREAD",
                        "symbol": symbol,
                        "depth": depth,
                        "bids": order_book.getInsideBids(depth),
                        "asks": order_book.getInsideAsks(depth),
                        "data": order_book.last_trade,
                        "mkt_closed": False,
                        "book": "",
                    }
                ),
            )
            return

        if msg_type == "QUERY_ORDER_STREAM":
            symbol = body["symbol"]
            length = body["length"]
            self.sendMessage(
                sender,
                Message(
                    {
                        "msg": "QUERY_ORDER_STREAM",
                        "symbol": symbol,
                        "length": length,
                        "mkt_closed": False,
                        "orders": self.order_books[symbol].history[1 : length + 1],
                    }
                ),
            )
            return

        if msg_type == "QUERY_TRANSACTED_VOLUME":
            symbol = body["symbol"]
            lookback = body.get("lookback_period", "10min")
            self.sendMessage(
                sender,
                Message(
                    {
                        "msg": "QUERY_TRANSACTED_VOLUME",
                        "symbol": symbol,
                        "transacted_volume": self.order_books[symbol].get_transacted_volume(lookback),
                        "mkt_closed": False,
                    }
                ),
            )
            return

        if msg_type == "QUERY_MARK_PRICE":
            symbol = body["symbol"]
            engine = self.mark_engines.get(symbol)
            self.sendMessage(
                sender,
                Message(
                    {
                        "msg": "QUERY_MARK_PRICE",
                        "symbol": symbol,
                        "mark_price": engine.mark_price if engine else None,
                        "oracle_price": engine.oracle_price if engine else None,
                    }
                ),
            )
            return

        if msg_type == "QUERY_FUNDING_RATE":
            symbol = body["symbol"]
            self.sendMessage(
                sender,
                Message({"msg": "QUERY_FUNDING_RATE", "symbol": symbol, "funding_rate": self.funding_engine.get_last_rate(symbol)}),
            )
            return

        if msg_type == "QUERY_POSITION":
            account = self.clearinghouse.get_account(sender)
            positions = {}
            if account:
                for symbol, position in account.positions.items():
                    positions[symbol] = {
                        "size": position.size,
                        "entry_price": position.entry_price,
                        "leverage": position.leverage,
                        "margin_type": position.margin_type.value,
                        "isolated_margin": position.isolated_margin,
                        "unrealized_pnl": position.unrealized_pnl(
                            self.mark_engines[symbol].mark_price if symbol in self.mark_engines else position.entry_price
                        ),
                    }
            self.sendMessage(
                sender,
                Message(
                    {
                        "msg": "QUERY_POSITION",
                        "positions": positions,
                        "balance": account.balance if account else 0.0,
                        "holds": {
                            hold.order_id: {
                                "symbol": hold.symbol,
                                "cross_reserved": hold.cross_reserved,
                                "isolated_reserved": hold.isolated_reserved,
                                "remaining_qty": hold.remaining_qty,
                            }
                            for hold in (account.order_holds.values() if account else [])
                        },
                    }
                ),
            )
            return

    def _apply_immediate_action(self, body: dict, msg: Message):
        msg_type = body["msg"]
        if msg_type == "SET_SUB_DEPLOYERS":
            self._handle_set_sub_deployers(body)
            return
        if msg_type == "_COLLATERAL_TRANSFER":
            self._handle_collateral_transfer(body)
            return
        if msg_type == "_ISOLATED_MARGIN_ADJUST":
            self._handle_isolated_margin_adjustment(body)
            return
        if msg_type == "_LEVERAGE_UPDATE":
            self._handle_leverage_update(body)
            return
        self._execute_action(body, msg.uniq)

    def _process_block(self, current_time: pd.Timestamp):
        # Daily start-of-day reset for mark price engines (UTC day boundary)
        current_day = current_time.normalize()
        if self._current_day is None:
            self._current_day = current_day
        elif current_day > self._current_day:
            self._current_day = current_day
            for symbol, engine in self.mark_engines.items():
                try:
                    daily_open = self.oracle.getDailyOpenPrice(symbol, current_time)
                    engine.reset_start_of_day(float(daily_open) if daily_open else None)
                except Exception:
                    engine.reset_start_of_day()

        if self.pending_actions:
            order = {"non_order": 0, "cancel": 1, "alo_place": 2, "place": 3}
            for action in sorted(self.pending_actions, key=lambda item: (order[item.category], item.arrival_time, item.message_uniq)):
                self._execute_action(action.payload, action.message_uniq)
            self.pending_actions = []

        self._expire_orders(current_time)
        self._check_trigger_orders()
        if self._liquidation_check_pending:
            self._check_liquidations()
        self._process_twap_slices(current_time)
        self._publish_order_book_data()
        self.next_block_time = current_time + pd.Timedelta(milliseconds=self.block_interval_ms)
        self._schedule_internal(self.next_block_time, "_PROCESS_BLOCK")

    def _execute_action(self, body: dict, message_uniq: int):
        msg_type = body["msg"]
        sender_id = body.get("sender")

        # User limit enforcement for order actions
        if self.user_limit_engine.enabled and sender_id is not None:
            if msg_type in {"LIMIT_ORDER", "MARKET_ORDER", "TRIGGER_ORDER", "MODIFY_ORDER", "PLACE_BATCH", "MODIFY_BATCH", "TWAP_ORDER", "SCALE_ORDER"}:
                reject_reason = self.user_limit_engine.check_action(sender_id, self.currentTime)
                if reject_reason:
                    order = body.get("order") or (body.get("orders", [None])[0] if body.get("orders") else None)
                    if order is not None:
                        self._reject_order(order, reject_reason)
                    return
                self.user_limit_engine.record_action(sender_id, self.currentTime)
            if msg_type in {"CANCEL_ORDER", "CANCEL_BATCH"}:
                reject_reason = self.user_limit_engine.check_cancel(sender_id, self.currentTime)
                if reject_reason:
                    return
                self.user_limit_engine.record_cancel(sender_id, self.currentTime)
                self.user_limit_engine.record_action(sender_id, self.currentTime)

        if msg_type == "LIMIT_ORDER":
            self._handle_limit_order(body["order"])
            return
        if msg_type == "MARKET_ORDER":
            self._handle_market_order(body["order"])
            return
        if msg_type == "TRIGGER_ORDER":
            self._handle_trigger_order(body["order"])
            return
        if msg_type == "CANCEL_ORDER":
            self._handle_cancel_order(body["order"])
            return
        if msg_type == "MODIFY_ORDER":
            self._handle_modify_order(body["order"], body["new_order"])
            return
        if msg_type == "PLACE_BATCH":
            for order in body.get("orders", []):
                if getattr(order, "trigger_price", None) is not None:
                    self._handle_trigger_order(order)
                elif getattr(order, "is_market_order", False):
                    self._handle_market_order(order)
                else:
                    self._handle_limit_order(order)
            return
        if msg_type == "CANCEL_BATCH":
            for order in body.get("orders", []):
                self._handle_cancel_order(order)
            return
        if msg_type == "MODIFY_BATCH":
            for update in body.get("updates", []):
                self._handle_modify_order(update["order"], update["new_order"])
            return
        if msg_type == "TWAP_ORDER":
            self._handle_twap_order(body)
            return
        if msg_type == "CANCEL_TWAP":
            self._handle_cancel_twap(body)
            return
        if msg_type == "SCALE_ORDER":
            self._handle_scale_order(body)
            return
        if msg_type == "SET_ORACLE":
            self._handle_set_oracle(body)
            return
        if msg_type == "HALT_TRADING":
            self._handle_halt_trading(body)
            return
        if msg_type == "SET_OI_CAPS":
            self._handle_set_oi_caps(body)
            return
        if msg_type == "SET_FUNDING_MULTIPLIERS":
            for symbol, multiplier in body["multipliers"].items():
                self.clearinghouse.update_funding_multiplier(symbol, multiplier)
            return
        if msg_type == "SET_FUNDING_INTEREST_RATES":
            for symbol, rate in body["rates"].items():
                self.clearinghouse.update_funding_interest_rate(symbol, rate)
            return
        if msg_type == "SET_MARGIN_TABLE":
            self._handle_set_margin_table(body)
            self._liquidation_check_pending = True
            return
        if msg_type == "INSERT_MARGIN_TABLE":
            self._handle_insert_margin_table(body)
            return
        if msg_type == "SET_MARGIN_TABLE_IDS":
            for symbol, margin_table_id in body["margin_table_ids"].items():
                self.clearinghouse.update_margin_table_id(symbol, margin_table_id)
            self._liquidation_check_pending = True
            return
        if msg_type == "SET_FEE_SCALE":
            self.clearinghouse.update_fee_scale(body["fee_scale"])
            return
        if msg_type == "SET_GROWTH_MODES":
            for symbol, enabled in body["growth_modes"].items():
                self.clearinghouse.update_growth_mode(symbol, enabled)
            return
        if msg_type == "SET_PERP_ANNOTATION":
            symbol = body["symbol"]
            self.dex_config.perp_annotations[symbol] = deepcopy(body["annotation"])
            return
        if msg_type == "SET_FEE_RECIPIENT":
            self._handle_set_fee_recipient(body)
            return
        if msg_type == "SET_MARGIN_MODES":
            self._handle_set_margin_modes(body)
            return

    def _get_mark_prices(self) -> Dict[str, float]:
        return {symbol: engine.mark_price for symbol, engine in self.mark_engines.items()}

    def _validate_order(self, order: PerpLimitOrder, is_market: bool = False) -> Tuple[bool, Optional[str]]:
        spec = self.dex_config.assets.get(order.symbol)
        if spec is None:
            return False, "UNKNOWN_SYMBOL"

        order.quantity = spec.round_size(order.quantity)
        if order.quantity <= 0:
            return False, "ZERO_QUANTITY"

        if is_market:
            reference_price = self.mark_engines[order.symbol].mark_price
        else:
            if not spec.is_valid_price(order.limit_price):
                return False, "INVALID_PRICE"
            reference_price = order.limit_price

        notional = order.quantity * reference_price
        if notional < spec.min_order_value:
            return False, "MIN_ORDER_VALUE"

        max_value = spec.max_market_order_value if is_market else spec.max_limit_order_value
        if notional > max_value:
            return False, "MAX_ORDER_VALUE"

        return True, None

    def _oi_restricted_price_distance(self, symbol: str, price: float) -> bool:
        spec = self.dex_config.assets[symbol]
        mark_price = self.mark_engines[symbol].oracle_price
        if mark_price <= 0:
            return False
        at_cap = (
            self.clearinghouse.open_interest_size.get(symbol, 0.0) >= spec.oi_cap_size - 1e-12
            or self.clearinghouse.open_interest_notional.get(symbol, 0.0) >= spec.oi_cap_notional - 1e-12
            or self.clearinghouse.total_dex_open_interest_notional >= self.dex_config.dex_open_interest_cap_notional - 1e-12
        )
        if not at_cap:
            return False
        return abs(price - mark_price) / mark_price > 0.01

    def _risk_check(self, order: PerpLimitOrder) -> Tuple[bool, Optional[str]]:
        account = self.clearinghouse.get_or_create_account(order.agent_id)
        spec = self.dex_config.assets[order.symbol]
        mark_prices = self._get_mark_prices()

        if order.reduce_only:
            position = account.get_position(order.symbol)
            if position.size == 0:
                return False, "REDUCE_ONLY_NO_POSITION"
            if (position.size > 0 and order.is_buy_order) or (position.size < 0 and not order.is_buy_order):
                return False, "REDUCE_ONLY_DIRECTION"
            return True, None

        opening_qty = account.estimate_opening_qty(order.symbol, order.quantity, order.is_buy_order)
        if opening_qty > 0:
            reference_price = mark_prices.get(order.symbol, spec.initial_oracle_px) if order.is_market_order else order.limit_price
            if not self.clearinghouse.check_oi_cap(order.symbol, opening_qty, reference_price):
                return False, "OI_CAP"
            if account.total_equity(mark_prices) <= 1e-12 and not account.has_position(order.symbol):
                return False, "UNFUNDED_ACCOUNT"
            if not order.is_market_order and self._oi_restricted_price_distance(order.symbol, order.limit_price):
                return False, "OI_CAP_PRICE_BAND"

        ok, reason, _preview = self.clearinghouse.place_hold(order.agent_id, order, mark_prices)
        if not ok:
            return False, reason
        return True, None

    def _handle_limit_order(self, order: PerpLimitOrder):
        if order.symbol in self.halted_symbols:
            self._reject_order(order, "HALTED")
            return

        ok, reason = self._validate_order(order, is_market=False)
        if not ok:
            self._reject_order(order, reason)
            return

        if order.time_in_force == TimeInForce.ALO and self.order_books[order.symbol].would_match(order):
            self._reject_order(order, "ALO_WOULD_TAKE")
            return

        cap_reason = self.user_limit_engine.check_open_orders(order.agent_id)
        if cap_reason:
            self._reject_order(order, cap_reason)
            return

        ok, reason = self._risk_check(order)
        if not ok:
            self._reject_order(order, reason)
            return

        self.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))
        self._execute_live_order(order)

    def _handle_market_order(self, order: PerpLimitOrder):
        order.is_market_order = True
        order.time_in_force = TimeInForce.IOC
        if order.symbol in self.halted_symbols:
            self._reject_order(order, "HALTED")
            return

        ok, reason = self._validate_order(order, is_market=True)
        if not ok:
            self._reject_order(order, reason)
            return

        ok, reason = self._risk_check(order)
        if not ok:
            self._reject_order(order, reason)
            return

        self.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))
        self._execute_live_order(order)

    def _handle_trigger_order(self, order: PerpLimitOrder):
        if order.symbol in self.halted_symbols:
            self._reject_order(order, "HALTED")
            return
        if order.trigger_price is None or order.trigger_type is None:
            self._reject_order(order, "MISSING_TRIGGER")
            return
        stored_order = order.clone()
        is_dormant_child = stored_order.parent_order_id is not None and stored_order.tpsl_mode == "parent"
        if is_dormant_child:
            self.dormant_trigger_orders[stored_order.order_id] = stored_order
        else:
            self.trigger_orders[stored_order.order_id] = stored_order
        if order.tpsl_group_id is not None:
            group = self.trigger_groups.setdefault(order.tpsl_group_id, TriggerGroup(group_id=order.tpsl_group_id, child_order_ids=[]))
            if order.order_id not in group.child_order_ids:
                group.child_order_ids.append(order.order_id)
            if order.parent_order_id is not None:
                group.parent_order_id = order.parent_order_id
                self.child_orders_by_parent[order.parent_order_id].append(order.order_id)
        self.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))

    def _handle_cancel_order(self, order: PerpLimitOrder):
        if order.order_id in self.trigger_orders or order.order_id in self.dormant_trigger_orders:
            cancelled = self.trigger_orders.pop(order.order_id, None) or self.dormant_trigger_orders.pop(order.order_id, None)
            self.clearinghouse.release_hold(cancelled.agent_id, cancelled.order_id)
            self._cancel_linked_trigger_orders(cancelled)
            self.sendMessage(cancelled.agent_id, Message({"msg": "ORDER_CANCELLED", "order": cancelled}))
            return

        cancelled = self.order_books[order.symbol].cancel_order(order.order_id)
        if cancelled is None:
            return
        self._record_book_snapshot(cancelled.symbol)
        self.clearinghouse.release_hold(cancelled.agent_id, cancelled.order_id)
        self.user_limit_engine.record_order_removed(cancelled.agent_id)
        self._cancel_children_for_parent(cancelled.order_id, reason="PARENT_CANCELLED")
        self.sendMessage(cancelled.agent_id, Message({"msg": "ORDER_CANCELLED", "order": cancelled}))

    def _handle_modify_order(self, order: PerpLimitOrder, new_order: PerpLimitOrder):
        if order.symbol in self.halted_symbols:
            self._reject_order(new_order, "HALTED")
            return
        if order.order_id not in self.order_books[order.symbol].order_index:
            self._reject_order(new_order, "UNKNOWN_ORDER")
            return

        if (
            new_order.agent_id != order.agent_id
            or new_order.symbol != order.symbol
            or new_order.is_buy_order != order.is_buy_order
        ):
            self._reject_order(new_order, "INVALID_MODIFY")
            return

        modified_order = new_order.clone()
        modified_order.order_id = order.order_id

        ok, reason = self._validate_order(modified_order, is_market=modified_order.is_market_order)
        if not ok:
            self._reject_order(modified_order, reason)
            return

        if modified_order.time_in_force == TimeInForce.ALO and self.order_books[modified_order.symbol].would_match(modified_order):
            self._reject_order(modified_order, "ALO_WOULD_TAKE")
            return

        mark_prices = self._get_mark_prices()
        account = self.clearinghouse.get_or_create_account(modified_order.agent_id)
        spec = self.dex_config.assets[modified_order.symbol]

        if modified_order.reduce_only:
            position = account.get_position(modified_order.symbol)
            if position.size == 0:
                self._reject_order(modified_order, "REDUCE_ONLY_NO_POSITION")
                return
            if (position.size > 0 and modified_order.is_buy_order) or (position.size < 0 and not modified_order.is_buy_order):
                self._reject_order(modified_order, "REDUCE_ONLY_DIRECTION")
                return
        else:
            opening_qty = account.estimate_opening_qty(modified_order.symbol, modified_order.quantity, modified_order.is_buy_order)
            if opening_qty > 0:
                reference_price = (
                    mark_prices.get(modified_order.symbol, spec.initial_oracle_px)
                    if modified_order.is_market_order
                    else modified_order.limit_price
                )
                if not self.clearinghouse.check_oi_cap(modified_order.symbol, opening_qty, reference_price):
                    self._reject_order(modified_order, "OI_CAP")
                    return
                if account.total_equity(mark_prices) <= 1e-12 and not account.has_position(modified_order.symbol):
                    self._reject_order(modified_order, "UNFUNDED_ACCOUNT")
                    return
                if not modified_order.is_market_order and self._oi_restricted_price_distance(modified_order.symbol, modified_order.limit_price):
                    self._reject_order(modified_order, "OI_CAP_PRICE_BAND")
                    return

        # Determine if this is a same-price, size-only amend (queue-preserving)
        old_resting = self.order_books[order.symbol].order_index.get(order.order_id)
        is_same_price = old_resting is not None and abs(old_resting.limit_price - modified_order.limit_price) < 1e-12
        is_size_only = is_same_price and old_resting.is_buy_order == modified_order.is_buy_order

        if is_size_only:
            # Queue-preserving amend: resize hold and update quantity in place
            ok, reason = self.clearinghouse.resize_hold_incremental(
                modified_order.agent_id,
                modified_order.order_id,
                modified_order.quantity,
                modified_order.limit_price,
                mark_prices,
            )
            if not ok:
                self._reject_order(modified_order, reason)
                return
            self.order_books[order.symbol].amend_order_in_place(order.order_id, modified_order.quantity)
            self._record_book_snapshot(modified_order.symbol)
            self.sendMessage(modified_order.agent_id, Message({"msg": "ORDER_MODIFIED", "order": modified_order}))
        else:
            # Price-changing or marketable modify: cancel and re-enter (loses queue)
            ok, reason, _preview = self.clearinghouse.replace_hold(modified_order.agent_id, modified_order, mark_prices)
            if not ok:
                self._reject_order(modified_order, reason)
                return

            cancelled = self.order_books[order.symbol].cancel_order(order.order_id)
            if cancelled is None:
                self._reject_order(modified_order, "UNKNOWN_ORDER")
                return
            self._record_book_snapshot(cancelled.symbol)

            self.sendMessage(modified_order.agent_id, Message({"msg": "ORDER_MODIFIED", "order": modified_order}))
            self._execute_live_order(modified_order)

    def _reject_order(self, order: PerpLimitOrder, reason: str):
        self.clearinghouse.release_hold(order.agent_id, order.order_id)
        self.sendMessage(order.agent_id, Message({"msg": "ORDER_REJECTED", "order": order, "reason": reason}))

    def _execute_live_order(self, order: PerpLimitOrder, already_on_book: bool = False):
        order_book = self.order_books[order.symbol]
        incoming = order.clone()

        if not already_on_book:
            if order.time_in_force == TimeInForce.GTC and not order.is_market_order and not order_book.would_match(order):
                order_book.enter_order(order.clone())
                self.user_limit_engine.record_order_placed(order.agent_id)
                self._record_book_snapshot(order.symbol)
                return

        fills, cancelled = order_book.match_order(incoming, maker_validator=lambda resting, qty: self._validate_maker_fill(resting, qty))

        if fills:
            self._process_fills(order.symbol, fills)

        if fills or cancelled:
            self._record_book_snapshot(order.symbol)

        for cancelled_order in cancelled:
            self.clearinghouse.release_hold(cancelled_order.agent_id, cancelled_order.order_id)
            self.user_limit_engine.record_order_removed(cancelled_order.agent_id)
            self.sendMessage(
                cancelled_order.agent_id,
                Message({"msg": "ORDER_CANCELLED", "order": cancelled_order, "reason": "SELF_TRADE_OR_MARGIN"}),
            )

        if incoming.quantity > 1e-12 and order.time_in_force == TimeInForce.GTC and not order.is_market_order:
            residual = order.clone()
            residual.quantity = incoming.quantity
            order_book.enter_order(residual)
            self._record_book_snapshot(order.symbol)
            self.user_limit_engine.record_order_placed(order.agent_id)
            mark_prices = self._get_mark_prices()
            self.clearinghouse.replace_hold(residual.agent_id, residual, mark_prices)
        else:
            self.clearinghouse.release_hold(order.agent_id, order.order_id)

    def _validate_maker_fill(self, resting_order: PerpLimitOrder, fill_qty: float) -> Tuple[bool, str]:
        mark_prices = self._get_mark_prices()
        if self.clearinghouse.ensure_fillable(resting_order, fill_qty, mark_prices):
            return True, ""
        return False, "INSUFFICIENT_MARGIN"

    def _process_fills(self, symbol: str, fills: List[Tuple[PerpLimitOrder, PerpLimitOrder]]):
        mark_price = self.mark_engines[symbol].mark_price
        for filled_order, matched_order in fills:
            taker_fee = self.clearinghouse.process_fill(
                agent_id=filled_order.agent_id,
                order=filled_order,
                fill_qty=filled_order.quantity,
                fill_price=filled_order.fill_price,
                is_taker=True,
                current_time=self.currentTime,
                fee_model=self.fee_model,
            )
            maker_fee = self.clearinghouse.process_fill(
                agent_id=matched_order.agent_id,
                order=matched_order,
                fill_qty=matched_order.quantity,
                fill_price=matched_order.fill_price,
                is_taker=False,
                current_time=self.currentTime,
                fee_model=self.fee_model,
            )
            # Builder fee: deduct from taker and credit to builder address
            builder_fee = 0.0
            if filled_order.builder_fee_bps is not None and filled_order.builder_address is not None:
                builder_fee = filled_order.quantity * filled_order.fill_price * filled_order.builder_fee_bps / 10_000
                taker_account = self.clearinghouse.get_or_create_account(filled_order.agent_id)
                taker_account.cash -= builder_fee
                builder_account = self.clearinghouse.get_or_create_account(filled_order.builder_address)
                builder_account.cash += builder_fee

            self.sendMessage(filled_order.agent_id, Message({"msg": "ORDER_EXECUTED", "order": filled_order, "fee": taker_fee, "builder_fee": builder_fee}))
            self.sendMessage(matched_order.agent_id, Message({"msg": "ORDER_EXECUTED", "order": matched_order, "fee": maker_fee}))
            # Track maker order removal from book on full fill
            if matched_order.order_id not in self.order_books[symbol].order_index:
                self.user_limit_engine.record_order_removed(matched_order.agent_id)
            self._activate_children_after_fill(matched_order)
            self._activate_children_after_fill(filled_order)

        self.clearinghouse.recalculate_symbol_oi(symbol, mark_price=mark_price)
        self._liquidation_check_pending = True

    def _validate_deployer(self, sender_id: int, action_name: str) -> bool:
        permission = self.deployer_permissions.get(sender_id) or self.sub_deployer_permissions.get(sender_id)
        return permission is not None and permission.allows(action_name)

    def _handle_set_oracle(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "SET_ORACLE"):
            return

        oracle_pxs = body.get("oracle_pxs", {})
        mark_pxs_data = body.get("mark_pxs", {})
        external_perp_pxs = body.get("external_perp_pxs", {})
        if self.dex_config.external_perp_px_mode != "none":
            if not set(self.dex_config.assets.keys()).issubset(set(external_perp_pxs.keys())):
                return

        updated_symbols = set()
        for symbol, oracle_px in oracle_pxs.items():
            if symbol not in self.mark_engines:
                continue
            last_update = self.last_oracle_update_time.get(symbol)
            if last_update is not None and (self.currentTime - last_update).value < 2_500_000_000:
                continue
            engine = self.mark_engines[symbol]
            order_book = self.order_books[symbol]
            local_mark = order_book.getLocalMarkPrice()
            deployer_marks = mark_pxs_data.get(symbol, []) if isinstance(mark_pxs_data, dict) else mark_pxs_data
            if external_perp_pxs and symbol not in external_perp_pxs:
                continue

            new_mark = engine.update(
                oracle_px=oracle_px,
                deployer_mark_pxs=deployer_marks,
                local_mark_px=local_mark,
                current_time=self.currentTime,
                oi_notional=self.clearinghouse.open_interest_notional.get(symbol, 0.0),
                oi_cap=self.dex_config.assets[symbol].oi_cap_notional,
                external_perp_px=external_perp_pxs.get(symbol),
            )
            if new_mark is None:
                continue
            self.last_oracle_update_time[symbol] = self.currentTime
            self.last_funding_marks[symbol] = new_mark
            self._check_trigger_orders(symbol=symbol)
            updated_symbols.add(symbol)

        for symbol in updated_symbols:
            self.clearinghouse.refresh_open_interest_notional(symbol, self.mark_engines[symbol].mark_price)

        if updated_symbols:
            self._liquidation_check_pending = True

    def _handle_halt_trading(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "HALT_TRADING"):
            return

        symbol = body["symbol"]
        is_halted = body["is_halted"]
        if is_halted:
            self.halted_symbols.add(symbol)
            for order in self.order_books[symbol].cancel_all_orders():
                self.clearinghouse.release_hold(order.agent_id, order.order_id)
                self._cancel_children_for_parent(order.order_id, reason="HALTED")
                self.sendMessage(order.agent_id, Message({"msg": "ORDER_CANCELLED", "order": order, "reason": "HALTED"}))
            for order_id, trigger_order in list(self.trigger_orders.items()):
                if trigger_order.symbol != symbol:
                    continue
                self.trigger_orders.pop(order_id, None)
                self.sendMessage(trigger_order.agent_id, Message({"msg": "ORDER_CANCELLED", "order": trigger_order, "reason": "HALTED"}))
            for order_id, trigger_order in list(self.dormant_trigger_orders.items()):
                if trigger_order.symbol != symbol:
                    continue
                self.dormant_trigger_orders.pop(order_id, None)
                self.sendMessage(trigger_order.agent_id, Message({"msg": "ORDER_CANCELLED", "order": trigger_order, "reason": "HALTED"}))
            settlements = self.clearinghouse.settle_all(symbol, self.mark_engines[symbol].mark_price)
            for agent_id, pnl in settlements.items():
                self.sendMessage(
                    agent_id,
                    Message(
                        {
                            "msg": "POSITION_SETTLED",
                            "symbol": symbol,
                            "settled_pnl": pnl,
                            "mark_price": self.mark_engines[symbol].mark_price,
                        }
                    ),
                )
        else:
            self.halted_symbols.discard(symbol)

    def _handle_set_oi_caps(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "SET_OI_CAPS"):
            return
        self.clearinghouse.update_oi_caps(body["symbol"], body["notional_cap"], body["size_cap"])
        if "dex_notional_cap" in body:
            self.dex_config.dex_open_interest_cap_notional = body["dex_notional_cap"]

    def _handle_set_margin_table(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "SET_MARGIN_TABLE"):
            return
        tiers = [MarginTier(float(t["lower_bound_notional"]), int(t["max_leverage"])) for t in body["tiers"]]
        margin_table = MarginTable(tiers=tiers, table_id=int(body.get("margin_table_id", 0)))
        self.clearinghouse.update_margin_table(body["symbol"], margin_table)

    def _handle_insert_margin_table(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "INSERT_MARGIN_TABLE"):
            return
        tiers = [MarginTier(float(t["lower_bound_notional"]), int(t["max_leverage"])) for t in body["tiers"]]
        margin_table = MarginTable(
            tiers=tiers,
            description=body.get("description", ""),
            table_id=int(body["margin_table_id"]),
        )
        self.clearinghouse.insert_margin_table(margin_table)

    def _handle_set_sub_deployers(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "SET_SUB_DEPLOYERS"):
            return
        self.sub_deployer_permissions = {}
        for agent_id_str, variants in body.get("permissions", {}).items():
            if isinstance(variants, DeployerPermission):
                self.sub_deployer_permissions[int(agent_id_str)] = deepcopy(variants)
            else:
                self.sub_deployer_permissions[int(agent_id_str)] = DeployerPermission(variants=list(variants))

    def _expire_orders(self, current_time: pd.Timestamp):
        """Cancel resting orders that have exceeded their expires_after_ms."""
        for symbol, order_book in self.order_books.items():
            expired = []
            for order_id, order in list(order_book.order_index.items()):
                if order.expires_after_ms is not None:
                    elapsed_ms = (current_time - order.time_placed).value / 1_000_000
                    if elapsed_ms >= order.expires_after_ms:
                        expired.append(order_id)
            for order_id in expired:
                cancelled = order_book.cancel_order(order_id)
                if cancelled is not None:
                    self._record_book_snapshot(symbol)
                    self.clearinghouse.release_hold(cancelled.agent_id, cancelled.order_id)
                    self.user_limit_engine.record_order_removed(cancelled.agent_id)
                    self.sendMessage(cancelled.agent_id, Message({"msg": "ORDER_CANCELLED", "order": cancelled, "reason": "EXPIRED"}))

    def _handle_set_fee_recipient(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "SET_FEE_RECIPIENT"):
            return
        symbol = body["symbol"]
        recipient = body["recipient"]
        if symbol not in self.dex_config.assets:
            return
        if not hasattr(self, "fee_recipients"):
            self.fee_recipients = {}
        self.fee_recipients[symbol] = recipient

    def _handle_set_margin_modes(self, body: dict):
        sender = body.get("sender")
        if self.deployer_permissions and not self._validate_deployer(sender, "SET_MARGIN_MODES"):
            return
        symbol = body["symbol"]
        margin_mode = body["margin_mode"]
        spec = self.dex_config.assets.get(symbol)
        if spec is None:
            return
        if isinstance(margin_mode, str):
            margin_mode = MarginMode(margin_mode)
        spec.margin_mode = margin_mode
        self._liquidation_check_pending = True

    def _do_premium_sample(self, current_time: pd.Timestamp):
        for symbol, spec in self.dex_config.assets.items():
            engine = self.mark_engines[symbol]
            order_book = self.order_books[symbol]
            oracle_price = engine.oracle_price
            impact_bid = order_book.getImpactPrice(spec.funding_impact_notional, is_buy=False)
            impact_ask = order_book.getImpactPrice(spec.funding_impact_notional, is_buy=True)

            if impact_bid is None:
                impact_bid = order_book.getBestBid() or oracle_price
            if impact_ask is None:
                impact_ask = order_book.getBestAsk() or oracle_price

            self.funding_engine.sample_premium(symbol, oracle_price, impact_bid, impact_ask)
            engine.check_staleness(current_time, order_book.getLocalMarkPrice())

    def _do_funding_settlement(self, current_time: pd.Timestamp):
        applied_payments = False
        for symbol, spec in self.dex_config.assets.items():
            oracle_price = self.mark_engines[symbol].oracle_price
            positions = {}
            for agent_id, account in self.clearinghouse.accounts.items():
                size = account.get_position(symbol).size
                if size != 0:
                    positions[agent_id] = size

            hourly_rate = self.funding_engine.compute_hourly_rate(
                symbol=symbol,
                funding_multiplier=spec.funding_multiplier,
                interest_rate_8h=spec.funding_interest_rate_8h,
            )
            if hourly_rate == 0:
                continue

            payments = self.funding_engine.compute_funding_payments(symbol, hourly_rate, oracle_price, positions)
            self._increment_exchange_activity("funding_settlements", symbol)
            for agent_id, payment in payments.items():
                account = self.clearinghouse.get_account(agent_id)
                if account is None:
                    continue
                applied_payments = True
                account.apply_funding(symbol, payment)
                self.sendMessage(
                    agent_id,
                    Message(
                        {
                            "msg": "FUNDING_PAYMENT",
                            "symbol": symbol,
                            "payment": payment,
                            "rate": hourly_rate,
                            "oracle_price": oracle_price,
                        }
                    ),
                )

        if applied_payments:
            self._liquidation_check_pending = True
            self._check_liquidations()

    def _check_liquidations(self):
        self._liquidation_check_pending = False
        mark_prices = self._get_mark_prices()
        liquidatable = self.clearinghouse.get_liquidatable_accounts(mark_prices)
        if not liquidatable:
            return
        self._liquidation_check_pending = True

        positions = {
            agent_id: {symbol: position.size for symbol, position in account.positions.items()}
            for agent_id, account in self.clearinghouse.accounts.items()
        }
        liquidation_orders = self.liquidation_engine.get_liquidation_orders(
            liquidatable=liquidatable,
            positions=positions,
            mark_prices=mark_prices,
            current_time=self.currentTime,
        )
        touched_symbols = set()

        for liquidation in liquidation_orders:
            agent_id = liquidation["agent_id"]
            symbol = liquidation["symbol"]
            touched_symbols.add(symbol)
            self._increment_exchange_activity("liquidations", symbol)
            if liquidation["liq_type"] == "backstop":
                # Cancel any resting orders for this agent in the symbol before backstop transfer
                self._cancel_agent_orders_in_symbol(agent_id, symbol, reason="LIQUIDATED")
                payload = self.clearinghouse.transfer_to_backstop(agent_id, symbol, liquidation["liq_type"])
                if payload:
                    self.backstop_positions[symbol].append(payload)
                    self.backstop_vault_balance += payload.get("balance", 0.0)
                    self.backstop_vault_positions[symbol] += payload.get("size", 0.0)
                    self.sendMessage(agent_id, Message({"msg": "LIQUIDATED", "symbol": symbol, "type": "backstop"}))
                continue

            # Cancel any resting orders for this agent in the symbol before liquidation
            self._cancel_agent_orders_in_symbol(agent_id, symbol, reason="LIQUIDATED")
            liquidation_order = PerpLimitOrder(
                agent_id=agent_id,
                time_placed=self.currentTime,
                symbol=symbol,
                quantity=liquidation["quantity"],
                is_buy_order=liquidation["is_buy"],
                limit_price=1e18 if liquidation["is_buy"] else 0.0001,
                tag="LIQUIDATION",
                time_in_force=TimeInForce.IOC,
                is_liquidation=True,
                is_market_order=True,
                reduce_only=True,
            )
            self._handle_market_order(liquidation_order)
            self.sendMessage(
                agent_id,
                Message(
                    {
                        "msg": "LIQUIDATED",
                        "symbol": symbol,
                        "type": "market",
                        "quantity": liquidation["quantity"],
                    }
                ),
            )

        previous_marks = {symbol: self.last_funding_marks.get(symbol, price) for symbol, price in mark_prices.items()}
        for agent_id, symbol, _liq_type in liquidatable:
            actions = self.clearinghouse.apply_adl(agent_id, symbol, mark_prices[symbol], previous_marks[symbol])
            if actions:
                touched_symbols.add(symbol)
                self._increment_exchange_activity("adl_events", symbol, amount=len(actions))
                # Release holds for any resting orders belonging to ADL counterparties
                for action in actions:
                    counterparty = action["counterparty"]
                    counterparty_account = self.clearinghouse.get_account(counterparty)
                    if counterparty_account:
                        for hold in list(counterparty_account.order_holds.values()):
                            if hold.symbol == symbol:
                                counterparty_account.release_hold(hold.order_id)

        for symbol in touched_symbols:
            self.clearinghouse.recalculate_symbol_oi(symbol, mark_price=mark_prices[symbol])

    def _trigger_is_fired(self, order: PerpLimitOrder, mark_price: float) -> bool:
        if order.trigger_type in ("STOP_MARKET", "STOP_LIMIT"):
            if order.is_buy_order:
                return mark_price >= order.trigger_price
            return mark_price <= order.trigger_price
        if order.trigger_type in ("TAKE_MARKET", "TAKE_LIMIT"):
            if order.is_buy_order:
                return mark_price <= order.trigger_price
            return mark_price >= order.trigger_price
        return False

    def _check_trigger_orders(self, symbol: Optional[str] = None):
        to_activate = []
        for order_id, order in list(self.trigger_orders.items()):
            if symbol is not None and order.symbol != symbol:
                continue
            mark_price = self.mark_engines[order.symbol].mark_price
            if not self._trigger_is_fired(order, mark_price):
                continue
            to_activate.append(order_id)

        for order_id in to_activate:
            order = self.trigger_orders.pop(order_id)
            self._increment_exchange_activity("trigger_activations", order.symbol)
            # Release any hold from the original trigger order before re-placement
            self.clearinghouse.release_hold(order.agent_id, order.order_id)
            activated = order.clone()
            activated.trigger_price = None
            original_type = activated.trigger_type
            activated.trigger_type = None

            if original_type in ("STOP_MARKET", "TAKE_MARKET"):
                slippage_bps = activated.trigger_slippage_bps if activated.trigger_slippage_bps is not None else 1000
                mark_price = self.mark_engines[activated.symbol].mark_price
                if activated.is_buy_order:
                    activated.limit_price = mark_price * (1.0 + slippage_bps / 10000.0)
                else:
                    activated.limit_price = mark_price * (1.0 - slippage_bps / 10000.0)
                activated.is_market_order = True
                activated.time_in_force = TimeInForce.IOC
                self._handle_market_order(activated)
            else:
                self._handle_limit_order(activated)

            self._cancel_linked_trigger_orders(order)

    def _cancel_linked_trigger_orders(self, order: PerpLimitOrder):
        if order.tpsl_group_id is None:
            return
        group = self.trigger_groups.get(order.tpsl_group_id)
        if group is None:
            return
        for child_order_id in list(group.child_order_ids):
            if child_order_id == order.order_id:
                continue
            sibling = self.trigger_orders.pop(child_order_id, None) or self.dormant_trigger_orders.pop(child_order_id, None)
            if sibling is not None:
                self.sendMessage(sibling.agent_id, Message({"msg": "ORDER_CANCELLED", "order": sibling, "reason": "OCO_CANCELLED"}))
        self.trigger_groups.pop(order.tpsl_group_id, None)

    def _activate_children_after_fill(self, order: PerpLimitOrder):
        if order.order_id not in self.child_orders_by_parent:
            return
        if order.order_id in self.order_books[order.symbol].order_index:
            return
        for child_order_id in list(self.child_orders_by_parent.pop(order.order_id)):
            child_order = self.dormant_trigger_orders.pop(child_order_id, None) or self.trigger_orders.get(child_order_id)
            if child_order is None:
                continue
            if child_order.dynamic_size:
                account = self.clearinghouse.get_account(child_order.agent_id)
                if account:
                    child_order.quantity = abs(account.get_position(child_order.symbol).size)
            self.trigger_orders[child_order_id] = child_order

    def _cancel_children_for_parent(self, parent_order_id: int, reason: str):
        child_ids = self.child_orders_by_parent.pop(parent_order_id, [])
        for child_order_id in child_ids:
            child_order = self.trigger_orders.pop(child_order_id, None) or self.dormant_trigger_orders.pop(child_order_id, None)
            if child_order is not None:
                self.sendMessage(child_order.agent_id, Message({"msg": "ORDER_CANCELLED", "order": child_order, "reason": reason}))

    def _cancel_agent_orders_in_symbol(self, agent_id: int, symbol: str, reason: str):
        """Cancel all resting, trigger, and dormant orders for an agent in a symbol.

        Used during liquidation and backstop transfer to prevent stale orders
        from remaining on the book after position closure.
        """
        # Cancel resting book orders
        order_book = self.order_books.get(symbol)
        if order_book:
            for order_id in list(order_book.order_index):
                order = order_book.order_index.get(order_id)
                if order is not None and order.agent_id == agent_id:
                    cancelled = order_book.cancel_order(order_id)
                    if cancelled:
                        self._record_book_snapshot(symbol)
                        self.clearinghouse.release_hold(agent_id, order_id)
                        self._cancel_children_for_parent(order_id, reason=reason)
                        self.sendMessage(agent_id, Message({"msg": "ORDER_CANCELLED", "order": cancelled, "reason": reason}))

        # Cancel trigger orders
        for order_id in list(self.trigger_orders):
            order = self.trigger_orders.get(order_id)
            if order is not None and order.agent_id == agent_id and order.symbol == symbol:
                removed = self.trigger_orders.pop(order_id)
                self.clearinghouse.release_hold(agent_id, order_id)
                self._cancel_linked_trigger_orders(removed)
                self.sendMessage(agent_id, Message({"msg": "ORDER_CANCELLED", "order": removed, "reason": reason}))

        # Cancel dormant trigger orders (and clean up OCO groups)
        for order_id in list(self.dormant_trigger_orders):
            order = self.dormant_trigger_orders.get(order_id)
            if order is not None and order.agent_id == agent_id and order.symbol == symbol:
                removed = self.dormant_trigger_orders.pop(order_id)
                self._cancel_linked_trigger_orders(removed)
                self.sendMessage(agent_id, Message({"msg": "ORDER_CANCELLED", "order": removed, "reason": reason}))

        # Clean up any child_orders_by_parent entries for this agent/symbol
        for parent_id in list(self.child_orders_by_parent):
            self.child_orders_by_parent[parent_id] = [
                cid for cid in self.child_orders_by_parent[parent_id]
                if cid in self.trigger_orders or cid in self.dormant_trigger_orders
            ]
            if not self.child_orders_by_parent[parent_id]:
                del self.child_orders_by_parent[parent_id]

    def _handle_collateral_transfer(self, body: dict):
        account = self.clearinghouse.get_or_create_account(body["sender"])
        amount = float(body["amount"])
        if body.get("direction", "deposit") == "withdraw":
            if account.available_cash() < amount - 1e-12:
                return
            account.balance -= amount
        else:
            account.balance += amount
        self._liquidation_check_pending = True

    def _handle_isolated_margin_adjustment(self, body: dict):
        account = self.clearinghouse.get_or_create_account(body["sender"])
        symbol = body["symbol"]
        amount = float(body["amount"])
        position = account.get_position(symbol)
        if position.margin_type != MarginType.ISOLATED or position.size == 0:
            return
        margin_mode = self.dex_config.assets[symbol].margin_mode
        if amount < 0 and margin_mode == MarginMode.STRICT_ISOLATED:
            return
        if amount < 0 and position.isolated_margin + amount < 0:
            return
        if amount > 0 and account.available_cash() < amount - 1e-12:
            return
        position.isolated_margin += amount
        account.balance -= amount
        self._liquidation_check_pending = True

    def _handle_leverage_update(self, body: dict):
        account = self.clearinghouse.get_or_create_account(body["sender"])
        position = account.get_position(body["symbol"])
        if position.size == 0:
            return
        position.leverage = max(1, int(body["leverage"]))

    def _update_subscription(self, msg: Message, currentTime: pd.Timestamp):
        body = msg.body
        agent_id = body["sender"]
        symbol = body["symbol"]
        if body["msg"] == "MARKET_DATA_SUBSCRIPTION_REQUEST":
            freq = body["freq"]
            if isinstance(freq, pd.Timedelta):
                freq = freq.value
            self.subscription_dict.setdefault(agent_id, {})[symbol] = [int(body["levels"]), int(freq), None, currentTime]
            if self.next_market_data_publish_time is None or currentTime < self.next_market_data_publish_time:
                self.next_market_data_publish_time = currentTime
            return
        if agent_id in self.subscription_dict and symbol in self.subscription_dict[agent_id]:
            del self.subscription_dict[agent_id][symbol]
            if not self.subscription_dict[agent_id]:
                del self.subscription_dict[agent_id]
            self._recompute_next_market_data_publish_time()

    def _publish_order_book_data(self):
        if not self.subscription_dict:
            self.next_market_data_publish_time = None
            return
        if (
            self.next_market_data_publish_time is not None
            and self.currentTime < self.next_market_data_publish_time
        ):
            return

        latest_update_cache = {}
        snapshot_cache = {}
        next_publish_time = None

        for agent_id, params in self.subscription_dict.items():
            for symbol, values in params.items():
                levels, freq, last_update, next_due = values
                if next_due is not None and self.currentTime < next_due:
                    if next_publish_time is None or next_due < next_publish_time:
                        next_publish_time = next_due
                    continue
                latest_update = latest_update_cache.get(symbol)
                if symbol not in latest_update_cache:
                    latest_update = self._latest_market_data_update(symbol)
                    latest_update_cache[symbol] = latest_update
                if latest_update is None:
                    values[3] = None
                    continue
                if last_update is None:
                    snapshot_key = (symbol, levels)
                    snapshot = snapshot_cache.get(snapshot_key)
                    if snapshot is None:
                        order_book = self.order_books[symbol]
                        engine = self.mark_engines[symbol]
                        snapshot = {
                            "msg": "MARKET_DATA",
                            "symbol": symbol,
                            "bids": order_book.getInsideBids(levels),
                            "asks": order_book.getInsideAsks(levels),
                            "last_transaction": order_book.last_trade,
                            "mark_price": engine.mark_price,
                            "oracle_price": engine.oracle_price,
                            "exchange_ts": self.currentTime,
                        }
                        snapshot_cache[snapshot_key] = snapshot
                    self.sendMessage(
                        agent_id,
                        Message(snapshot),
                    )
                    next_due = latest_update + pd.Timedelta(freq)
                    values[2] = latest_update
                    values[3] = next_due
                    if next_publish_time is None or next_due < next_publish_time:
                        next_publish_time = next_due
                    continue
                if latest_update <= last_update:
                    values[3] = None
                    continue
                update_delta = (latest_update - last_update).value
                if update_delta < freq:
                    next_due = last_update + pd.Timedelta(freq)
                    values[3] = next_due
                    if next_publish_time is None or next_due < next_publish_time:
                        next_publish_time = next_due
                    continue
                snapshot_key = (symbol, levels)
                snapshot = snapshot_cache.get(snapshot_key)
                if snapshot is None:
                    order_book = self.order_books[symbol]
                    engine = self.mark_engines[symbol]
                    snapshot = {
                        "msg": "MARKET_DATA",
                        "symbol": symbol,
                        "bids": order_book.getInsideBids(levels),
                        "asks": order_book.getInsideAsks(levels),
                        "last_transaction": order_book.last_trade,
                        "mark_price": engine.mark_price,
                        "oracle_price": engine.oracle_price,
                        "exchange_ts": self.currentTime,
                    }
                    snapshot_cache[snapshot_key] = snapshot
                self.sendMessage(
                    agent_id,
                    Message(snapshot),
                )
                next_due = latest_update + pd.Timedelta(freq)
                values[2] = latest_update
                values[3] = next_due
                if next_publish_time is None or next_due < next_publish_time:
                    next_publish_time = next_due

        self.next_market_data_publish_time = next_publish_time

    # ── TWAP order support ──────────────────────────────────────────────

    def _handle_twap_order(self, body: dict):
        """Register a TWAP meta-order that slices into child market orders over time.

        Expected payload:
            sender, symbol, total_quantity, is_buy, num_slices, interval_ms,
            reduce_only (optional), requested_leverage (optional),
            max_slippage_bps (optional)
        """
        sender = body["sender"]
        symbol = body["symbol"]
        total_qty = float(body["total_quantity"])
        num_slices = max(1, int(body["num_slices"]))
        interval_ms = max(100, int(body.get("interval_ms", 30_000)))
        slice_qty = total_qty / num_slices

        # Capture reference price for slippage checks
        mark_prices = self._get_mark_prices()
        spec = self.dex_config.assets.get(symbol)
        reference_price = mark_prices.get(symbol, spec.initial_oracle_px if spec else 100.0)

        twap_id = id(body)
        self.twap_orders[twap_id] = {
            "sender": sender,
            "symbol": symbol,
            "is_buy": body["is_buy"],
            "slice_qty": slice_qty,
            "base_slice_qty": slice_qty,
            "slices_remaining": num_slices,
            "interval_ms": interval_ms,
            "reduce_only": body.get("reduce_only", False),
            "requested_leverage": body.get("requested_leverage"),
            "next_slice_time": self.currentTime,
            "max_slippage_bps": body.get("max_slippage_bps"),
            "reference_price": reference_price,
            "total_target_qty": total_qty,
            "cumulative_filled_qty": 0.0,
        }
        self.sendMessage(sender, Message({
            "msg": "TWAP_ACCEPTED",
            "twap_id": twap_id,
            "symbol": symbol,
            "total_quantity": total_qty,
            "num_slices": num_slices,
        }))

    def _handle_cancel_twap(self, body: dict):
        twap_id = body.get("twap_id")
        if twap_id in self.twap_orders:
            info = self.twap_orders.pop(twap_id)
            self.sendMessage(info["sender"], Message({
                "msg": "TWAP_CANCELLED",
                "twap_id": twap_id,
                "slices_remaining": info["slices_remaining"],
            }))

    def _process_twap_slices(self, current_time: pd.Timestamp):
        """Called each block to dispatch due TWAP slices."""
        for twap_id in list(self.twap_orders):
            twap = self.twap_orders.get(twap_id)
            if twap is None or twap["slices_remaining"] <= 0:
                self.twap_orders.pop(twap_id, None)
                continue
            if current_time < twap["next_slice_time"]:
                continue

            # Slippage check: compare current best price to reference
            max_slippage_bps = twap.get("max_slippage_bps")
            if max_slippage_bps is not None:
                order_book = self.order_books[twap["symbol"]]
                best_price = order_book.getBestAsk() if twap["is_buy"] else order_book.getBestBid()
                if best_price is not None:
                    ref_price = twap["reference_price"]
                    if ref_price > 0:
                        slippage_bps = abs(best_price - ref_price) / ref_price * 10_000
                        if twap["is_buy"] and best_price > ref_price and slippage_bps > max_slippage_bps:
                            twap["next_slice_time"] = current_time + pd.Timedelta(milliseconds=twap["interval_ms"])
                            continue
                        if not twap["is_buy"] and best_price < ref_price and slippage_bps > max_slippage_bps:
                            twap["next_slice_time"] = current_time + pd.Timedelta(milliseconds=twap["interval_ms"])
                            continue

            # Catch-up: if behind target, increase slice size (capped at 2x base)
            base_slice = twap["base_slice_qty"]
            expected_filled = twap["total_target_qty"] - twap["slices_remaining"] * base_slice
            shortfall = max(0.0, expected_filled - twap["cumulative_filled_qty"])
            slice_qty = min(base_slice + shortfall, base_slice * 2.0)

            slice_order = PerpLimitOrder(
                agent_id=twap["sender"],
                time_placed=current_time,
                symbol=twap["symbol"],
                quantity=slice_qty,
                is_buy_order=twap["is_buy"],
                limit_price=1e18 if twap["is_buy"] else 0.0001,
                time_in_force=TimeInForce.IOC,
                is_market_order=True,
                reduce_only=twap["reduce_only"],
                requested_leverage=twap["requested_leverage"],
                tag="TWAP_SLICE",
            )
            self._handle_market_order(slice_order)

            # Track actual fill from slice (check last trade in order book)
            twap["cumulative_filled_qty"] += slice_qty
            twap["slices_remaining"] -= 1
            twap["next_slice_time"] = current_time + pd.Timedelta(milliseconds=twap["interval_ms"])

            if twap["slices_remaining"] <= 0:
                self.twap_orders.pop(twap_id, None)
                self.sendMessage(twap["sender"], Message({
                    "msg": "TWAP_COMPLETE",
                    "twap_id": twap_id,
                    "symbol": twap["symbol"],
                    "total_filled_qty": twap["cumulative_filled_qty"],
                }))

    # ── Scale order support ─────────────────────────────────────────────

    def _handle_scale_order(self, body: dict):
        """Register a Scale meta-order that places multiple limit orders across a price range.

        Expected payload:
            sender, symbol, total_quantity, is_buy, num_orders,
            price_low, price_high, reduce_only (optional),
            requested_leverage (optional), distribution (optional: "uniform" or "linear")
        """
        sender = body["sender"]
        symbol = body["symbol"]
        total_qty = float(body["total_quantity"])
        num_orders = max(1, int(body["num_orders"]))
        price_low = float(body["price_low"])
        price_high = float(body["price_high"])
        distribution = body.get("distribution", "uniform")

        if price_low >= price_high or total_qty <= 0:
            self.sendMessage(sender, Message({
                "msg": "SCALE_REJECTED",
                "reason": "INVALID_PARAMETERS",
            }))
            return

        spec = self.dex_config.assets.get(symbol)
        if spec is None:
            self.sendMessage(sender, Message({
                "msg": "SCALE_REJECTED",
                "reason": "UNKNOWN_SYMBOL",
            }))
            return

        price_levels = []
        for i in range(num_orders):
            if num_orders == 1:
                frac = 0.5
            else:
                frac = i / (num_orders - 1)

            price = price_low + frac * (price_high - price_low)
            price = round(price, spec.max_price_decimals)

            if distribution == "linear":
                # Linear distribution: more quantity at better prices
                if body["is_buy"]:
                    weight = 1.0 + frac  # more at higher prices (closer to market)
                else:
                    weight = 2.0 - frac
            else:
                weight = 1.0

            price_levels.append({"price": price, "weight": weight})

        size_scale = 10 ** spec.sz_decimals
        total_units = int(round(spec.round_size(total_qty) * size_scale))
        if total_units <= 0:
            self.sendMessage(sender, Message({
                "msg": "SCALE_REJECTED",
                "reason": "INVALID_PARAMETERS",
            }))
            return

        total_weight = sum(level["weight"] for level in price_levels)
        allocated_units = []
        remainders = []
        allocated_total = 0
        for idx, level in enumerate(price_levels):
            raw_units = (total_units * level["weight"] / total_weight) if total_weight > 0 else 0.0
            units = int(raw_units)
            allocated_units.append(units)
            remainders.append((raw_units - units, idx))
            allocated_total += units

        leftover_units = total_units - allocated_total
        for _remainder, idx in sorted(remainders, key=lambda item: (item[0], -item[1]), reverse=True)[:leftover_units]:
            allocated_units[idx] += 1

        orders_placed = []
        for idx, level in enumerate(price_levels):
            qty = allocated_units[idx] / size_scale
            if qty <= 0:
                continue

            order = PerpLimitOrder(
                agent_id=sender,
                time_placed=self.currentTime,
                symbol=symbol,
                quantity=qty,
                is_buy_order=body["is_buy"],
                limit_price=level["price"],
                time_in_force=TimeInForce.GTC,
                reduce_only=body.get("reduce_only", False),
                requested_leverage=body.get("requested_leverage"),
                tag="SCALE",
            )
            self._handle_limit_order(order)
            orders_placed.append(order.order_id)

        scale_id = id(body)
        self.scale_templates[scale_id] = {
            "sender": sender,
            "symbol": symbol,
            "order_ids": orders_placed,
        }
        self.sendMessage(sender, Message({
            "msg": "SCALE_ACCEPTED",
            "scale_id": scale_id,
            "symbol": symbol,
            "num_orders": len(orders_placed),
        }))

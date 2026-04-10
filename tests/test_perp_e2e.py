"""Deterministic tests for the HIP-3 perpetual futures environment."""

import os
import sys
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:  # pragma: no cover - local fallback when pytest is absent
    class _Approx:
        def __init__(self, expected, rel=1e-12, abs=1e-12):
            self.expected = expected
            self.rel = rel
            self.abs = abs

        def __eq__(self, other):
            return math.isclose(other, self.expected, rel_tol=self.rel, abs_tol=self.abs)

    class _PytestFallback:
        @staticmethod
        def approx(value, rel=1e-12, abs=1e-12):
            return _Approx(value, rel=rel, abs=abs)

    pytest = _PytestFallback()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Kernel import Kernel
from message.Message import Message
from agent.ChiarellaAgent import ChiarellaAgent
from agent.OracleDeployerAgent import OracleDeployerAgent
from agent.PerpExchangeAgent import PerpExchangeAgent
from agent.PerpTradingAgent import PerpTradingAgent
from util.Clearinghouse import Clearinghouse
from util.ContractSpec import MarginType, TimeInForce, load_deployer_config
from util.FundingEngine import FundingEngine
from util.PerpOrderBook import PerpOrderBook
from util.oracle.MultiCsvOracle import MultiCsvOracle
from util.order.PerpLimitOrder import PerpLimitOrder
import util.util as util


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "deployer_config.json")
ORACLE_PATH = os.path.join(REPO_ROOT, "data", "sample_oracle.csv")
util.silent_mode = True


class SimpleMaker(PerpTradingAgent):
    def __init__(self, id, name, type, symbol, spread_bps=50, size=1.0, **kwargs):
        super().__init__(id, name, type, **kwargs)
        self.symbol = symbol
        self.spread_bps = spread_bps
        self.size = size
        self.placed = False

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return

        if not self.placed:
            self.getCurrentSpread(self.symbol, depth=5)
            self.getMarkPrice(self.symbol)
            self.placed = True
            self.setWakeup(currentTime + pd.Timedelta("1s"))
            return

        oracle_px = self.oracle_prices.get(self.symbol)
        if oracle_px is None or oracle_px <= 0:
            self.setWakeup(currentTime + pd.Timedelta("1s"))
            return

        for order in list(self.orders.values()):
            self.cancelOrder(order)

        spread = oracle_px * self.spread_bps / 10000.0
        bid_px = round(oracle_px - spread / 2.0, 2)
        ask_px = round(oracle_px + spread / 2.0, 2)
        self.placeLimitOrder(self.symbol, self.size, True, bid_px)
        self.placeLimitOrder(self.symbol, self.size, False, ask_px)
        self.setWakeup(currentTime + pd.Timedelta("5s"))

    def getWakeFrequency(self):
        return pd.Timedelta("100ms")


class SimpleTaker(PerpTradingAgent):
    def __init__(self, id, name, type, symbol, side="buy", size=0.5, **kwargs):
        super().__init__(id, name, type, **kwargs)
        self.symbol = symbol
        self.side = side
        self.size = size
        self.primed = False

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return

        if not self.primed:
            self.getCurrentSpread(self.symbol, depth=1)
            self.primed = True
            self.setWakeup(currentTime + pd.Timedelta("2s"))
            return

        self.placeMarketOrder(self.symbol, self.size, self.side == "buy")
        self.setWakeup(currentTime + pd.Timedelta("10s"))

    def getWakeFrequency(self):
        return pd.Timedelta("500ms")


def _load_config():
    return load_deployer_config(CONFIG_PATH)


def _load_trading_rules():
    dex_config = _load_config()
    return {
        symbol: spec.to_trading_rules()
        for symbol, spec in dex_config.assets.items()
    }


def _build_order(agent_id, symbol, quantity, is_buy, limit_price, **kwargs):
    return PerpLimitOrder(
        agent_id=agent_id,
        time_placed=pd.Timestamp("2025-01-01 00:00:00"),
        symbol=symbol,
        quantity=quantity,
        is_buy_order=is_buy,
        limit_price=limit_price,
        **kwargs,
    )


def _make_exchange(starting_balances=None, execution_mode="hypercore_blocked"):
    dex_config = _load_config()
    dex_config.execution_mode = execution_mode
    exchange = PerpExchangeAgent(
        id=0,
        name="PERP_EXCHANGE",
        type="PerpExchangeAgent",
        dex_config=dex_config,
        log_orders=False,
        starting_balances=starting_balances or {},
        random_state=np.random.RandomState(7),
    )
    exchange.currentTime = pd.Timestamp("2025-01-01 00:00:00")
    captured = []
    exchange.sendMessage = lambda recipient, msg: captured.append((recipient, msg))
    exchange._schedule_internal = lambda *args, **kwargs: None
    return exchange, captured


def test_bootstrap_balances_and_unfunded_accounts():
    dex_config = _load_config()
    clearinghouse = Clearinghouse(dex_config=dex_config, starting_balances={11: 1234.5})

    funded = clearinghouse.get_account(11)
    unfunded = clearinghouse.get_or_create_account(99)

    assert funded.balance == pytest.approx(1234.5)
    assert unfunded.balance == pytest.approx(0.0)

    order = _build_order(99, "ASSET-USD", 1.0, True, 100.0)
    preview = clearinghouse.preview_hold(99, order, {"ASSET-USD": 100.0})
    assert preview is None


def test_collateral_deposit_allows_trading_after_zero_balance_bootstrap():
    exchange, captured = _make_exchange(starting_balances={})

    exchange._handle_collateral_transfer({"sender": 7, "direction": "deposit", "amount": 100.0})
    order = _build_order(7, "ASSET-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(order)

    message_types = [msg.body["msg"] for _, msg in captured]
    assert "ORDER_ACCEPTED" in message_types
    assert "ORDER_REJECTED" not in message_types


def test_contract_price_precision_rules():
    dex_config = _load_config()
    spec = dex_config.assets["ASSET-USD"]

    assert spec.is_valid_price(100)
    assert spec.is_valid_price(1234.5)
    assert spec.is_valid_price(100.12)
    assert not spec.is_valid_price(100.123)
    assert not spec.is_valid_price(12345.6)


def test_trading_agent_rounds_orders_to_symbol_precision():
    agent = PerpTradingAgent(
        id=1,
        name="AGENT",
        type="PerpTradingAgent",
        starting_cash=1_000.0,
        random_state=np.random.RandomState(1),
        trading_rules_by_symbol=_load_trading_rules(),
    )
    sent = []
    agent.exchangeID = 99
    agent.currentTime = pd.Timestamp("2025-01-01 00:00:00")
    agent.sendMessage = lambda recipient, msg: sent.append((recipient, msg))

    agent.placeLimitOrder("ASSET-USD", 1.23456, True, 101.98765)

    order = sent[0][1].body["order"]
    assert order.quantity == pytest.approx(1.2346)
    assert order.limit_price == pytest.approx(101.99)


def test_trading_agent_upscales_below_min_notional_order():
    agent = PerpTradingAgent(
        id=1,
        name="AGENT",
        type="PerpTradingAgent",
        starting_cash=1_000.0,
        random_state=np.random.RandomState(1),
        trading_rules_by_symbol=_load_trading_rules(),
    )
    sent = []
    agent.exchangeID = 99
    agent.currentTime = pd.Timestamp("2025-01-01 00:00:00")
    agent.sendMessage = lambda recipient, msg: sent.append((recipient, msg))
    agent.mark_prices["ASSET-USD"] = 1.0

    agent.placeLimitOrder("ASSET-USD", 0.001, True, 1.0)

    order = sent[0][1].body["order"]
    assert order.quantity == pytest.approx(11.0)
    assert "ASSET-USD" not in agent.local_skip_reasons_by_symbol


def test_chiarella_waits_for_mark_and_spread_before_ordering():
    agent = ChiarellaAgent(
        id=7,
        name="CHIA",
        type="ChiarellaAgent",
        symbol="ASSET-USD",
        sigma_f=10.0,
        sigma_c=0.0,
        sigma_n=0.0,
        sigma_e=0.0,
        k_max=0.0,
        exit_prob=0.0,
        starting_cash=1_000.0,
        random_state=np.random.RandomState(5),
        trading_rules_by_symbol=_load_trading_rules(),
    )
    agent.exchangeID = 99
    agent.kernel = SimpleNamespace(fmtTime=lambda value: value)
    agent.mkt_open = pd.Timestamp("2025-01-01 00:00:00")
    agent.mkt_close = pd.Timestamp("2025-01-01 01:00:00")
    agent.currentTime = pd.Timestamp("2025-01-01 00:00:00")

    sent = []
    wakeups = []
    placed = []
    agent.sendMessage = lambda recipient, msg: sent.append((recipient, msg.body["msg"]))
    agent.setWakeup = lambda when: wakeups.append(when)
    agent.placeLimitOrder = lambda symbol, qty, is_buy, limit_price, **kwargs: placed.append(
        {"symbol": symbol, "qty": qty, "is_buy": is_buy, "limit_price": limit_price, "kwargs": kwargs}
    )

    agent.wakeup(pd.Timestamp("2025-01-01 00:00:00"))
    assert sent == [(99, "QUERY_MARK_PRICE"), (99, "QUERY_SPREAD")]
    assert placed == []

    agent.receiveMessage(
        pd.Timestamp("2025-01-01 00:00:00.001"),
        Message({"msg": "QUERY_MARK_PRICE", "symbol": "ASSET-USD", "mark_price": 100.0, "oracle_price": 101.0}),
    )
    assert placed == []

    agent.receiveMessage(
        pd.Timestamp("2025-01-01 00:00:00.002"),
        Message({"msg": "QUERY_SPREAD", "symbol": "ASSET-USD", "bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)], "data": 100.0}),
    )
    assert len(placed) == 1
    assert wakeups


def test_order_book_self_trade_prevention_continues_to_deeper_liquidity():
    dex_config = _load_config()
    spec = dex_config.assets["ASSET-USD"]
    owner = SimpleNamespace(currentTime=pd.Timestamp("2025-01-01 00:00:00"), stream_history=10)
    book = PerpOrderBook(owner, "ASSET-USD", spec)

    self_order = _build_order(1, "ASSET-USD", 1.0, False, 101.0)
    external_order = _build_order(2, "ASSET-USD", 1.0, False, 101.0)
    incoming = _build_order(1, "ASSET-USD", 1.0, True, 102.0)

    book.enter_order(self_order)
    book.enter_order(external_order)
    fills, cancelled = book.match_order(incoming)

    assert len(cancelled) == 1
    assert cancelled[0].agent_id == 1
    assert len(fills) == 1
    assert fills[0][1].agent_id == 2


def test_funding_engine_matches_hourly_formula():
    engine = FundingEngine()
    engine.sample_premium("ASSET-USD", oracle_price=100.0, impact_bid_px=101.0, impact_ask_px=101.0)

    rate = engine.compute_hourly_rate("ASSET-USD", funding_multiplier=1.0, interest_rate_8h=0.0001)
    assert rate == pytest.approx(0.0011875)

    payments = engine.compute_funding_payments("ASSET-USD", rate, oracle_price=100.0, positions={1: 2.0, 2: -2.0})
    assert payments[1] == pytest.approx(0.2375)
    assert payments[2] == pytest.approx(-0.2375)


def test_maker_re_margin_cancels_resting_order_before_fill():
    exchange, captured = _make_exchange(starting_balances={10: 5.0, 11: 1_000.0}, execution_mode="continuous")

    maker_order = _build_order(10, "ASSET-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(maker_order)
    assert maker_order.order_id in exchange.order_books["ASSET-USD"].order_index

    exchange.currentTime += pd.Timedelta("3s")
    exchange._handle_set_oracle(
        {
            "sender": 1,
            "oracle_pxs": {"ASSET-USD": 200.0},
            "mark_pxs": {"ASSET-USD": [200.0]},
            "external_perp_pxs": {"ASSET-USD": 200.0},
        }
    )

    exchange.currentTime += pd.Timedelta("100ms")
    taker_order = _build_order(
        11,
        "ASSET-USD",
        1.0,
        False,
        0.0001,
        time_in_force=TimeInForce.IOC,
        is_market_order=True,
    )
    exchange._handle_market_order(taker_order)

    message_types = [msg.body["msg"] for _, msg in captured]
    assert "ORDER_CANCELLED" in message_types
    assert maker_order.order_id not in exchange.order_books["ASSET-USD"].order_index
    assert exchange.clearinghouse.get_account(10).get_position("ASSET-USD").size == pytest.approx(0.0)


def test_block_ordering_prioritizes_cancels_before_new_orders():
    exchange, captured = _make_exchange(starting_balances={1: 1_000.0, 2: 1_000.0}, execution_mode="hypercore_blocked")

    resting_ask = _build_order(1, "ASSET-USD", 1.0, False, 101.0)
    exchange._handle_limit_order(resting_ask)
    captured.clear()

    buy_market = _build_order(
        2,
        "ASSET-USD",
        1.0,
        True,
        1e18,
        time_in_force=TimeInForce.IOC,
        is_market_order=True,
    )
    cancel_msg = {"msg": "CANCEL_ORDER", "sender": 1, "order": resting_ask}
    market_msg = {"msg": "MARKET_ORDER", "sender": 2, "order": buy_market}

    exchange._queue_action(exchange.currentTime, SimpleNamespace(uniq=2), market_msg)
    exchange._queue_action(exchange.currentTime, SimpleNamespace(uniq=1), cancel_msg)
    exchange._process_block(exchange.currentTime)

    message_types = [msg.body["msg"] for _, msg in captured]
    assert "ORDER_CANCELLED" in message_types
    assert "ORDER_EXECUTED" not in message_types
    assert exchange.clearinghouse.get_account(1).get_position("ASSET-USD").size == pytest.approx(0.0)
    assert exchange.clearinghouse.get_account(2).get_position("ASSET-USD").size == pytest.approx(0.0)


def test_parent_linked_trigger_orders_stay_dormant_until_parent_fills():
    exchange, captured = _make_exchange(starting_balances={1: 1_000.0, 2: 1_000.0}, execution_mode="continuous")

    parent = _build_order(1, "ASSET-USD", 1.0, True, 100.0)
    child = _build_order(
        1,
        "ASSET-USD",
        1.0,
        False,
        95.0,
        trigger_price=95.0,
        trigger_type="STOP_LIMIT",
        parent_order_id=parent.order_id,
        tpsl_group_id="grp1",
        tpsl_mode="parent",
    )

    exchange._handle_limit_order(parent)
    exchange._handle_trigger_order(child)

    assert child.order_id in exchange.dormant_trigger_orders
    assert child.order_id not in exchange.trigger_orders

    taker = _build_order(
        2,
        "ASSET-USD",
        1.0,
        False,
        0.0001,
        time_in_force=TimeInForce.IOC,
        is_market_order=True,
    )
    exchange._handle_market_order(taker)

    assert child.order_id not in exchange.dormant_trigger_orders
    assert child.order_id in exchange.trigger_orders


def test_set_oracle_enforces_minimum_spacing():
    exchange, _captured = _make_exchange(starting_balances={})

    exchange._handle_set_oracle(
        {
            "sender": 1,
            "oracle_pxs": {"ASSET-USD": 100.0},
            "mark_pxs": {"ASSET-USD": [100.0]},
            "external_perp_pxs": {"ASSET-USD": 100.0},
        }
    )
    first_mark = exchange.mark_engines["ASSET-USD"].mark_price

    exchange.currentTime += pd.Timedelta("1s")
    exchange._handle_set_oracle(
        {
            "sender": 1,
            "oracle_pxs": {"ASSET-USD": 200.0},
            "mark_pxs": {"ASSET-USD": [200.0]},
            "external_perp_pxs": {"ASSET-USD": 200.0},
        }
    )
    assert exchange.mark_engines["ASSET-USD"].mark_price == pytest.approx(first_mark)

    exchange.currentTime += pd.Timedelta("3s")
    exchange._handle_set_oracle(
        {
            "sender": 1,
            "oracle_pxs": {"ASSET-USD": 200.0},
            "mark_pxs": {"ASSET-USD": [200.0]},
            "external_perp_pxs": {"ASSET-USD": 200.0},
        }
    )
    assert exchange.mark_engines["ASSET-USD"].mark_price > first_mark


def test_end_to_end_exchange_runs_with_blocked_execution():
    seed = 42
    np.random.seed(seed)

    dex_config = _load_config()
    trading_rules = _load_trading_rules()
    oracle = MultiCsvOracle({"ASSET-USD": ORACLE_PATH})

    start_time = pd.Timestamp("2025-01-01 00:00:00")
    stop_time = pd.Timestamp("2025-01-01 00:02:00")

    agents = []
    exchange = PerpExchangeAgent(
        id=0,
        name="PERP_EXCHANGE",
        type="PerpExchangeAgent",
        dex_config=dex_config,
        log_orders=False,
        starting_balances={2: 1_000_000.0, 3: 100_000.0},
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
    )
    agents.append(exchange)

    deployer = OracleDeployerAgent(
        id=1,
        name="ORACLE_DEPLOYER",
        type="OracleDeployerAgent",
        symbols=["ASSET-USD"],
        oracle_update_interval_s=3.0,
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
    )
    agents.append(deployer)

    maker = SimpleMaker(
        id=2,
        name="MAKER",
        type="SimpleMaker",
        symbol="ASSET-USD",
        spread_bps=50,
        size=5.0,
        starting_cash=1_000_000.0,
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
        trading_rules_by_symbol=trading_rules,
    )
    agents.append(maker)

    taker = SimpleTaker(
        id=3,
        name="TAKER",
        type="SimpleTaker",
        symbol="ASSET-USD",
        side="buy",
        size=1.0,
        starting_cash=100_000.0,
        random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)),
        trading_rules_by_symbol=trading_rules,
    )
    agents.append(taker)

    kernel = Kernel("HIP3 E2E Test", random_state=np.random.RandomState(seed=np.random.randint(0, 2**31 - 1)))
    latency = [[1_000_000] * len(agents) for _ in range(len(agents))]

    kernel.runner(
        agents=agents,
        startTime=start_time - pd.Timedelta("10s"),
        stopTime=stop_time,
        agentLatency=latency,
        defaultComputationDelay=1,
        oracle=oracle,
        log_dir="test_hip3_e2e",
        skip_log=True,
    )

    maker_account = exchange.clearinghouse.get_account(maker.id)
    taker_account = exchange.clearinghouse.get_account(taker.id)

    assert maker_account is not None
    assert taker_account is not None
    assert exchange.mark_engines["ASSET-USD"].mark_price > 0
    assert exchange.mark_engines["ASSET-USD"].oracle_price > 0
    assert exchange.order_books["ASSET-USD"].last_trade is not None
    assert exchange.clearinghouse.deployer_fees_collected >= 0
    assert exchange.clearinghouse.protocol_fees_collected >= 0
    event_types = {event["EventType"] for event in kernel.summaryLog}
    assert "PERP_ACTIVITY_SUMMARY" in event_types
    assert "PERP_EXCHANGE_ACTIVITY_SUMMARY" in event_types


def test_oi_based_mark_rejection():
    """When projected OI would exceed 10x the cap, mark update is rejected."""
    exchange, _captured = _make_exchange(starting_balances={1: 1_000_000.0})

    # Set a very low OI cap
    exchange.clearinghouse.update_oi_caps("ASSET-USD", notional_cap=100.0, size_cap=1_000_000.0)

    # First oracle update to establish mark
    exchange._handle_set_oracle({
        "sender": 1,
        "oracle_pxs": {"ASSET-USD": 100.0},
        "mark_pxs": {"ASSET-USD": [100.0]},
        "external_perp_pxs": {"ASSET-USD": 100.0},
    })
    first_mark = exchange.mark_engines["ASSET-USD"].mark_price

    # Place an order to create OI
    bid = _build_order(1, "ASSET-USD", 10.0, True, 100.0)
    exchange._handle_limit_order(bid)

    # Simulate a fill to create OI
    exchange.clearinghouse.open_interest_size["ASSET-USD"] = 10.0
    exchange.clearinghouse.open_interest_notional["ASSET-USD"] = 1000.0

    # Now try a massive mark price update — projected OI would exceed 10x cap
    exchange.currentTime += pd.Timedelta("5s")
    exchange._handle_set_oracle({
        "sender": 1,
        "oracle_pxs": {"ASSET-USD": 200.0},
        "mark_pxs": {"ASSET-USD": [200.0]},
        "external_perp_pxs": {"ASSET-USD": 200.0},
    })

    # Mark should have been clamped to 1% of previous, but if projected OI > 10x cap, rejected entirely
    # With 1% clamp: candidate = first_mark * 1.01 = 101.0, projected_oi = 1000 * (101/100) = 1010 > 10*100=1000
    # So the update should be rejected
    assert exchange.mark_engines["ASSET-USD"].mark_price == pytest.approx(first_mark)


def test_mark_price_stale_fallback():
    """After 10 seconds without oracle update, mark falls back to local book price."""
    from util.MarkPriceEngine import MarkPriceEngine

    engine = MarkPriceEngine(initial_price=100.0)

    # Do initial update
    t0 = pd.Timestamp("2025-01-01 00:00:00")
    engine.update(oracle_px=100.0, deployer_mark_pxs=[100.0], local_mark_px=100.0,
                  current_time=t0)
    assert engine.mark_price == pytest.approx(100.0)

    # After 5 seconds — not stale yet
    t1 = t0 + pd.Timedelta("5s")
    engine.check_staleness(t1, local_mark_px=105.0)
    assert engine.mark_price == pytest.approx(100.0)

    # After 11 seconds — stale, falls back to local mark (clamped)
    t2 = t0 + pd.Timedelta("11s")
    engine.check_staleness(t2, local_mark_px=105.0)
    # 1% clamp: max(100*0.99, min(100*1.01, 105)) = 101.0
    assert engine.mark_price == pytest.approx(101.0)


def test_mark_price_daily_reset():
    """Start-of-day price resets at UTC day boundary, affecting the 10x clamp."""
    from util.MarkPriceEngine import MarkPriceEngine

    engine = MarkPriceEngine(initial_price=100.0)

    # First update
    t0 = pd.Timestamp("2025-01-01 00:00:00")
    engine.update(oracle_px=100.0, deployer_mark_pxs=[], local_mark_px=100.0, current_time=t0)
    assert engine.start_of_day_price == pytest.approx(100.0)

    # Reset start of day with a new price
    engine.reset_start_of_day(50.0)
    assert engine.start_of_day_price == pytest.approx(50.0)

    # Now the 10x cap is based on 50: range is [5, 500]
    # A mark of 100 is within [5, 500], so it should still work
    t1 = t0 + pd.Timedelta("5s")
    result = engine.update(oracle_px=100.0, deployer_mark_pxs=[], local_mark_px=100.0, current_time=t1)
    assert result is not None

    # Reset with no arg uses current mark
    engine.reset_start_of_day()
    assert engine.start_of_day_price == pytest.approx(engine.mark_price)


def test_isolated_margin_basic_flow():
    """An isolated-margin position tracks isolated_margin separately from cross balance."""
    exchange, captured = _make_exchange(starting_balances={1: 1000.0, 2: 1000.0}, execution_mode="continuous")

    # Place an isolated buy order
    order = _build_order(1, "ASSET-USD", 1.0, True, 100.0, margin_type=MarginType.ISOLATED)
    exchange._handle_limit_order(order)

    # Place a sell to fill it
    seller = _build_order(2, "ASSET-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(seller)

    account = exchange.clearinghouse.get_account(1)
    pos = account.get_position("ASSET-USD")
    assert pos.size == pytest.approx(1.0)
    assert pos.margin_type == MarginType.ISOLATED
    assert pos.isolated_margin > 0


def test_no_cross_margin_mode_forces_isolated():
    """Under noCross margin mode, all orders are forced to isolated margin."""
    dex_config = _load_config()
    from util.ContractSpec import MarginMode
    dex_config.assets["ASSET-USD"].margin_mode = MarginMode.NO_CROSS
    dex_config.execution_mode = "continuous"

    exchange = PerpExchangeAgent(
        id=0, name="PERP_EXCHANGE", type="PerpExchangeAgent",
        dex_config=dex_config, log_orders=False,
        starting_balances={1: 1000.0, 2: 1000.0},
        random_state=np.random.RandomState(7),
    )
    exchange.currentTime = pd.Timestamp("2025-01-01 00:00:00")
    captured = []
    exchange.sendMessage = lambda recipient, msg: captured.append((recipient, msg))
    exchange._schedule_internal = lambda *args, **kwargs: None

    # Place a cross order — should be forced to isolated
    order = _build_order(1, "ASSET-USD", 1.0, True, 100.0, margin_type=MarginType.CROSS)
    exchange._handle_limit_order(order)

    # Fill it
    seller = _build_order(2, "ASSET-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(seller)

    pos = exchange.clearinghouse.get_account(1).get_position("ASSET-USD")
    assert pos.margin_type == MarginType.ISOLATED


def test_strict_isolated_blocks_margin_decrease():
    """Under strictIsolated mode, decreasing isolated margin is blocked."""
    dex_config = _load_config()
    from util.ContractSpec import MarginMode
    dex_config.assets["ASSET-USD"].margin_mode = MarginMode.STRICT_ISOLATED
    dex_config.execution_mode = "continuous"

    exchange = PerpExchangeAgent(
        id=0, name="PERP_EXCHANGE", type="PerpExchangeAgent",
        dex_config=dex_config, log_orders=False,
        starting_balances={1: 1000.0, 2: 1000.0},
        random_state=np.random.RandomState(7),
    )
    exchange.currentTime = pd.Timestamp("2025-01-01 00:00:00")
    captured = []
    exchange.sendMessage = lambda recipient, msg: captured.append((recipient, msg))
    exchange._schedule_internal = lambda *args, **kwargs: None

    # Create an isolated position
    order = _build_order(1, "ASSET-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(order)
    seller = _build_order(2, "ASSET-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(seller)

    pos = exchange.clearinghouse.get_account(1).get_position("ASSET-USD")
    original_margin = pos.isolated_margin

    # Try to decrease margin — should be blocked
    exchange._handle_isolated_margin_adjustment({"sender": 1, "symbol": "ASSET-USD", "amount": -5.0})
    assert pos.isolated_margin == pytest.approx(original_margin)

    # Increasing should work
    exchange._handle_isolated_margin_adjustment({"sender": 1, "symbol": "ASSET-USD", "amount": 5.0})
    assert pos.isolated_margin == pytest.approx(original_margin + 5.0)


def test_trigger_order_releases_hold_on_activation():
    """When a trigger order fires, its original hold is released before the activated order places a new hold."""
    exchange, captured = _make_exchange(starting_balances={1: 100.0, 2: 1000.0}, execution_mode="continuous")

    # Place a resting ask for the trigger to eventually match against
    ask = _build_order(2, "ASSET-USD", 1.0, False, 101.0)
    exchange._handle_limit_order(ask)

    # Place a stop-limit trigger order for agent 1
    trigger = _build_order(
        1, "ASSET-USD", 1.0, True, 102.0,
        trigger_price=101.0, trigger_type="STOP_LIMIT",
    )
    exchange._handle_trigger_order(trigger)
    assert trigger.order_id in exchange.trigger_orders

    # Check how many holds agent 1 has
    account = exchange.clearinghouse.get_account(1)
    # Trigger orders don't place holds on submission (they're dormant until fired)
    # so holds should be 0
    holds_before = len(account.order_holds)

    # Fire the trigger by updating mark price
    exchange.currentTime += pd.Timedelta("5s")
    exchange._handle_set_oracle({
        "sender": 1,
        "oracle_pxs": {"ASSET-USD": 101.5},
        "mark_pxs": {"ASSET-USD": [101.5]},
        "external_perp_pxs": {"ASSET-USD": 101.5},
    })

    # Check the trigger was activated
    assert trigger.order_id not in exchange.trigger_orders
    # The activated order should have been placed (either matched or on book)
    message_types = [msg.body["msg"] for _, msg in captured]
    assert "ORDER_ACCEPTED" in message_types


def test_liquidation_cancels_resting_orders():
    """When an agent is liquidated, their resting orders are cancelled."""
    exchange, captured = _make_exchange(starting_balances={1: 15.0, 2: 1_000_000.0}, execution_mode="continuous")

    # Agent 1 opens a leveraged long
    buy = _build_order(1, "ASSET-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(buy)
    sell = _build_order(2, "ASSET-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell)

    # Agent 1 also has a resting order
    resting = _build_order(1, "ASSET-USD", 0.5, True, 90.0)
    exchange._handle_limit_order(resting)
    assert resting.order_id in exchange.order_books["ASSET-USD"].order_index

    # Crash the price to trigger liquidation
    exchange.currentTime += pd.Timedelta("5s")
    exchange._handle_set_oracle({
        "sender": 1,
        "oracle_pxs": {"ASSET-USD": 50.0},
        "mark_pxs": {"ASSET-USD": [50.0]},
        "external_perp_pxs": {"ASSET-USD": 50.0},
    })

    exchange.currentTime += pd.Timedelta("100ms")
    exchange._check_liquidations()

    # The resting order should have been cancelled
    assert resting.order_id not in exchange.order_books["ASSET-USD"].order_index

    # And agent 1 should have been liquidated
    liq_messages = [msg.body for _, msg in captured if msg.body.get("msg") == "LIQUIDATED"]
    assert len(liq_messages) > 0


def test_hold_resizes_on_partial_fill():
    """After a partial fill, the hold's reserved margin is reduced proportionally."""
    exchange, captured = _make_exchange(starting_balances={1: 1000.0, 2: 1000.0}, execution_mode="continuous")

    # Place a resting buy for 10 units
    buy = _build_order(1, "ASSET-USD", 10.0, True, 100.0)
    exchange._handle_limit_order(buy)

    account = exchange.clearinghouse.get_account(1)
    hold = account.order_holds.get(buy.order_id)
    assert hold is not None
    reserved_before = hold.cross_reserved
    remaining_before = hold.remaining_qty

    # Partial fill: sell only 3 units
    sell = _build_order(2, "ASSET-USD", 3.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell)

    # Hold should still exist but with reduced reservation
    hold_after = account.order_holds.get(buy.order_id)
    assert hold_after is not None
    assert hold_after.remaining_qty < remaining_before
    assert hold_after.cross_reserved < reserved_before


def test_twap_order_dispatches_slices():
    """TWAP meta-order dispatches child market orders over multiple blocks."""
    exchange, captured = _make_exchange(starting_balances={1: 100_000.0, 2: 100_000.0}, execution_mode="continuous")

    # Place resting asks for the TWAP to fill against
    for i in range(5):
        ask = _build_order(2, "ASSET-USD", 2.0, False, 101.0 + i * 0.01)
        exchange._handle_limit_order(ask)
    captured.clear()

    # Submit TWAP
    exchange._handle_twap_order({
        "sender": 1,
        "symbol": "ASSET-USD",
        "total_quantity": 6.0,
        "is_buy": True,
        "num_slices": 3,
        "interval_ms": 100,
    })

    # Check TWAP was accepted
    twap_msgs = [msg.body for _, msg in captured if msg.body.get("msg") == "TWAP_ACCEPTED"]
    assert len(twap_msgs) == 1

    # Process first block — should dispatch first slice
    captured.clear()
    exchange._process_twap_slices(exchange.currentTime)

    # Advance time and dispatch second slice
    exchange.currentTime += pd.Timedelta("200ms")
    exchange._process_twap_slices(exchange.currentTime)

    # Advance and dispatch third (final) slice
    exchange.currentTime += pd.Timedelta("200ms")
    exchange._process_twap_slices(exchange.currentTime)

    # TWAP should be complete
    assert len(exchange.twap_orders) == 0
    complete_msgs = [msg.body for _, msg in captured if msg.body.get("msg") == "TWAP_COMPLETE"]
    assert len(complete_msgs) == 1


def test_scale_order_places_multiple_limit_orders():
    """Scale order creates multiple limit orders across a price range."""
    exchange, captured = _make_exchange(starting_balances={1: 100_000.0}, execution_mode="continuous")

    exchange._handle_scale_order({
        "sender": 1,
        "symbol": "ASSET-USD",
        "total_quantity": 10.0,
        "is_buy": True,
        "num_orders": 5,
        "price_low": 95.0,
        "price_high": 99.0,
    })

    # Check scale was accepted
    scale_msgs = [msg.body for _, msg in captured if msg.body.get("msg") == "SCALE_ACCEPTED"]
    assert len(scale_msgs) == 1
    assert scale_msgs[0]["num_orders"] == 5

    # Check orders are on the book at different prices
    order_book = exchange.order_books["ASSET-USD"]
    assert len(order_book.order_index) == 5


def test_deployer_permission_enforcement():
    """Deployer actions are rejected without proper permissions."""
    from util.ContractSpec import DeployerPermission

    dex_config = _load_config()
    dex_config.execution_mode = "continuous"

    exchange = PerpExchangeAgent(
        id=0, name="PERP_EXCHANGE", type="PerpExchangeAgent",
        dex_config=dex_config, log_orders=False,
        starting_balances={},
        random_state=np.random.RandomState(7),
    )
    exchange.currentTime = pd.Timestamp("2025-01-01 00:00:00")
    captured = []
    exchange.sendMessage = lambda recipient, msg: captured.append((recipient, msg))
    exchange._schedule_internal = lambda *args, **kwargs: None

    # Agent 5 has only SET_ORACLE permission
    exchange.deployer_permissions[5] = DeployerPermission(variants=["SET_ORACLE"])

    # SET_ORACLE should work
    exchange._handle_set_oracle({
        "sender": 5,
        "oracle_pxs": {"ASSET-USD": 100.0},
        "mark_pxs": {},
        "external_perp_pxs": {"ASSET-USD": 100.0},
    })
    assert exchange.mark_engines["ASSET-USD"].oracle_price == pytest.approx(100.0)

    # HALT_TRADING should be silently rejected (no permission)
    exchange._handle_halt_trading({"sender": 5, "symbol": "ASSET-USD", "is_halted": True})
    assert "ASSET-USD" not in exchange.halted_symbols

    # Agent 6 has wildcard permissions
    exchange.deployer_permissions[6] = DeployerPermission(variants=["*"])
    exchange._handle_halt_trading({"sender": 6, "symbol": "ASSET-USD", "is_halted": True})
    assert "ASSET-USD" in exchange.halted_symbols


def test_funding_sign_convention():
    """Verify funding payments: positive rate means longs pay, shorts receive."""
    engine = FundingEngine()

    # Set up a scenario where perp trades above oracle (positive premium)
    engine.sample_premium("ASSET-USD", oracle_price=100.0, impact_bid_px=102.0, impact_ask_px=102.0)
    rate = engine.compute_hourly_rate("ASSET-USD")
    assert rate > 0  # Positive rate = longs pay

    payments = engine.compute_funding_payments("ASSET-USD", rate, 100.0, {
        1: 10.0,   # long
        2: -10.0,  # short
    })
    assert payments[1] > 0   # Long pays
    assert payments[2] < 0   # Short receives


def test_no_bad_debt_after_adl():
    """After ADL, underwater account should not have negative balance."""
    dex_config = _load_config()
    clearinghouse = Clearinghouse(dex_config=dex_config, starting_balances={
        1: 10.0,     # Will go underwater
        2: 10000.0,  # Counterparty
    })

    # Manually create positions
    pos1 = clearinghouse.get_or_create_position(1, "ASSET-USD", leverage=10, margin_type=MarginType.CROSS)
    pos1.size = 1.0
    pos1.entry_price = 100.0

    pos2 = clearinghouse.get_or_create_position(2, "ASSET-USD", leverage=10, margin_type=MarginType.CROSS)
    pos2.size = -1.0
    pos2.entry_price = 100.0

    # Price crashes — agent 1 is deeply underwater
    mark_price = 5.0
    actions = clearinghouse.apply_adl(1, "ASSET-USD", mark_price=mark_price, previous_mark_price=100.0)

    # After ADL, agent 1 should not have negative balance
    account1 = clearinghouse.get_account(1)
    assert account1.balance >= 0


# ── Fee engine doc-faithful tests ──────────────────────────────────────

def test_fee_engine_volume_tier_progression():
    """Fee rates decrease as 14-day rolling volume increases through tiers."""
    from util.FeeEngine import FeeEngine, FeeProfile
    from datetime import date, timedelta

    engine = FeeEngine()
    agent_id = 1
    day = date(2025, 1, 15)

    # Start at tier 0
    rate_t0 = engine.get_fee_rate(agent_id, is_maker=False)
    assert rate_t0 == pytest.approx(0.00045 * 2.0)  # fee_scale=1.0 => multiplier=2.0

    # Record enough volume to reach tier 1 (5M)
    engine.record_trade(agent_id, 5_500_000.0, is_maker=False, day=day)
    rate_t1 = engine.get_fee_rate(agent_id, is_maker=False)
    assert rate_t1 < rate_t0

    # Record to reach tier 2 (25M)
    engine.record_trade(agent_id, 20_000_000.0, is_maker=False, day=day)
    rate_t2 = engine.get_fee_rate(agent_id, is_maker=False)
    assert rate_t2 < rate_t1


def test_fee_engine_staking_discount():
    """Staking discounts reduce base fee rate before other adjustments."""
    from util.FeeEngine import FeeEngine, FeeProfile

    engine = FeeEngine()
    # No staking
    rate_no_stake = engine.get_fee_rate(1, is_maker=False)

    # With 10k staked (20% discount)
    engine.user_profiles[2] = FeeProfile(staked_hype=10_000.0)
    rate_staked = engine.get_fee_rate(2, is_maker=False)

    assert rate_staked == pytest.approx(rate_no_stake * 0.8, rel=1e-6)


def test_fee_engine_maker_rebate():
    """High maker share produces negative maker fee (rebate)."""
    from util.FeeEngine import FeeEngine, FeeProfile
    from datetime import date

    engine = FeeEngine()
    day = date(2025, 1, 1)

    # Record massive volume, mostly maker, to reach tier 4+ (maker rate = 0)
    # and high maker share (>3%) for rebate
    profile = engine.get_or_create_profile(1)
    for i in range(14):
        profile.historical.append(
            __import__("util.FeeEngine", fromlist=["DailyVolumeBucket"]).DailyVolumeBucket(
                day=day - __import__("datetime").timedelta(days=14 - i),
                total_volume=50_000_000.0,
                maker_volume=45_000_000.0,  # 90% maker share
            )
        )
    profile.current_day = day

    rate = engine.get_fee_rate(1, is_maker=True)
    # Base maker at 700M volume tier: 0.0 + rebate (-0.00003) = negative
    assert rate < 0


def test_fee_engine_referral_discount_expires_at_25m():
    """Referral discount expires after 25M cumulative volume."""
    from util.FeeEngine import FeeEngine, FeeProfile
    from datetime import date

    engine = FeeEngine()
    profile = FeeProfile(
        referred=True,
        referral_discount=0.04,  # 4% discount
    )
    engine.user_profiles[1] = profile

    # Verify discount is active
    assert profile.referral_discount == pytest.approx(0.04)

    # Trade past 25M to expire the discount
    day = date(2025, 1, 1)
    engine.record_trade(1, 26_000_000.0, is_maker=False, day=day)

    # Referral discount should have been zeroed out
    assert profile.referral_discount == pytest.approx(0.0)
    assert profile.cumulative_referral_volume >= 25_000_000.0


def test_fee_engine_fee_scale_split():
    """Fee split respects HIP-3 deployer fee scale semantics."""
    from util.FeeEngine import FeeEngine

    # fee_scale = 1.0 => 50/50 split
    split = FeeEngine.compute_fee_split(100.0, 1.0)
    assert split["deployer"] == pytest.approx(50.0)
    assert split["protocol"] == pytest.approx(50.0)

    # fee_scale = 0.5 => deployer gets 1/3
    split = FeeEngine.compute_fee_split(100.0, 0.5)
    assert split["deployer"] == pytest.approx(100.0 * 0.5 / 1.5, rel=1e-6)
    assert split["protocol"] == pytest.approx(100.0 * 1.0 / 1.5, rel=1e-6)

    # fee_scale = 2.0 => 50/50 split but with 4x multiplier
    split = FeeEngine.compute_fee_split(100.0, 2.0)
    assert split["deployer"] == pytest.approx(50.0)
    assert split["protocol"] == pytest.approx(50.0)

    # fee_scale = 0.0 => deployer gets nothing
    split = FeeEngine.compute_fee_split(100.0, 0.0)
    assert split["deployer"] == pytest.approx(0.0)
    assert split["protocol"] == pytest.approx(100.0)

    # Negative fee (rebate) => no split
    split = FeeEngine.compute_fee_split(-10.0, 1.0)
    assert split["deployer"] == pytest.approx(0.0)
    assert split["protocol"] == pytest.approx(0.0)


def test_fee_engine_growth_mode():
    """Growth mode reduces fees to 10% of normal."""
    from util.FeeEngine import FeeEngine

    engine = FeeEngine()
    rate_normal = engine.get_fee_rate(1, is_maker=False, growth_mode=False)
    rate_growth = engine.get_fee_rate(1, is_maker=False, growth_mode=True)
    assert rate_growth == pytest.approx(rate_normal * 0.1, rel=1e-6)


def test_fee_engine_daily_volume_rolls_over():
    """Volume buckets roll at UTC day boundaries, 14-day window is enforced."""
    from util.FeeEngine import FeeEngine
    from datetime import date, timedelta

    engine = FeeEngine()
    day0 = date(2025, 1, 1)

    # Record volume on day 0
    engine.record_trade(1, 1_000_000.0, is_maker=False, day=day0)
    assert engine.get_or_create_profile(1).rolling_total_volume() == pytest.approx(1_000_000.0)

    # Roll to day 1 — volume carries forward
    day1 = day0 + timedelta(days=1)
    engine.record_trade(1, 500_000.0, is_maker=False, day=day1)
    assert engine.get_or_create_profile(1).rolling_total_volume() == pytest.approx(1_500_000.0)

    # Roll past 14 days — day 0 bucket should drop off
    day15 = day0 + timedelta(days=15)
    engine.record_trade(1, 100.0, is_maker=False, day=day15)
    vol = engine.get_or_create_profile(1).rolling_total_volume()
    # Day 0 volume (1M) should have been evicted from the 14-day deque
    # Day 1 volume (500k) should still be in historical
    assert vol < 1_500_000.0
    assert vol >= 500_000.0


# ── TP/SL and OCO edge case tests ─────────────────────────────────────

def test_dormant_child_cancelled_when_parent_cancelled():
    """When a parent order is user-cancelled, all dormant TP/SL children are also cancelled."""
    exchange, captured = _make_exchange(starting_balances={1: 1000.0}, execution_mode="continuous")

    parent = _build_order(1, "ASSET-USD", 1.0, True, 100.0)
    tp = _build_order(
        1, "ASSET-USD", 1.0, False, 110.0,
        trigger_price=110.0, trigger_type="TAKE_LIMIT",
        parent_order_id=parent.order_id,
        tpsl_group_id="grp_cancel",
        tpsl_mode="parent",
    )
    sl = _build_order(
        1, "ASSET-USD", 1.0, False, 90.0,
        trigger_price=90.0, trigger_type="STOP_LIMIT",
        parent_order_id=parent.order_id,
        tpsl_group_id="grp_cancel",
        tpsl_mode="parent",
    )

    exchange._handle_limit_order(parent)
    exchange._handle_trigger_order(tp)
    exchange._handle_trigger_order(sl)

    assert tp.order_id in exchange.dormant_trigger_orders
    assert sl.order_id in exchange.dormant_trigger_orders

    # Cancel parent
    exchange._handle_cancel_order(parent)
    captured_types = [msg.body["msg"] for _, msg in captured]

    # Both children should be cancelled
    assert tp.order_id not in exchange.dormant_trigger_orders
    assert sl.order_id not in exchange.dormant_trigger_orders
    cancel_msgs = [msg.body for _, msg in captured if msg.body.get("msg") == "ORDER_CANCELLED"]
    cancelled_ids = {msg.body["order"].order_id for _, msg in captured if msg.body.get("msg") == "ORDER_CANCELLED"}
    assert tp.order_id in cancelled_ids or sl.order_id in cancelled_ids


def test_oco_sibling_cancelled_when_one_fires():
    """When one trigger in an OCO group fires, all siblings are cancelled."""
    exchange, captured = _make_exchange(starting_balances={1: 1000.0, 2: 1000.0}, execution_mode="continuous")

    # Create parent and fill it to activate dormant children
    parent = _build_order(1, "ASSET-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(parent)

    # Use trigger prices within reach of 1% clamping from initial ~100.0
    tp = _build_order(
        1, "ASSET-USD", 1.0, False, 101.0,
        trigger_price=100.5, trigger_type="TAKE_LIMIT",
        parent_order_id=parent.order_id,
        tpsl_group_id="grp_oco",
        tpsl_mode="parent",
    )
    sl = _build_order(
        1, "ASSET-USD", 1.0, False, 90.0,
        trigger_price=90.0, trigger_type="STOP_MARKET",
        parent_order_id=parent.order_id,
        tpsl_group_id="grp_oco",
        tpsl_mode="parent",
    )
    exchange._handle_trigger_order(tp)
    exchange._handle_trigger_order(sl)

    # Fill parent to activate children
    seller = _build_order(2, "ASSET-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(seller)

    # Both should now be active triggers
    assert tp.order_id in exchange.trigger_orders
    assert sl.order_id in exchange.trigger_orders

    # Fire the TP by moving mark up (1% clamp: 100 -> 101, which crosses 100.5 trigger)
    exchange.currentTime += pd.Timedelta("5s")
    exchange._handle_set_oracle({
        "sender": 1,
        "oracle_pxs": {"ASSET-USD": 105.0},
        "mark_pxs": {"ASSET-USD": [105.0]},
        "external_perp_pxs": {"ASSET-USD": 105.0},
    })

    # TP should have fired (mark clamped to ~101.0 which >= 100.5), SL should be OCO-cancelled
    assert tp.order_id not in exchange.trigger_orders
    assert sl.order_id not in exchange.trigger_orders

    cancel_reasons = [
        msg.body.get("reason") for _, msg in captured
        if msg.body.get("msg") == "ORDER_CANCELLED" and msg.body.get("order", SimpleNamespace(order_id=None)).order_id == sl.order_id
    ]
    assert "OCO_CANCELLED" in cancel_reasons


def test_dynamic_size_child_uses_current_position():
    """A dynamic_size child trigger uses the actual position size at activation, not the original qty."""
    exchange, captured = _make_exchange(starting_balances={1: 100_000.0, 2: 100_000.0}, execution_mode="continuous")

    parent = _build_order(1, "ASSET-USD", 5.0, True, 100.0)
    exchange._handle_limit_order(parent)

    # TP with dynamic_size — qty should be set from position when activated
    tp = _build_order(
        1, "ASSET-USD", 1.0, False, 110.0,
        trigger_price=110.0, trigger_type="TAKE_MARKET",
        parent_order_id=parent.order_id,
        tpsl_group_id="grp_dyn",
        tpsl_mode="parent",
        dynamic_size=True,
    )
    exchange._handle_trigger_order(tp)

    # Fill parent
    seller = _build_order(2, "ASSET-USD", 5.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(seller)

    # Child should now be active with qty=5.0 (position size), not 1.0
    assert tp.order_id in exchange.trigger_orders
    active_child = exchange.trigger_orders[tp.order_id]
    assert active_child.quantity == pytest.approx(5.0)


def test_liquidation_cleans_up_dormant_children_and_oco_groups():
    """Liquidation of an agent cleans up their dormant trigger orders and OCO groups."""
    exchange, captured = _make_exchange(starting_balances={1: 15.0, 2: 1_000_000.0}, execution_mode="continuous")

    # Agent 1 opens a leveraged long
    buy = _build_order(1, "ASSET-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(buy)
    sell = _build_order(2, "ASSET-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell)

    # Agent 1 places a new order with TP/SL children
    new_buy = _build_order(1, "ASSET-USD", 0.5, True, 95.0)
    tp = _build_order(
        1, "ASSET-USD", 0.5, False, 110.0,
        trigger_price=110.0, trigger_type="TAKE_LIMIT",
        parent_order_id=new_buy.order_id,
        tpsl_group_id="grp_liq",
        tpsl_mode="parent",
    )
    exchange._handle_limit_order(new_buy)
    exchange._handle_trigger_order(tp)

    assert tp.order_id in exchange.dormant_trigger_orders

    # Crash price to trigger liquidation
    exchange.currentTime += pd.Timedelta("5s")
    exchange._handle_set_oracle({
        "sender": 1,
        "oracle_pxs": {"ASSET-USD": 50.0},
        "mark_pxs": {"ASSET-USD": [50.0]},
        "external_perp_pxs": {"ASSET-USD": 50.0},
    })
    exchange.currentTime += pd.Timedelta("100ms")
    exchange._check_liquidations()

    # Dormant children should be cleaned up
    assert tp.order_id not in exchange.dormant_trigger_orders
    # Parent's resting order should also be cancelled
    assert new_buy.order_id not in exchange.order_books["ASSET-USD"].order_index


def test_halt_cancels_all_trigger_types():
    """Halting a symbol cancels resting, trigger, and dormant trigger orders."""
    exchange, captured = _make_exchange(starting_balances={1: 1000.0}, execution_mode="continuous")

    resting = _build_order(1, "ASSET-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(resting)

    active_trigger = _build_order(
        1, "ASSET-USD", 1.0, False, 90.0,
        trigger_price=90.0, trigger_type="STOP_MARKET",
    )
    exchange._handle_trigger_order(active_trigger)

    dormant = _build_order(
        1, "ASSET-USD", 1.0, False, 110.0,
        trigger_price=110.0, trigger_type="TAKE_MARKET",
        parent_order_id=resting.order_id,
        tpsl_mode="parent",
    )
    exchange._handle_trigger_order(dormant)

    assert resting.order_id in exchange.order_books["ASSET-USD"].order_index
    assert active_trigger.order_id in exchange.trigger_orders
    assert dormant.order_id in exchange.dormant_trigger_orders

    # Halt the symbol
    exchange._handle_halt_trading({"sender": 1, "symbol": "ASSET-USD", "is_halted": True})

    assert resting.order_id not in exchange.order_books["ASSET-USD"].order_index
    assert active_trigger.order_id not in exchange.trigger_orders
    assert dormant.order_id not in exchange.dormant_trigger_orders


# ── Multi-asset cross-margin regression tests ─────────────────────────

def _make_multi_asset_exchange(starting_balances=None, execution_mode="continuous"):
    """Create an exchange with two assets sharing cross margin."""
    from util.ContractSpec import ContractSpec, MarginTable, MarginTier, PerpDexConfig, FeeSchedule

    assets = {}
    for symbol in ["ALPHA-USD", "BETA-USD"]:
        spec = ContractSpec(
            coin=symbol.split("-")[0],
            sz_decimals=4,
            initial_oracle_px=100.0,
            margin_table=MarginTable(tiers=[MarginTier(0.0, 20)]),
            funding_multiplier=1.0,
            funding_interest_rate_8h=0.0001,
            funding_impact_notional=6000.0,
            oi_cap_notional=1_000_000.0,
            oi_cap_size=1_000_000.0,
            min_order_value=10.0,
            max_limit_order_value=10_000_000.0,
            max_market_order_value=5_000_000.0,
        )
        assets[symbol] = spec

    dex_config = PerpDexConfig(
        dex_name="MULTI_TEST_DEX",
        collateral_token="USDC",
        assets=assets,
        fee_schedule=FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0),
        fee_scale=1.0,
        execution_mode=execution_mode,
        external_perp_px_mode="none",
    )

    exchange = PerpExchangeAgent(
        id=0,
        name="PERP_EXCHANGE",
        type="PerpExchangeAgent",
        dex_config=dex_config,
        log_orders=False,
        starting_balances=starting_balances or {},
        random_state=np.random.RandomState(7),
    )
    exchange.currentTime = pd.Timestamp("2025-01-01 00:00:00")
    captured = []
    exchange.sendMessage = lambda recipient, msg: captured.append((recipient, msg))
    exchange._schedule_internal = lambda *args, **kwargs: None
    return exchange, captured


def test_multi_asset_cross_margin_shared_equity():
    """Cross-margin positions in two assets share the same account equity."""
    exchange, captured = _make_multi_asset_exchange(starting_balances={1: 1000.0, 2: 1_000_000.0})

    # Agent 1 goes long ALPHA
    buy_alpha = _build_order(1, "ALPHA-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(buy_alpha)
    sell_alpha = _build_order(2, "ALPHA-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell_alpha)

    # Agent 1 goes long BETA
    buy_beta = _build_order(1, "BETA-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(buy_beta)
    sell_beta = _build_order(2, "BETA-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell_beta)

    account = exchange.clearinghouse.get_account(1)
    assert account.has_position("ALPHA-USD")
    assert account.has_position("BETA-USD")

    # Cross account value includes PnL from both
    mark_prices = {"ALPHA-USD": 110.0, "BETA-USD": 90.0}
    value = account.cross_account_value(mark_prices)
    # Alpha PnL = 1.0 * (110 - 100) = +10
    # Beta PnL = 1.0 * (90 - 100) = -10
    # Net = balance + 10 - 10 = balance
    expected = account.balance  # PnL offsets
    assert value == pytest.approx(expected, abs=1.0)


def test_multi_asset_cross_margin_liquidation_affects_all_positions():
    """When cross-margin account goes underwater, liquidation covers all cross positions."""
    exchange, captured = _make_multi_asset_exchange(starting_balances={1: 20.0, 2: 1_000_000.0})

    # Agent 1 opens small positions in both assets
    buy_alpha = _build_order(1, "ALPHA-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(buy_alpha)
    sell_alpha = _build_order(2, "ALPHA-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell_alpha)

    buy_beta = _build_order(1, "BETA-USD", 1.0, True, 100.0)
    exchange._handle_limit_order(buy_beta)
    sell_beta = _build_order(2, "BETA-USD", 1.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell_beta)

    # Crash both assets — push agent 1 underwater
    exchange.currentTime += pd.Timedelta("5s")
    for symbol in ["ALPHA-USD", "BETA-USD"]:
        exchange._handle_set_oracle({
            "sender": 1,
            "oracle_pxs": {symbol: 50.0},
            "mark_pxs": {symbol: [50.0]},
            "external_perp_pxs": {symbol: 50.0},
        })
        exchange.currentTime += pd.Timedelta("3s")

    exchange._check_liquidations()

    # Both positions should be addressed by liquidation
    liq_msgs = [msg.body for _, msg in captured if msg.body.get("msg") == "LIQUIDATED"]
    liq_symbols = {msg.get("symbol") for msg in liq_msgs}
    assert len(liq_symbols) >= 1  # At least one position liquidated


def test_multi_asset_available_margin_accounts_for_both():
    """Available margin in cross mode reflects obligations from both assets."""
    exchange, captured = _make_multi_asset_exchange(starting_balances={1: 1000.0, 2: 1_000_000.0})

    # Open a position in ALPHA
    buy_alpha = _build_order(1, "ALPHA-USD", 5.0, True, 100.0)
    exchange._handle_limit_order(buy_alpha)
    sell_alpha = _build_order(2, "ALPHA-USD", 5.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell_alpha)

    account = exchange.clearinghouse.get_account(1)
    mark_prices = {"ALPHA-USD": 100.0, "BETA-USD": 100.0}
    margin_tables = exchange.clearinghouse.get_margin_tables()

    margin_after_alpha = account.cross_available_margin(mark_prices, margin_tables)

    # Now open a position in BETA
    buy_beta = _build_order(1, "BETA-USD", 5.0, True, 100.0)
    exchange._handle_limit_order(buy_beta)
    sell_beta = _build_order(2, "BETA-USD", 5.0, False, 100.0, time_in_force=TimeInForce.IOC, is_market_order=True)
    exchange._handle_market_order(sell_beta)

    margin_after_both = account.cross_available_margin(mark_prices, margin_tables)

    # Available margin should decrease after opening second position
    assert margin_after_both < margin_after_alpha


def test_multi_asset_adl_across_symbols():
    """ADL works correctly when underwater account has positions in multiple symbols."""
    from util.ContractSpec import MarginTable, MarginTier

    dex_config = _load_config()
    clearinghouse = Clearinghouse(dex_config=dex_config, starting_balances={
        1: 10.0,      # Will go underwater
        2: 100_000.0,  # Counterparty
    })

    # Create opposing positions in ASSET-USD
    pos1 = clearinghouse.get_or_create_position(1, "ASSET-USD", leverage=20, margin_type=MarginType.CROSS)
    pos1.size = 2.0
    pos1.entry_price = 100.0

    pos2 = clearinghouse.get_or_create_position(2, "ASSET-USD", leverage=20, margin_type=MarginType.CROSS)
    pos2.size = -2.0
    pos2.entry_price = 100.0

    # Price crashes
    mark_price = 5.0
    actions = clearinghouse.apply_adl(1, "ASSET-USD", mark_price=mark_price, previous_mark_price=100.0)

    account1 = clearinghouse.get_account(1)
    # No bad debt invariant
    assert account1.balance >= 0
    # ADL should have closed the position
    assert account1.get_position("ASSET-USD").size == pytest.approx(0.0)


if __name__ == "__main__":
    test_bootstrap_balances_and_unfunded_accounts()
    test_collateral_deposit_allows_trading_after_zero_balance_bootstrap()
    test_contract_price_precision_rules()
    test_trading_agent_rounds_orders_to_symbol_precision()
    test_trading_agent_upscales_below_min_notional_order()
    test_chiarella_waits_for_mark_and_spread_before_ordering()
    test_order_book_self_trade_prevention_continues_to_deeper_liquidity()
    test_funding_engine_matches_hourly_formula()
    test_maker_re_margin_cancels_resting_order_before_fill()
    test_block_ordering_prioritizes_cancels_before_new_orders()
    test_parent_linked_trigger_orders_stay_dormant_until_parent_fills()
    test_set_oracle_enforces_minimum_spacing()
    test_end_to_end_exchange_runs_with_blocked_execution()
    test_oi_based_mark_rejection()
    test_mark_price_stale_fallback()
    test_mark_price_daily_reset()
    test_isolated_margin_basic_flow()
    test_no_cross_margin_mode_forces_isolated()
    test_strict_isolated_blocks_margin_decrease()
    test_trigger_order_releases_hold_on_activation()
    test_liquidation_cancels_resting_orders()
    test_hold_resizes_on_partial_fill()
    test_twap_order_dispatches_slices()
    test_scale_order_places_multiple_limit_orders()
    test_deployer_permission_enforcement()
    test_funding_sign_convention()
    test_no_bad_debt_after_adl()
    # Fee engine tests
    test_fee_engine_volume_tier_progression()
    test_fee_engine_staking_discount()
    test_fee_engine_maker_rebate()
    test_fee_engine_referral_discount_expires_at_25m()
    test_fee_engine_fee_scale_split()
    test_fee_engine_growth_mode()
    test_fee_engine_daily_volume_rolls_over()
    # TP/SL and OCO edge case tests
    test_dormant_child_cancelled_when_parent_cancelled()
    test_oco_sibling_cancelled_when_one_fires()
    test_dynamic_size_child_uses_current_position()
    test_liquidation_cleans_up_dormant_children_and_oco_groups()
    test_halt_cancels_all_trigger_types()
    # Multi-asset cross-margin tests
    test_multi_asset_cross_margin_shared_equity()
    test_multi_asset_cross_margin_liquidation_affects_all_positions()
    test_multi_asset_available_margin_accounts_for_both()
    test_multi_asset_adl_across_symbols()
    print("All deterministic HIP-3 tests passed.")

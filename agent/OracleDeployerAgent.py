"""HIP-3 Oracle Deployer Agent.

Acts as the HIP-3 deployer, reading from the ABIDES oracle system (CsvOracle or any
other oracle) and pushing SET_ORACLE messages to the PerpExchangeAgent at a configurable
interval (default 3 seconds, matching Hyperliquid validator oracle update frequency).

Also supports issuing runtime deployer control messages:
  - HALT_TRADING, SET_OI_CAPS, SET_FUNDING_MULTIPLIERS, SET_MARGIN_TABLE
"""

from agent.Agent import Agent
from agent.PerpExchangeAgent import PerpExchangeAgent
from message.Message import Message
from util.ContractSpec import DeployerPermission
from util.util import log_print

import pandas as pd
import math


class OracleDeployerAgent(Agent):

    def __init__(self, id, name, type, symbols, oracle_update_interval_s=3.0,
                 deployer_mark_px_mode="none", external_perp_px_mode="ema_of_mark",
                 random_state=None, log_to_file=True):
        super().__init__(id, name, type, random_state, log_to_file)

        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.oracle_update_interval_ns = int(oracle_update_interval_s * 1e9)
        self.deployer_mark_px_mode = deployer_mark_px_mode
        self.external_perp_px_mode = external_perp_px_mode

        self.exchangeID = None
        self.oracle = None

        self.mark_prices = {}
        self.oracle_prices = {}

        # EMA state for mark-based externalPerpPxs
        self._ema_states = {}  # symbol -> (numerator, denominator, last_time)
        self._ema_tau_s = 150.0  # 2.5 minutes

    def kernelInitializing(self, kernel):
        super().kernelInitializing(kernel)
        self.oracle = self.kernel.oracle

    def kernelStarting(self, startTime):
        self.exchangeID = self.kernel.findAgentByType(PerpExchangeAgent)
        log_print("OracleDeployerAgent {} found exchange ID: {}", self.id, self.exchangeID)

        # Schedule first oracle update
        self.setWakeup(startTime + pd.Timedelta(self.oracle_update_interval_ns))

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if self.external_perp_px_mode == "ema_of_mark":
            for symbol in self.symbols:
                self.sendMessage(self.exchangeID, Message({"msg": "QUERY_MARK_PRICE", "sender": self.id, "symbol": symbol}))
        self._push_oracle(currentTime)

        # Schedule next update
        self.setWakeup(currentTime + pd.Timedelta(self.oracle_update_interval_ns))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body.get("msg") == "QUERY_MARK_PRICE":
            symbol = msg.body["symbol"]
            if msg.body.get("mark_price") is not None:
                self.mark_prices[symbol] = msg.body["mark_price"]
            if msg.body.get("oracle_price") is not None:
                self.oracle_prices[symbol] = msg.body["oracle_price"]

    def _push_oracle(self, currentTime):
        """Read oracle prices and push SET_ORACLE to the exchange."""
        oracle_pxs = {}
        for symbol in self.symbols:
            try:
                px = self.oracle.observePrice(symbol, currentTime, sigma_n=0, random_state=self.random_state)
                oracle_pxs[symbol] = float(px)
            except Exception as e:
                log_print("OracleDeployerAgent: error reading oracle for {}: {}", symbol, e)

        if not oracle_pxs:
            return

        # Determine deployer mark price inputs (per-symbol)
        mark_pxs = {}
        if self.deployer_mark_px_mode == "oracle_based":
            for symbol in self.symbols:
                if symbol in oracle_pxs:
                    mark_pxs[symbol] = [oracle_pxs[symbol]]
        # "none" mode: mark_pxs stays empty (mark = local book only)
        # "custom" mode: subclasses can override _get_custom_mark_pxs

        # Determine externalPerpPxs
        external_perp_pxs = {}
        if self.external_perp_px_mode == "ema_of_mark":
            for symbol in self.symbols:
                mark_sample = self.mark_prices.get(symbol, oracle_pxs.get(symbol))
                if mark_sample is not None:
                    external_perp_pxs[symbol] = self._update_ema(symbol, mark_sample, currentTime)
        # "none" mode: external_perp_pxs stays empty

        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_ORACLE",
            "sender": self.id,
            "oracle_pxs": oracle_pxs,
            "mark_pxs": mark_pxs,
            "external_perp_pxs": external_perp_pxs,
        }))

    def _update_ema(self, symbol, sample, current_time):
        """Update EMA for externalPerpPxs fallback."""
        if symbol not in self._ema_states:
            self._ema_states[symbol] = (sample * 1.0, 1.0, current_time)
            return sample

        numerator, denominator, last_time = self._ema_states[symbol]
        t_s = max(0.001, (current_time - last_time).value / 1e9)
        decay = math.exp(-t_s / self._ema_tau_s)
        numerator = numerator * decay + sample * t_s
        denominator = denominator * decay + t_s
        self._ema_states[symbol] = (numerator, denominator, current_time)

        return numerator / denominator if denominator > 0 else sample

    # ── Runtime deployer control methods ────────────────────────────────

    def haltTrading(self, symbol, is_halted=True):
        self.sendMessage(self.exchangeID, Message({
            "msg": "HALT_TRADING",
            "sender": self.id,
            "symbol": symbol,
            "is_halted": is_halted,
        }))

    def setOiCaps(self, symbol, notional_cap, size_cap):
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_OI_CAPS",
            "sender": self.id,
            "symbol": symbol,
            "notional_cap": notional_cap,
            "size_cap": size_cap,
        }))

    def setFundingMultipliers(self, multipliers):
        """multipliers: dict of symbol -> float (0 to 10)"""
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_FUNDING_MULTIPLIERS",
            "sender": self.id,
            "multipliers": multipliers,
        }))

    def setMarginTable(self, symbol, tiers):
        """tiers: list of dicts with keys 'lower_bound_notional', 'max_leverage'"""
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_MARGIN_TABLE",
            "sender": self.id,
            "symbol": symbol,
            "tiers": tiers,
        }))

    def setFundingInterestRates(self, rates):
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_FUNDING_INTEREST_RATES",
            "sender": self.id,
            "rates": rates,
        }))

    def setFeeScale(self, fee_scale):
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_FEE_SCALE",
            "sender": self.id,
            "fee_scale": fee_scale,
        }))

    def setGrowthModes(self, growth_modes):
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_GROWTH_MODES",
            "sender": self.id,
            "growth_modes": growth_modes,
        }))

    def insertMarginTable(self, margin_table_id, tiers, description=""):
        self.sendMessage(self.exchangeID, Message({
            "msg": "INSERT_MARGIN_TABLE",
            "sender": self.id,
            "margin_table_id": margin_table_id,
            "tiers": tiers,
            "description": description,
        }))

    def setMarginTableIds(self, margin_table_ids):
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_MARGIN_TABLE_IDS",
            "sender": self.id,
            "margin_table_ids": margin_table_ids,
        }))

    def setSubDeployers(self, permissions):
        permission_map = {}
        for agent_id, variants in permissions.items():
            if isinstance(variants, DeployerPermission):
                permission_map[agent_id] = variants
            else:
                permission_map[agent_id] = DeployerPermission(variants=list(variants))
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_SUB_DEPLOYERS",
            "sender": self.id,
            "permissions": permission_map,
        }))

    def setPerpAnnotation(self, symbol, annotation):
        self.sendMessage(self.exchangeID, Message({
            "msg": "SET_PERP_ANNOTATION",
            "sender": self.id,
            "symbol": symbol,
            "annotation": annotation,
        }))

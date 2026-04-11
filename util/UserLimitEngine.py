"""Per-user rate limits and open-order caps for venue realism.

Models Hyperliquid-style action budgets:
- Rolling action-per-minute budget
- Rolling cancel-per-minute budget
- Open-order cap (default 128, expandable to 2048)
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Set

import pandas as pd


@dataclass
class UserLimits:
    max_actions_per_minute: int = 1000
    max_cancels_per_minute: int = 500
    max_open_orders: int = 128
    expanded_open_orders: int = 2048


class UserLimitEngine:
    """Tracks and enforces per-user rate limits and open-order caps."""

    def __init__(self, limits: Optional[UserLimits] = None, enabled: bool = True):
        self.limits = limits or UserLimits()
        self.enabled = enabled
        self._action_times: Dict[int, Deque[pd.Timestamp]] = defaultdict(lambda: deque())
        self._cancel_times: Dict[int, Deque[pd.Timestamp]] = defaultdict(lambda: deque())
        self._open_order_counts: Dict[int, int] = defaultdict(int)
        self._expanded_users: Set[int] = set()

    def _purge_old(self, dq: Deque[pd.Timestamp], cutoff: pd.Timestamp) -> None:
        while dq and dq[0] < cutoff:
            dq.popleft()

    def check_action(self, agent_id: int, current_time: pd.Timestamp) -> Optional[str]:
        """Return reject reason if action budget is exhausted, else None."""
        if not self.enabled:
            return None
        cutoff = current_time - pd.Timedelta(minutes=1)
        dq = self._action_times[agent_id]
        self._purge_old(dq, cutoff)
        if len(dq) >= self.limits.max_actions_per_minute:
            return "RATE_LIMIT"
        return None

    def check_cancel(self, agent_id: int, current_time: pd.Timestamp) -> Optional[str]:
        """Return reject reason if cancel budget is exhausted, else None."""
        if not self.enabled:
            return None
        cutoff = current_time - pd.Timedelta(minutes=1)
        dq = self._cancel_times[agent_id]
        self._purge_old(dq, cutoff)
        if len(dq) >= self.limits.max_cancels_per_minute:
            return "CANCEL_RATE_LIMIT"
        return None

    def check_open_orders(self, agent_id: int) -> Optional[str]:
        """Return reject reason if open-order cap is reached, else None."""
        if not self.enabled:
            return None
        cap = self.limits.expanded_open_orders if agent_id in self._expanded_users else self.limits.max_open_orders
        if self._open_order_counts[agent_id] >= cap:
            return "OPEN_ORDER_CAP"
        return None

    def record_action(self, agent_id: int, current_time: pd.Timestamp) -> None:
        self._action_times[agent_id].append(current_time)

    def record_cancel(self, agent_id: int, current_time: pd.Timestamp) -> None:
        self._cancel_times[agent_id].append(current_time)

    def record_order_placed(self, agent_id: int) -> None:
        self._open_order_counts[agent_id] += 1

    def record_order_removed(self, agent_id: int) -> None:
        self._open_order_counts[agent_id] = max(0, self._open_order_counts[agent_id] - 1)

    def grant_expanded_limits(self, agent_id: int) -> None:
        self._expanded_users.add(agent_id)

    def revoke_expanded_limits(self, agent_id: int) -> None:
        self._expanded_users.discard(agent_id)

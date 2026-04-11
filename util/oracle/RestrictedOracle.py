"""Oracle wrapper that enforces information boundaries for agent access.

Supports three access modes:
  - "strict":       clamps queries to current simulation time (default)
  - "preview":      allows querying up to lead_time_ms ahead, optionally with noise
  - "unrestricted": full pass-through access (for deployers, stress agents)
"""

from math import sqrt

import numpy as np
import pandas as pd


class RestrictedOracle:
    """Wraps a CsvOracle or MultiCsvOracle with time-boundary enforcement."""

    def __init__(self, inner_oracle, mode="strict", lead_time_ms=0, noise_sigma=0.0):
        """
        Args:
            inner_oracle: The underlying oracle (CsvOracle or MultiCsvOracle).
            mode: One of "strict", "preview", or "unrestricted".
            lead_time_ms: For "preview" mode, how far ahead (in ms) the agent may query.
            noise_sigma: For "preview" mode, noise variance applied to preview observations.
        """
        self.inner = inner_oracle
        self.mode = mode
        self.lead_time_ms = lead_time_ms
        self.noise_sigma = noise_sigma
        self._sim_time = None

        # Proxy f_log for compatibility
        self.f_log = getattr(inner_oracle, "f_log", {})

    def update_sim_time(self, t: pd.Timestamp) -> None:
        """Called by the kernel each message dispatch to set the current simulation time."""
        self._sim_time = t

    def getDailyOpenPrice(self, symbol, mkt_open):
        return self.inner.getDailyOpenPrice(symbol, mkt_open)

    def observePrice(self, symbol, currentTime, sigma_n=0, random_state=None):
        """Observe oracle price with time-boundary enforcement.

        In strict mode, clamps currentTime to simulation time.
        In preview mode, allows up to lead_time_ms ahead with optional noise.
        In unrestricted mode, passes through directly.
        """
        if self.mode == "unrestricted" or self._sim_time is None:
            return self.inner.observePrice(symbol, currentTime, sigma_n=sigma_n, random_state=random_state)

        if self.mode == "strict":
            clamped_time = min(currentTime, self._sim_time) if currentTime > self._sim_time else currentTime
            return self.inner.observePrice(symbol, clamped_time, sigma_n=sigma_n, random_state=random_state)

        if self.mode == "preview":
            max_allowed = self._sim_time + pd.Timedelta(milliseconds=self.lead_time_ms)
            clamped_time = min(currentTime, max_allowed) if currentTime > max_allowed else currentTime
            price = self.inner.observePrice(symbol, clamped_time, sigma_n=sigma_n, random_state=random_state)

            # Apply preview noise if configured
            if self.noise_sigma > 0 and random_state is not None:
                price = random_state.normal(loc=price, scale=sqrt(self.noise_sigma) * price)

            return price

        # Fallback: treat unknown modes as strict
        clamped_time = min(currentTime, self._sim_time) if currentTime > self._sim_time else currentTime
        return self.inner.observePrice(symbol, clamped_time, sigma_n=sigma_n, random_state=random_state)

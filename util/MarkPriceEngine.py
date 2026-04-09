"""HIP-3 mark price engine.

Mark price = median(deployer_mark_pxs + [local_book_mark]).

Implements:
- 1% per-update clamping
- 10x start-of-day clamping
- OI-cap interaction guard
- 10-second staleness fallback to local book mark
"""

from statistics import median
import math


class MarkPriceEngine:

    def __init__(self, initial_price: float):
        self.mark_price = initial_price
        self.oracle_price = initial_price
        self.start_of_day_price = initial_price
        self.last_update_time = None
        self.stale_threshold_ns = 10_000_000_000  # 10 seconds in nanoseconds

        # EMA state for externalPerpPxs when mode is "ema_of_mark"
        self._ema_numerator = initial_price * 1.0
        self._ema_denominator = 1.0
        self._ema_tau_s = 150.0  # 2.5 minutes

    def update(self, oracle_px, deployer_mark_pxs, local_mark_px,
               current_time, oi_notional=0.0, oi_cap=float('inf'),
               external_perp_px=None):
        """Full setOracle update cycle.
        
        Args:
            oracle_px: The oracle/index price from the CSV feed.
            deployer_mark_pxs: List of 0, 1, or 2 external mark price inputs from deployer.
            local_mark_px: median(best_bid, best_ask, last_trade) from the order book.
            current_time: pd.Timestamp of the update.
            oi_notional: Current open interest in notional terms.
            oi_cap: Deployer-configured OI cap.
            external_perp_px: Optional external perp price for safety clamping.
            
        Returns:
            The new mark price, or None if update was rejected.
        """
        self.oracle_price = oracle_px

        # Compute candidate mark price
        all_inputs = list(deployer_mark_pxs)
        if local_mark_px is not None:
            all_inputs.append(local_mark_px)

        if not all_inputs:
            return self.mark_price

        raw_mark = median(all_inputs)

        # Apply 1% clamping from previous mark
        clamped = self._clamp_1pct(raw_mark)

        # Apply 10x start-of-day cap
        clamped = min(clamped, self.start_of_day_price * 10.0)
        clamped = max(clamped, self.start_of_day_price / 10.0)

        # HIP-3 safety clamp: mark must stay within 20% of external perp price
        if external_perp_px is not None and external_perp_px > 0:
            clamped = max(external_perp_px * 0.80, min(external_perp_px * 1.20, clamped))

        # OI-cap interaction: reject if OI * new_mark > 10 * oi_cap
        if oi_cap < float('inf') and oi_notional > 0:
            projected_oi = oi_notional * (clamped / self.mark_price) if self.mark_price > 0 else 0
            if projected_oi > 10.0 * oi_cap:
                return None

        self.mark_price = clamped
        self.last_update_time = current_time

        # Update EMA for externalPerpPxs fallback
        self._update_ema(clamped, current_time)

        return self.mark_price

    def check_staleness(self, current_time, local_mark_px):
        """If no setOracle for 10 seconds, fall back to local book mark."""
        if self.last_update_time is None:
            return

        elapsed_ns = (current_time - self.last_update_time).value
        if elapsed_ns > self.stale_threshold_ns and local_mark_px is not None:
            self.mark_price = self._clamp_1pct(local_mark_px)
            self.last_update_time = current_time

    def reset_start_of_day(self, price=None):
        if price is not None:
            self.start_of_day_price = price
        else:
            self.start_of_day_price = self.mark_price

    def get_external_perp_px_ema(self):
        """EMA of recent mark prices, used as externalPerpPxs fallback."""
        if self._ema_denominator <= 0:
            return self.mark_price
        return self._ema_numerator / self._ema_denominator

    def _clamp_1pct(self, candidate):
        if self.mark_price <= 0:
            return candidate
        low = self.mark_price * 0.99
        high = self.mark_price * 1.01
        return max(low, min(high, candidate))

    def _update_ema(self, sample, current_time):
        if self.last_update_time is None:
            return
        t_s = max(0.001, (current_time - self.last_update_time).value / 1e9)
        decay = math.exp(-t_s / self._ema_tau_s)
        self._ema_numerator = self._ema_numerator * decay + sample * t_s
        self._ema_denominator = self._ema_denominator * decay + t_s

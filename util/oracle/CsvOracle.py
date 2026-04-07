import numpy as np
import pandas as pd
from math import sqrt
from util.util import log_print


class CsvOracle:
    """Oracle that reads (timestamp, price) pairs from a CSV file.
    
    Designed for sub-second resolution data with potentially millions of rows.
    Uses numpy searchsorted for O(log n) lookups.
    
    CSV format: two columns, first is timestamp (Unix epoch float or ISO datetime),
    second is price (float). Header row is optional and auto-detected.
    """

    def __init__(self, symbol, csv_path, timestamp_col=0, price_col=1, has_header=True):
        self.symbol = symbol
        self.csv_path = csv_path
        self.f_log = {symbol: []}

        df = pd.read_csv(csv_path, header=0 if has_header else None)
        ts_raw = df.iloc[:, timestamp_col]
        prices_raw = df.iloc[:, price_col].values.astype(np.float64)

        if pd.api.types.is_numeric_dtype(ts_raw):
            timestamps = pd.to_datetime(ts_raw, unit='s', utc=True).values
        else:
            timestamps = pd.to_datetime(ts_raw).values

        sort_idx = np.argsort(timestamps)
        self.timestamps = timestamps[sort_idx]
        self.prices = prices_raw[sort_idx]

        log_print("CsvOracle: loaded {} data points for {} from {}", len(self.prices), symbol, csv_path)
        log_print("CsvOracle: time range {} to {}", self.timestamps[0], self.timestamps[-1])

    def getDailyOpenPrice(self, symbol, mkt_open):
        if symbol != self.symbol:
            log_print("CsvOracle: unknown symbol {}", symbol)
            return 0.0
        return self._price_at_time(mkt_open)

    def observePrice(self, symbol, currentTime, sigma_n=0, random_state=None):
        if symbol != self.symbol:
            return 0.0

        true_price = self._price_at_time(currentTime)

        if sigma_n > 0 and random_state is not None:
            observed = random_state.normal(loc=true_price, scale=sqrt(sigma_n) * true_price)
        else:
            observed = true_price

        self.f_log[symbol].append({'FundamentalTime': currentTime, 'FundamentalValue': observed})
        return observed

    def _price_at_time(self, query_time):
        """Return the most recent price at or before query_time via binary search."""
        ts = np.datetime64(pd.Timestamp(query_time))
        idx = np.searchsorted(self.timestamps, ts, side='right') - 1

        if idx < 0:
            return float(self.prices[0])
        if idx >= len(self.prices):
            return float(self.prices[-1])

        return float(self.prices[idx])

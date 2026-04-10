"""Composite oracle backed by one CSV oracle per symbol."""

from typing import Dict

from util.oracle.CsvOracle import CsvOracle


class MultiCsvOracle:
    """Route oracle requests to symbol-specific CsvOracle instances."""

    def __init__(self, symbol_to_csv: Dict[str, str]):
        self.oracles = {symbol: CsvOracle(symbol, path) for symbol, path in symbol_to_csv.items()}
        self.f_log = {symbol: oracle.f_log[symbol] for symbol, oracle in self.oracles.items()}

    def getDailyOpenPrice(self, symbol, mkt_open):
        oracle = self.oracles.get(symbol)
        if oracle is None:
            return 0.0
        return oracle.getDailyOpenPrice(symbol, mkt_open)

    def observePrice(self, symbol, currentTime, sigma_n=0, random_state=None):
        oracle = self.oracles.get(symbol)
        if oracle is None:
            return 0.0
        return oracle.observePrice(symbol, currentTime, sigma_n=sigma_n, random_state=random_state)

import pandas as pd
import numpy as np


class TradingRule:
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.signal = None
        self.bisignal = None
        self.pnl = None

    def set_pnl(self) -> None:
        logret = np.log(self.close).diff().shift(-1)
        pnl = self.bisignal * logret
        self.pnl = pnl
        

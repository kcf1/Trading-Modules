from __future__ import annotations

import pandas as pd
import numpy as np

from .indicators import Indicator
from .clean_data import clean, normalize
from scipy.stats import linregress
from numpy.lib.stride_tricks import sliding_window_view
from math import sqrt


class TechnicalIndicator(Indicator):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__()
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.sort()

    def sort(self) -> None:
        if self.open is not None:
            self.open.sort_index(inplace=True)
        if self.high is not None:
            self.high.sort_index(inplace=True)
        if self.low is not None:
            self.low.sort_index(inplace=True)
        if self.close is not None:
            self.close.sort_index(inplace=True)
        if self.volume is not None:
            self.volume.sort_index(inplace=True)

    def get_ewvol(self, vol_lookback: int) -> pd.Series:
        """
        Volatility forecast
        vol = 0.7 * vol + 0.3 * avg_vol
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        vol = np.sqrt((log_ret**2).ewm(vol_lookback).mean())
        return clean(vol)

    def get_combvol(self, vol_lookback: int, avg_vol_lookback: int) -> pd.Series:
        """
        Volatility forecast
        vol = 0.7 * vol + 0.3 * avg_vol
        """
        assert self.close is not None, "close is required"

        vol = self.get_ewvol(vol_lookback)
        avg_vol = vol.rolling(avg_vol_lookback).mean()
        comb_vol = 0.7 * vol + 0.3 * avg_vol
        return clean(comb_vol)

    def get_ema(self, lookback: int) -> pd.Series:
        """
        Exponential moving average
        """
        assert self.close is not None, "close is required"

        ema = self.close.ewm(lookback).mean()
        return clean(ema)

    def get_bb(
        self, ema_lookback: int, vol_lookback: int, avg_vol_lookback: int, width: float
    ) -> pd.Series:
        """
        Bollinger bands
        """
        assert self.close is not None, "close is required"

        ema = self.get_ema(ema_lookback)
        vol = self.get_combvol(vol_lookback, avg_vol_lookback) * np.sqrt(vol_lookback)
        p_vol = ema * vol
        upper = ema + width * p_vol
        lower = ema - width * p_vol
        return ema, upper, lower

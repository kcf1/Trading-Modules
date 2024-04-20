import pandas as pd
import numpy as np
from indicators import Indicator


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

    def get_all(self, lookbacks: list) -> pd.DataFrame:
        assert self.open is not None, "open is required"
        assert self.high is not None, "high is required"
        assert self.low is not None, "low is required"
        assert self.close is not None, "close is required"

        df = pd.DataFrame()
        for lookback in lookbacks:
            df[f"cchv_{lookback}"] = self.get_CCHV(lookback)
            df[f"ewhv_{lookback}"] = self.get_EWHV(lookback)
            df[f"atr_{lookback}"] = self.get_ATR(lookback)
            df[f"phv_{lookback}"] = self.get_PHV(lookback)
            df[f"gkhv_{lookback}"] = self.get_GKHV(lookback)
            df[f"rshv_{lookback}"] = self.get_RSHV(lookback)
            df[f"avghv_{lookback}"] = self.get_AVGHV(lookback)

            df[f"ewmac_{lookback}"] = self.get_EWMAC(lookback)
            df[f"cbo_{lookback}"] = self.get_CBO(lookback)
            df[f"npm_{lookback}"] = self.get_NPM(lookback)
            df[f"bbz_{lookback}"] = self.get_BBZ(lookback)
            df[f"skp_{lookback}"] = self.get_SKP(lookback)

            df[f"volz_{lookback}"] = self.get_VOLZ(lookback)
        df.ffill(inplace=True)
        return df

    def get_CCHV(self, lookback: int) -> pd.Series:
        """
        Close-to-close volatility
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        cchv = np.sqrt((log_ret**2).rolling(lookback).mean())
        return cchv

    def get_EWHV(self, lookback: int) -> pd.Series:
        """
        Exponentially-weighted historical volatility
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        ewhv = np.sqrt((log_ret**2).ewm(lookback, min_periods=lookback).mean())
        return ewhv

    def get_ATR(self, lookback: int) -> pd.Series:
        """
        Average true range
        """
        assert self.high is not None, "high is required"
        assert self.low is not None, "low is required"
        assert self.close is not None, "close is required"

        log_hl = np.log(self.high / self.low)
        log_hc = np.log(self.high / self.close.shift(1))
        log_lc = np.log(self.low / self.close.shift(1))
        tr = pd.concat([log_hl, log_hc, log_lc], axis=1).abs().max(axis=1)
        atr = tr.rolling(lookback).mean()
        return atr

    def get_PHV(self, lookback: int) -> pd.Series:
        """
        Parkinson historical volatility
        """
        assert self.high is not None, "high is required"
        assert self.low is not None, "low is required"

        k = 1 / (4 * np.log(2))
        log_hl = np.log(self.high / self.low)
        mean_squared_hl = (log_hl**2).rolling(lookback).mean()
        phv = np.sqrt(k * mean_squared_hl)
        return phv

    def get_GKHV(self, lookback: int) -> pd.Series:
        """
        Garman-Klass historical volatility
        """
        assert self.open is not None, "open is required"
        assert self.high is not None, "high is required"
        assert self.low is not None, "low is required"
        assert self.close is not None, "close is required"

        log_hl = np.log(self.high / self.low)
        log_co = np.log(self.close / self.open)
        mean_squared_hl = (log_hl**2).rolling(lookback).mean()
        mean_squared_co = (log_co**2).rolling(lookback).mean()
        gkhv = np.sqrt(1 / 2 * mean_squared_hl - (2 * np.log(2) - 1) * mean_squared_co)
        return gkhv

    def get_RSHV(self, lookback: int) -> pd.Series:
        """
        Rogers-Satchell historical volatility
        """
        assert self.open is not None, "open is required"
        assert self.high is not None, "high is required"
        assert self.low is not None, "low is required"
        assert self.close is not None, "close is required"

        log_hc = np.log(self.high / self.close)
        log_ho = np.log(self.high / self.open)
        log_lc = np.log(self.low / self.close)
        log_lo = np.log(self.low / self.open)
        rshv = np.sqrt((log_hc * log_ho + log_lc * log_lo).rolling(lookback).mean())
        return rshv

    def get_AVGHV(self, lookback: int = 120) -> pd.Series:
        """
        Equal-weighted average of CCHV, EWHV, ATR, PHV, GKHV, RSHV

        Default lookback = 120 ~ 5 days (considering converging time)
        """
        cchv = self.get_CCHV(lookback)
        ewhv = self.get_EWHV(lookback)
        atr = self.get_ATR(lookback)
        phv = self.get_PHV(lookback)
        gkhv = self.get_GKHV(lookback)
        rshv = self.get_RSHV(lookback)
        avghv = (cchv + ewhv + atr + phv + gkhv + rshv) / 6
        return avghv

    def get_EWMAC(self, lookback: int) -> pd.Series:
        """
        Exponentially-weighted moving average crossover

        Slow MA = 4 x Fast MA
        """
        assert self.close is not None, "close is required"

        vol = self.get_AVGHV()
        fast_ma = self.close.ewm(lookback, min_periods=lookback).mean()
        slow_ma = self.close.ewm(lookback * 4, min_periods=lookback * 4).mean()
        ewmac = (fast_ma - slow_ma) / vol
        return ewmac

    def get_CBO(self, lookback: int) -> pd.Series:
        """
        Channel breakout

        Only close price to avoid extreme values
        """
        assert self.close is not None, "close is required"

        upper = self.close.rolling(lookback).max()
        lower = self.close.rolling(lookback).min()
        middle = (upper + lower) / 2
        channel = upper - lower
        cbo = (self.close - middle) / channel
        return cbo

    def get_NPM(self, lookback: int) -> pd.Series:
        """
        N-period momentum
        """
        assert self.close is not None, "close is required"

        vol = self.get_AVGHV()
        log_ret = np.log(self.close).diff(lookback)
        period_ret = log_ret / lookback
        npm = period_ret / vol
        return npm

    def get_BBZ(self, lookback: int) -> pd.Series:
        """
        Bollinger band z-score
        """
        assert self.close is not None, "close is required"

        p_vol = self.close * self.get_AVGHV(lookback)
        ma = self.close.rolling(lookback).mean()
        bbz = (self.close - ma) / p_vol
        return bbz

    def get_SKP(self, lookback: int) -> pd.Series:
        """
        Skewness premium
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        skp = log_ret.rolling(lookback).skew().ffill()
        return skp

    def get_CMR(self, mean_ret: pd.Series) -> pd.Series:
        """
        Cross-sectional mean reversion
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        cmr = (log_ret - mean_ret).expanding().rank(pct=True)
        return cmr

    def get_VOLZ(self, lookback: int) -> pd.Series:
        """
        Volume z-score
        """
        assert self.volume is not None, "volume is required"

        log_v = np.log(self.volume)
        ma = log_v.ewm(lookback, min_periods=lookback).mean()
        ew_std = np.sqrt(((log_v - ma) ** 2).ewm(lookback).mean())
        volz = (log_v - ma) / ew_std
        return volz

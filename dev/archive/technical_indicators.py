import pandas as pd
import numpy as np
from indicators import Indicator, clean, normalize
from scipy.stats import linregress
from numpy.lib.stride_tricks import sliding_window_view


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

    def get_all(self, params: dict) -> pd.DataFrame:
        assert self.open is not None, "open is required"
        assert self.high is not None, "high is required"
        assert self.low is not None, "low is required"
        assert self.close is not None, "close is required"

        df = pd.DataFrame()

        for lookback in params["ewmac"]["lookbacks"]:
            df[f"ewmac_{lookback}"] = self.get_EWMAC(lookback)
        for lookback in params["macd"]["lookbacks"]:
            df[f"macd_{lookback}"] = self.get_MACD(lookback)
        for lookback in params["dcb"]["lookbacks"]:
            df[f"dcb_{lookback}"] = self.get_DCB(lookback)
        for lookback in params["bbz"]["lookbacks"]:
            df[f"bbz_{lookback}"] = self.get_BBZ(lookback)
        for lookback in params["rsi"]["lookbacks"]:
            df[f"rsi_{lookback}"] = self.get_RSI(lookback)
        for lookback in params["skew"]["lookbacks"]:
            df[f"skew_{lookback}"] = self.get_SKEW(lookback)

        for lookback in params["volz"]["lookbacks"]:
            df[f"volz_{lookback}"] = self.get_VOLZ(lookback)

        # for lookback in params["ewhv"]["lookbacks"]:
        #    df[f"ewhv_{lookback}"] = self.get_EWHV(lookback)
        # for lookback in params["rshv"]["lookbacks"]:
        #    df[f"rshv_{lookback}"] = self.get_RSHV(lookback)
        for lookback in params["avghv"]["lookbacks"]:
            df[f"avghv_{lookback}"] = self.get_AVGHV(lookback)

        df.ffill(inplace=True)
        return df

    def get_CCHV(self, lookback: int) -> pd.Series:
        """
        Close-to-close volatility
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        cchv = np.sqrt((log_ret**2).rolling(lookback).mean())
        return clean(cchv)

    def get_EWHV(self, lookback: int) -> pd.Series:
        """
        Exponentially-weighted historical volatility
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        ewhv = np.sqrt((log_ret**2).ewm(lookback, min_periods=lookback).mean())
        return clean(ewhv)

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
        atr = tr.ewm(lookback).mean()
        return clean(atr)

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
        return clean(phv)

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
        return clean(gkhv)

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
        return clean(rshv)

    def get_AVGHV(self, lookback: int = 96) -> pd.Series:
        """
        Equal-weighted average of CCHV, EWHV, ATR, PHV, GKHV, RSHV

        Default lookback = 120 ~ 5 days (considering converging time)
        """
        ewhv = self.get_EWHV(lookback)
        rshv = self.get_RSHV(lookback)
        avghv = (ewhv + rshv) / 2
        return clean(avghv)

    def get_EWMAC(self, lookback: int) -> pd.Series:
        """
        Exponentially-weighted moving average crossover

        Slow MA = 4 x Fast MA
        """
        assert self.close is not None, "close is required"

        p_vol = self.close * self.get_AVGHV()
        fast_ma = self.close.ewm(lookback, min_periods=lookback).mean()
        slow_ma = self.close.ewm(lookback * 4, min_periods=lookback * 4).mean()
        ewmac = (fast_ma - slow_ma) / p_vol
        ewmac = normalize(ewmac, method="zeromean")
        return clean(ewmac)

    def get_MACD(self, lookback: int) -> pd.Series:
        """
        Moving average convergence divergence

        MAC = Fast MA - slow MA
        MACD = MAC - ema(MAC)
        """
        assert self.close is not None, "close is required"

        p_vol = self.close * self.get_AVGHV()
        fast_ma = self.close.ewm(lookback, min_periods=lookback).mean()
        slow_ma = self.close.ewm(lookback * 4, min_periods=lookback * 4).mean()
        mac = fast_ma - slow_ma
        macd = (mac - mac.ewm(lookback, min_periods=lookback).mean()) / p_vol
        macd = normalize(macd, method="zeromean")
        return clean(macd)

    def get_DCB(self, lookback: int) -> pd.Series:
        """
        Donchian channel breakout
        """
        assert self.close is not None, "close is required"

        upper = self.close.rolling(lookback).max()
        lower = self.close.rolling(lookback).min()
        middle = (upper + lower) / 2
        channel = upper - lower
        cbo = (self.close - middle) / channel
        cbo = cbo * 2
        return clean(cbo)

    def get_TRT(self, lookback: int) -> pd.Series:
        """
        Time regression t-score

        t = beta / stderr
        """
        assert self.close is not None, "close is required"

        def ols_time(log_prc):
            y, x = log_prc, np.arange(lookback)
            reg = linregress(x, y)
            beta, stderr = reg[0], reg[4]
            t_score = beta / stderr
            return t_score

        # vectorize approach to run rolling OLS

        # create rolling datasets for each day
        log_prc = np.log(self.close)
        rolling = sliding_window_view(log_prc, window_shape=lookback)
        # apply OLS function on each dataset
        trt = np.apply_along_axis(ols_time, axis=1, arr=rolling)
        trt = pd.Series(trt, index=log_prc.index[lookback - 1 :]) / np.sqrt(lookback)
        trt = trt / 2
        return clean(trt)

    def get_BBZ(self, lookback: int) -> pd.Series:
        """
        Bollinger band z-score
        """
        assert self.close is not None, "close is required"

        p_vol = self.close * self.get_AVGHV()
        ma = self.close.ewm(lookback).mean()
        bbz = (self.close - ma) / p_vol
        bbz = normalize(bbz, method="zeromean")
        return clean(bbz)

    def get_RSI(self, lookback: int) -> pd.Series:
        """
        Relative strength index
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        gain, loss = log_ret.copy(), log_ret.copy()
        gain.loc[gain < 0] = 0
        loss.loc[loss > 0] = 0
        avg_gain = gain.rolling(lookback).mean()
        avg_loss = -loss.rolling(lookback).mean()
        rs = avg_gain / avg_loss
        rsi = 1 - 1 / (1 + rs)
        rsi = (rsi - 0.5) * 2
        return clean(rsi)

    def get_SKEW(self, lookback: int) -> pd.Series:
        """
        Skewness
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        skew = log_ret.rolling(lookback).skew()
        return clean(skew)

    def get_CMR(self, mean_ret: pd.Series) -> pd.Series:
        """
        Cross-sectional mean reversion
        """
        assert self.close is not None, "close is required"

        log_ret = np.log(self.close).diff()
        cmr = (log_ret - mean_ret).rank(pct=True)
        return clean(cmr)

    def get_VOLZ(self, lookback: int) -> pd.Series:
        """
        Volume z-score
        """
        assert self.volume is not None, "volume is required"

        log_v = np.log(self.volume)
        ma = log_v.ewm(lookback, min_periods=lookback).mean()
        ew_std = np.sqrt(((log_v - ma) ** 2).ewm(lookback).mean())
        volz = (log_v - ma) / ew_std
        return clean(volz)

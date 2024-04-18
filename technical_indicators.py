import pandas as pd
import numpy as np
from indicators import Indicator


class TechnicalIndicator(Indicator):
    def __init__(
        self,
        name: str = None,
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
        
    def get_CCVOL(self,lookback:int)->pd.Series:
        '''
        Close-to-close volatility
        '''
        log_ret = np.log(self.close).diff()
        ccvol = log_ret.rolling(lookback).std()
        return ccvol
    
    def get_EWVOL(self,lookback:int)->pd.Series:
        '''
        Exponentially-weighted volatility
        '''
        log_ret = np.log(self.close).diff()
        ewvol = np.sqrt((log_ret**2).ewm(lookback, min_periods=lookback).mean())
        return ewvol
    def get_PVOL(self,lookback:int)->pd.Series:
        '''
        Parkinson volatility
        '''
        f = 1/(4*lookback*np.log(2))
        log_ret = np.log(self.high/self.low)
        pvol = np.sqrt(f*(log_ret**2).sum())
        return pvol
        
    def get_EWMAC(self):
        


class EWMAC(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        self.lookback = lookback
        self.name = f"EWMAC_{lookback}"
        self.set_signal()

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        fast = log_close.ewm(lookback, min_periods=lookback).mean()
        slow = log_close.ewm(lookback * 4, min_periods=lookback * 4).mean()
        # approximate 0.05 alpha
        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())

        ewmac = ((fast - slow) / ew_vol).ffill()
        self.signal = ewmac
        self.signal.name = self.name

class ChannelBreakout(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        thr: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        assert thr is not None, "thr cannot be None"
        self.lookback = lookback
        self.thr = thr
        self.name = f"ChannelBreakout_{lookback}_{thr}"
        self.set_signal()
        self.set_bisignal()
        self.set_pnl()

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        lookback = self.lookback

        upper = log_close.rolling(lookback).max()
        lower = log_close.rolling(lookback).min()

        channel_pos = ((log_close - lower) / (upper - lower) - 0.5).ffill()
        channel_pos
        self.signal = channel_pos
        self.signal.name = self.name

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal >= 1 - thr, 1, 0)
        short_signal = np.where(signal <= thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal
        self.bisignal.name = self.name


# Deprecated
"""
class LongFilter(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        thr: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        assert thr is not None, "thr cannot be None"
        self.lookback = lookback
        self.thr = thr
        self.set_signal()
        self.signal.name = f"LongFilter_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"LongFilter_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"LongFilter_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())
        up = log_close - log_close.rolling(lookback).min()

        vol_up = (up / ew_vol).ffill()
        self.signal = vol_up

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        bisignal = pd.Series(np.where(signal >= thr, 1, 0), index=signal.index)
        self.bisignal = bisignal


class ShortFilter(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        thr: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        assert thr is not None, "thr cannot be None"
        self.lookback = lookback
        self.thr = thr
        self.set_signal()
        self.signal.name = f"ShortFilter_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"ShortFilter_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"ShortFilter_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())
        down = log_close.rolling(lookback).max() - log_close

        vol_down = (down / ew_vol).ffill()
        self.signal = vol_down

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        bisignal = pd.Series(np.where(signal >= thr, -1, 0), index=signal.index)
        self.bisignal = bisignal
"""


class NDayMomentum(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        thr: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        assert thr is not None, "thr cannot be None"
        self.lookback = lookback
        self.thr = thr
        self.name = f"NDayMomentum_{lookback}_{thr}"
        self.set_signal()
        self.set_bisignal()
        self.set_pnl()

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())

        vol_change = (log_close.diff(lookback) / ew_vol).ffill()
        self.signal = vol_change
        self.signal.name = self.name

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal >= thr, 1, 0)
        short_signal = np.where(signal <= -thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal
        self.bisignal.name = self.name


class BollingerBand(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        thr: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        assert thr is not None, "thr cannot be None"
        self.lookback = lookback
        self.thr = thr
        self.name = f"BollingerBand_{lookback}_{thr}"
        self.set_signal()
        self.set_bisignal()
        self.set_pnl()

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())
        dev = log_close - log_close.ewm(lookback).mean()

        z_score = (dev / ew_vol).ffill()
        self.signal = z_score
        self.signal.name = self.name

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= -thr, 1, 0)
        short_signal = np.where(signal >= thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal
        self.bisignal.name = self.name


class SkewPremium(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        thr: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        assert thr is not None, "thr cannot be None"
        self.lookback = lookback
        self.thr = thr
        self.name = f"SkewPremium_{lookback}_{thr}"
        self.set_signal()
        self.set_bisignal()
        self.set_pnl()

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        lookback = self.lookback

        skew = (
            log_close.diff()
            .rolling(lookback)
            .skew()
            .rolling(lookback * 4)
            .rank(pct=True)
        ).ffill()
        self.signal = skew
        self.signal.name = self.name

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= thr, 1, 0)
        short_signal = np.where(signal >= 1 - thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal
        self.bisignal.name = self.name


class KurtReversal(TechnicalIndicator):
    def __init__(
        self,
        lookback: int,
        thr: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        assert close is not None, "close cannot be None"
        assert lookback is not None, "lookback cannot be None"
        assert thr is not None, "thr cannot be None"
        self.lookback = lookback
        self.thr = thr
        self.name = f"KurtReversal_{lookback}_{thr}"
        self.set_signal()
        self.set_bisignal()
        self.set_pnl()

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        lookback = self.lookback

        skew = (
            log_close.diff()
            .rolling(lookback)
            .kurt()
            .rolling(lookback * 4)
            .rank(pct=True)
        ).ffill()
        self.signal = skew
        self.signal.name = self.name

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= thr, 1, 0)
        short_signal = np.where(signal >= 1 - thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal
        self.bisignal.name = self.name

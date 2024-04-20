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
        
    def get_CCHV(self,lookback:int)->np.float64:
        '''
        Close-to-close volatility
        '''
        log_ret = np.log(self.close).diff()
        cchv = np.sqrt((log_ret**2).mean())
        return cchv
    
    def get_EWHV(self,lookback:int)->pd.Series:
        '''
        Exponentially-weighted historical volatility
        '''
        log_ret = np.log(self.close).diff()
        ewhv = np.sqrt((log_ret**2).ewm(lookback, min_periods=lookback).mean())
        return ewhv
    
    def get_PHV(self,lookback:int)->pd.Series:
        '''
        Parkinson historical volatility
        '''
        k = 1/(4*np.log(2))
        log_hl = np.log(self.high/self.low)
        mean_squared_hl = (log_hl**2).rolling(lookback).mean()
        phv = np.sqrt(k*mean_squared_hl)
        return phv
    
    
    def get_GKHV(self,lookback:int)->pd.Series:
        '''
        Garman-Klass historical volatility
        '''
        log_hl = np.log(self.high/self.low)
        log_co = np.log(self.close/self.open)
        mean_squared_hl = (log_hl ** 2).rolling(lookback).mean()
        mean_squared_co = (log_co ** 2).rolling(lookback).mean()
        gkhv = np.sqrt(1/2 * mean_squared_hl - (2*np.log(2)-1) * mean_squared_co)
        return gkhv
    
    def get_RSHV(self,lookback:int)->pd.Series:
        '''
        Rogers-Satchell historical volatility
        '''
        log_hc = np.log(self.high/self.close)
        log_ho = np.log(self.high/self.open)
        log_lc = np.log(self.low/self.close)
        log_lo = np.log(self.low/self.open)
        rshv = np.sqrt((log_hc*log_ho + log_lc*log_lo).rolling(lookback).mean())
        return rshv
    
    def get_YZHV(self,lookback:int)->pd.Series:
        '''
        Yang-Zhang historical volatility
        '''
    
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

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


class EWMAC(TradingRule):
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
        self.lookback = lookback
        self.thr = thr
        self.set_signal()
        self.signal.name = f"EWMAC_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"EWMAC_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"EWMAC_{lookback}_{thr}"

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

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= -thr, 1, 0)
        short_signal = np.where(signal >= thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal


class ChannelBreakout(TradingRule):
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
        self.signal.name = f"ChannelBreakout_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"ChannelBreakout_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"ChannelBreakout_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        lookback = self.lookback

        upper = log_close.rolling(lookback).max()
        lower = log_close.rolling(lookback).min()

        channel_pos = ((log_close - lower) / (upper - lower) - 0.5).ffill()
        channel_pos
        self.signal = channel_pos

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal >= 1 - thr, 1, 0)
        short_signal = np.where(signal <= thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal


# Deprecated
"""
class LongFilter(TradingRule):
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


class ShortFilter(TradingRule):
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


class NDayMomentum(TradingRule):
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
        self.signal.name = f"NDayMomentum_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"NDayMomentum_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"NDayMomentum_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())

        vol_change = (log_close.diff(lookback) / ew_vol).ffill()
        self.signal = vol_change

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal >= thr, 1, 0)
        short_signal = np.where(signal <= -thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal


class BollingerBand(TradingRule):
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
        self.signal.name = f"BollingerBand_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"BollingerBand_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"BollingerBand_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())
        dev = log_close - log_close.ewm(lookback).mean()

        z_score = (dev / ew_vol).ffill()
        self.signal = z_score

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= -thr, 1, 0)
        short_signal = np.where(signal >= thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal


class SkewPremium(TradingRule):
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
        self.signal.name = f"SkewPremium_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"SkewPremium_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"SkewPremium_{lookback}_{thr}"

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

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= thr, 1, 0)
        short_signal = np.where(signal >= 1 - thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal


class KurtReversal(TradingRule):
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
        self.signal.name = f"KurtReversal_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"KurtReversal_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"KurtReversal_{lookback}_{thr}"

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

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= thr, 1, 0)
        short_signal = np.where(signal >= 1 - thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal

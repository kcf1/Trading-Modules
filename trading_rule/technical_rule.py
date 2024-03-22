import pandas as pd
import numpy as np
from trading_rule import TradingRule


class TechnicalRule(TradingRule):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)


class EWMAC(TechnicalRule):
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
        self.set_signal()
        self.signal.name = f"S_EWMAC_{lookback}"
        self.set_bisignal()
        self.bisignal.name = f"B_EWMAC_{lookback}"
        self.set_pnl()
        self.pnl.name = f"P_EWMAC_{lookback}"

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
        bisignal = pd.Series(np.where(signal >= 0, 1, -1), index=signal.index)
        self.bisignal = bisignal


class ChannelBreakout(TechnicalRule):
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
        self.signal.name = f"S_ChannelBreakout_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"B_ChannelBreakout_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"P_ChannelBreakout_{lookback}_{thr}"

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


class LongFilter(TechnicalRule):
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
        self.signal.name = f"S_LongFilter_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"B_LongFilter_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"P_LongFilter_{lookback}_{thr}"

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


class ShortFilter(TechnicalRule):
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
        self.signal.name = f"S_ShortFilter_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"B_ShortFilter_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"P_ShortFilter_{lookback}_{thr}"

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


class NDayMomentum(TechnicalRule):
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
        self.signal.name = f"S_NDayMomentum_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"B_NDayMomentum_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"P_NDayMomentum_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean()) * lookback**0.5

        vol_change = (log_close.diff(lookback) / ew_vol).ffill()
        self.signal = vol_change

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal >= thr, 1, 0)
        short_signal = np.where(signal <= -thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal


class BollingerBand(TechnicalRule):
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
        self.signal.name = f"S_BollingerBand_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"B_BollingerBand_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"P_BollingerBand_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        log_ret = log_close.diff()
        lookback = self.lookback

        ew_vol = np.sqrt((log_ret**2).ewm(40, min_periods=40).mean())
        dev = log_close - log_close.ewm(lookback, lookback)

        z_score = (dev / ew_vol).ffill()
        self.signal = z_score

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= -thr, 1, 0)
        short_signal = np.where(signal >= thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal


class SkewPremium(TechnicalRule):
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
        self.signal.name = f"S_RSIReversal_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"B_RSIReversal_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"P_RSIReversal_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        lookback = self.lookback

        skew = (log_close.diff().rolling(lookback).skew().rolling(lookback * 4).rank(pct=True)).ffill()
        self.signal = skew

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= thr, 1, 0)
        short_signal = np.where(signal >= 1 - thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal
        

class KurtReversal(TechnicalRule):
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
        self.signal.name = f"S_RSIReversal_{lookback}_{thr}"
        self.set_bisignal()
        self.bisignal.name = f"B_RSIReversal_{lookback}_{thr}"
        self.set_pnl()
        self.pnl.name = f"P_RSIReversal_{lookback}_{thr}"

    def set_signal(self) -> None:
        log_close = np.log(self.close)
        lookback = self.lookback

        skew = (log_close.diff().rolling(lookback).kurt().rolling(lookback * 4).rank(pct=True)).ffill()
        self.signal = skew

    def set_bisignal(self) -> None:
        signal = self.signal
        thr = self.thr

        long_signal = np.where(signal <= thr, 1, 0)
        short_signal = np.where(signal >= 1 - thr, -1, 0)

        bisignal = pd.Series(long_signal + short_signal, index=signal.index)
        self.bisignal = bisignal

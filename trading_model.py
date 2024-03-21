import pandas as pd
import numpy as np
from trading_rule import *
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


class TradingModel:
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
        self.rules_signals_df = pd.DataFrame()
        self.rules_bisignals_df = pd.DataFrame()
        self.rules_pnls_df = pd.DataFrame()
        self.label = None
        self.model = None
        self.signal = None
        self.pnl = None

    def set_pnl(self) -> None:
        logret = np.log(self.close).diff().shift(-1)
        pnl = self.bisignal * logret
        self.pnl = pnl

    def set_rules_signals(self, rule_obj: TradingRule) -> None:
        self.rules_signals_df[rule_obj.name] = rule_obj.signal
        self.rules_bisignals_df[rule_obj.name] = rule_obj.bisignal
        self.rules_pnls_df[rule_obj.name] = rule_obj.pnl

    def set_label(self) -> None:
        tp, sl, hd = 5, 5, 24
        logret = np.log(self.close).diff().shift(-1)
        ew_vol = np.sqrt((logret**2).ewm(40, min_periods=40).mean())

        label = []

        for i, ret in logret.items():
            up_bar = ew_vol.loc[i] * tp
            low_bar = ew_vol.loc[i] * -sl
            time_bar = hd

            future_ret = logret.loc[i:]
            cum_ret = ret
            cum_day = 1
            while True:
                if cum_ret >= up_bar:
                    label.append(+2)
                    break
                elif cum_ret <= low_bar:
                    label.append(np.sign(cum_ret))
                    break
                elif cum_day >= time_bar or cum_day >= future_ret.shape[0]:
                    label.append(-2)
                    break

                ret = future_ret.iloc[cum_day]
                cum_day += 1
                cum_ret += ret

        label = pd.Series(label, index=logret.index, name="LABEL")
        self.label = label

    def train_model(self) -> None:
        dataset = pd.concat([self.label, self.rules_bisignals_df], axis=1).dropna()
        train_y, train_x = dataset["LABEL"], dataset.drop(columns="LABEL")
        # model = RandomForestClassifier(
        #    n_estimators=1000, max_depth=3, random_state=300300
        # )
        model = GaussianNB()
        # model = LogisticRegression(penalty="l2", random_state=300300)
        model.fit(y=train_y, X=train_x)
        self.model = model

    def model_pred(self) -> None:
        pred = self.model.predict(self.rules_bisignals_df.iloc[-1, :].to_frame().T)[0]
        return pred


class ShortTermKurtReversal(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [12, 24]
        thrs = [0.85, 0.95, 0.99]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            kurt_rule = KurtReversal(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(kurt_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class MidTermKurtReversal(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [48, 96, 192, 384]
        thrs = [0.85, 0.95, 0.99]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            kurt_rule = KurtReversal(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(kurt_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class ShortTermBollingerBand(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [12, 24]
        thrs = [1, 2, 3]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = BollingerBand(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class MidTermBollingerBand(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [48, 96, 192]
        thrs = [1, 2, 3]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = BollingerBand(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class ShortTermChannelBreakout(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [12, 24, 48]
        thrs = [0.85, 0.95, 0.99]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = ChannelBreakout(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class MidTermChannelBreakout(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [96, 192, 384]
        thrs = [0.85, 0.95, 0.99]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = ChannelBreakout(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class ShortTermSkewPremium(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [12, 24, 48, 96]
        thrs = [0.85, 0.95, 0.99]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = SkewPremium(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class LongTermSkewPremium(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [192, 384]
        thrs = [0.85, 0.95, 0.99]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = SkewPremium(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class ShortTermNDayMomentum(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [12, 24, 48]
        thrs = [1, 2, 3]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = NDayMomentum(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class LongTermNDayMomentum(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [96, 192, 384]
        thrs = [1, 2, 3]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = NDayMomentum(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class ShortTermEWMAC(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [24, 48, 96]
        thrs = [1, 2, 3]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = EWMAC(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal


class LongTermEWMAC(TradingModel):
    def __init__(
        self,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
    ) -> None:
        super().__init__(open, high, low, close, volume)
        self.set_signal()

    def set_signal(self) -> None:
        lookbacks = [192, 384]
        thrs = [1, 2, 3]
        params = product(lookbacks, thrs)
        for lookback, thr in params:
            bollband_rule = EWMAC(
                lookback, thr, self.open, self.high, self.low, self.close, self.volume
            )
            self.set_rules_signals(bollband_rule)
        self.set_label()
        self.train_model()
        signal = self.model_pred()
        self.signal = signal

import pandas as pd
import numpy as np
from models import Model, save_pickle, load_pickle, get_cusum, get_tribar_label
from technical_indicators import TechnicalIndicator
from indicators import normalize, get_entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from scipy.stats import norm


class TechnicalModel(Model):
    def __init__(self, model_weight: float) -> None:
        super().__init__(model_weight=model_weight)

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


class EWMAC_Ridge(TechnicalModel):
    def __init__(
        self,
        model_weight: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
        lookbacks: pd.Series = None,
        lookforward: int = 6,
        meta_params: pd.Series = None,
        model_filename: str = "models/EWMAC_Ridge.pkl",
    ) -> None:
        super().__init__(model_weight=model_weight)

        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self.lookbacks = lookbacks
        self.lookforward = lookforward
        self.meta_params = meta_params

        self.model_filename = model_filename
        try:
            models = load_pickle(model_filename)
            self.ridge = models["ridge"]
            self.meta_logistic = models["meta_logistic"]
        except:
            self.ridge = RidgeCV(cv=10, fit_intercept=False)
            self.meta_logistic = LogisticRegressionCV(
                cv=10,
                fit_intercept=False,
                penalty="l2",
                solver="sag",
                max_iter=500,
                n_jobs=-1,
            )
            print("Models not found, reinitialized")

    def train_model(self) -> None:
        """
        Train Ridge
        """
        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)
        raw_signal = pd.DataFrame(
            {f"ewmac_{n}": ti.get_EWMAC(n) for n in self.lookbacks}
        ).dropna()

        # Train Ridge
        log_close = np.log(self.close)
        log_ret = log_close.diff(1).shift(-1)

        label = get_tribar_label(
            log_ret, ti.get_AVGHV(), tp=5, sl=5, period=self.lookforward
        ).rename("label")
        # label = normalize(log_ret / ti.get_AVGHV(), method="zeromean").rename("label")

        train = pd.concat([label, raw_signal], axis=1).dropna()

        self.ridge.fit(y=train.label, X=train.drop(columns="label"))

    def train_metamodel(self) -> None:
        """
        Train meta Logistic
        """
        assert self.forecasts is not None, "Try produce_forecasts() first"

        # self.meta_logistic = self.ridge
        # return

        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)
        feature = ti.get_all(self.meta_params)

        log_close = np.log(self.close)
        log_ret = log_close.diff(1).shift(-1)

        pnl = self.forecasts.clip(lower=-1, upper=1) * log_ret.shift(-1)

        feature["forecast"] = self.forecasts.clip(lower=-1, upper=1)
        feature["pnl"] = pnl

        label = (
            get_tribar_label(pnl, ti.get_AVGHV(), tp=5, sl=2, period=self.lookforward)
            .clip(lower=0)
            .rename("label")
        )

        train = pd.concat([label, feature], axis=1).dropna()
        # print("number of meta samples:")
        # print(len(train))

        self.meta_logistic.fit(y=train.label, X=train.drop(columns="label"))

    def save_models(self) -> None:
        """
        Save models
        """
        models = {"ridge": self.ridge, "meta_logistic": self.meta_logistic}
        save_pickle(models, self.model_filename)

    def produce_forecasts(self) -> None:
        """
        EWMAC -> Ridge
        """
        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)

        raw_signal = pd.DataFrame(
            {f"ewmac_{n}": ti.get_EWMAC(n) for n in self.lookbacks}
        ).dropna()

        forecasts = pd.Series(self.ridge.predict(raw_signal), index=raw_signal.index)

        forecasts = (
            get_cusum(normalize(forecasts, method="zeromean"))
            .reindex(forecasts.index)
            .ffill()
        )

        assert (
            self.close.index[-1] == forecasts.index[-1]
        ), "forecasts date not matched with input"

        # self.forecasts = float(forecasts.values[-1])
        self.forecasts = forecasts

    def produce_metaforecasts(self) -> None:
        """
        forecasts -> meta Logistic -> bet size
        """
        # self.metaforecasts = self.forecasts
        # return
        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)
        feature = ti.get_all(self.meta_params)

        log_close = np.log(self.close)

        feature["forecast"] = self.forecasts
        feature["pnl"] = np.sign(self.forecasts).shift(
            self.lookforward
        ) * log_close.diff(self.lookforward)

        feature.dropna(inplace=True)

        prob = self.meta_logistic.predict_proba(feature)

        pred = self.meta_logistic.predict(feature)

        p = prob.max(axis=1)
        z = (p - 1 / 2) / np.sqrt(p * (1 - p))
        z = norm.cdf(z)

        betsize = pd.Series((pred) * (2 * z - 1), index=feature.index)

        metaforecasts = normalize(np.sign(self.forecasts) * betsize, method="zeromean")

        self.metaforecasts = metaforecasts


class EWMAC_Logistic(TechnicalModel):
    def __init__(
        self,
        model_weight: float,
        open: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None,
        close: pd.Series = None,
        volume: pd.Series = None,
        lookbacks: pd.Series = None,
        lookforward: int = 6,
        meta_params: pd.Series = None,
        model_filename: str = "models/EWMAC_Logistic.pkl",
    ) -> None:
        super().__init__(model_weight=model_weight)

        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self.lookbacks = lookbacks
        self.lookforward = lookforward
        self.meta_params = meta_params

        self.model_filename = model_filename
        try:
            models = load_pickle(model_filename)
            self.logistic = models["logistic"]
            self.meta_logistic = models["meta_logistic"]
        except:
            self.logistic = LogisticRegressionCV(
                cv=10,
                fit_intercept=True,
                solver="sag",
                max_iter=500,
                n_jobs=-1,
            )
            self.meta_logistic = LogisticRegressionCV(
                cv=10,
                fit_intercept=False,
                solver="sag",
                max_iter=500,
                n_jobs=-1,
            )
            print("Models not found, reinitialized")

    def train_model(self) -> None:
        """
        Train Logistic
        """
        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)
        raw_signal = pd.DataFrame(
            {f"ewmac_{n}": ti.get_EWMAC(n) for n in self.lookbacks}
        ).dropna()

        # Train Logistic
        log_close = np.log(self.close)
        log_ret = log_close.diff(1).shift(-1)

        label = get_tribar_label(
            log_ret, ti.get_AVGHV(), tp=5, sl=5, period=self.lookforward
        ).rename("label")
        # label = normalize(log_ret / ti.get_AVGHV(), method="zeromean").rename("label")

        train = pd.concat([label, raw_signal], axis=1).dropna()

        self.logistic.fit(y=train.label, X=train.drop(columns="label"))

    def train_metamodel(self) -> None:
        """
        Train meta Logistic
        """
        assert self.forecasts is not None, "Try produce_forecasts() first"

        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)
        feature = ti.get_all(self.meta_params)

        log_close = np.log(self.close)
        log_ret = log_close.diff(1).shift(-1)

        pnl = self.forecasts.rolling(self.lookforward).mean() * log_ret.shift(-1)

        feature["pnl"] = pnl

        label = (
            get_tribar_label(pnl, ti.get_AVGHV(), tp=5, sl=2, period=self.lookforward)
            .clip(lower=0)
            .rename("label")
        )

        train = pd.concat([label, feature], axis=1).dropna()
        # print("number of meta samples:")
        # print(len(train))

        self.meta_logistic.fit(y=train.label, X=train.drop(columns="label"))

    def save_models(self) -> None:
        """
        Save models
        """
        models = {"logistic": self.logistic, "meta_logistic": self.meta_logistic}
        save_pickle(models, self.model_filename)

    def produce_forecasts(self) -> None:
        """
        EWMAC -> Logistic
        """
        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)

        raw_signal = pd.DataFrame(
            {f"ewmac_{n}": ti.get_EWMAC(n) for n in self.lookbacks}
        ).dropna()

        forecasts = pd.Series(self.logistic.predict(raw_signal), index=raw_signal.index)

        assert (
            self.close.index[-1] == forecasts.index[-1]
        ), "forecasts date not matched with input"

        # self.forecasts = float(forecasts.values[-1])
        self.forecasts = forecasts

    def produce_metaforecasts(self) -> None:
        """
        forecasts -> meta Logistic -> bet size
        """

        ti = TechnicalIndicator(self.open, self.high, self.low, self.close, self.volume)
        feature = ti.get_all(self.meta_params)

        log_close = np.log(self.close)
        log_ret = log_close.diff(1).shift(-1)

        pnl = self.forecasts.rolling(self.lookforward).mean() * log_ret.shift(-1)

        feature["pnl"] = pnl

        feature.dropna(inplace=True)

        prob = self.meta_logistic.predict_proba(feature)

        pred = self.meta_logistic.predict(feature)

        p = prob.max(axis=1)
        z = (p - 1 / 2) / np.sqrt(p * (1 - p))
        z = norm.cdf(z)

        betsize = pd.Series((pred) * (2 * z - 1), index=feature.index)

        metaforecasts = normalize(self.forecasts * betsize, method="zeromean")

        self.metaforecasts = metaforecasts

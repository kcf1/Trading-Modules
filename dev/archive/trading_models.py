import pandas as pd
import numpy as np
import pickle


def save_pickle(obj: object, filename: str) -> None:
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str) -> object:
    with open(filename, "rb") as handle:
        obj = pickle.load(handle)
        return obj


def get_cusum(data: pd.Series, thr: float = 0.25, n_std: float = None) -> pd.Series:
    events = []
    s_pos, s_neg = 0, 0
    if n_std is not None:
        thr = data.diff().std() * n_std
    diff = data.diff()
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        if s_neg < -thr:
            s_neg = 0
            events.append(i)
        elif s_pos > thr:
            s_pos = 0
            events.append(i)
    idx = pd.DatetimeIndex(events)
    return data.loc[idx]


def get_tribar_label(
    ret: pd.Series,
    vol: pd.Series,
    tp: float = None,
    sl: float = None,
    period: int = None,
) -> pd.Series:

    label = []

    for i in ret.index:
        up_bar = vol.loc[i] * tp
        low_bar = vol.loc[i] * -sl
        time_bar = period

        future_ret = ret.loc[i:]
        cum_ret = ret.loc[i]
        cum_period = 1
        while True:
            if up_bar is not None and cum_ret >= up_bar:
                label.append(1)
                break
            elif low_bar is not None and cum_ret <= low_bar:
                label.append(-1)
                break
            elif (
                time_bar is not None
                and cum_period >= time_bar
                or cum_period >= len(future_ret)
            ):
                label.append(0)
                break

            cum_period += 1
            cum_ret += future_ret.iloc[cum_period - 1]

    label = pd.Series(label, index=ret.index)
    return label


class TradingModel:
    def __init__(self, weight: float) -> None:
        assert weight >= 0 and weight <= 1, "Weight must between [0,1]"
        self.weight = weight

        self.models = None
        self.forecasts = None
        self.sides = None
        self.sizes = None
        self.bets = None
        self.weighted_bets = None

    def load_models(self, filename: str):
        """
        Load dict of models from pickle file
        """
        self.models = load_pickle(filename)

    def save_models(self, filename: str):
        """
        Save dict of models into pickle file
        """
        save_pickle(self.models, filename)

    def train_models(self) -> None:
        """
        Train underlying model
        """
        pass

    def produce_forecasts(self) -> None:
        """
        Raw forecast produced by the model
        """
        pass

    def produce_sides(self) -> None:
        """
        Buy/sell condition imposed on the raw forecast
        {-1, 0, +1}
        """
        pass

    def produce_sizes(self) -> None:
        """
        Bet size produced by the model or meta-model
        [0,1]
        """
        pass

    def produce_bets(self) -> None:
        """
        Bet = side * size
        Weighted bet = weight * bet
        """
        assert self.sides is not None, "produce_sides() first"
        assert self.sizes is not None, "produce_sizes() first"
        self.bets = self.sides * self.sizes
        self.weighted_bets = self.weight * self.bets

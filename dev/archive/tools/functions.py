import pandas as pd
import pickle


def save_pickle(obj: object, filename: str) -> None:
    """
    Save pickle
    """
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str) -> object:
    """
    Return object read from pickle
    """
    with open(filename, "rb") as handle:
        obj = pickle.load(handle)
        return obj


def get_cusum(data: pd.Series, thr: float = 0.25, n_std: float = None) -> pd.Series:
    """
    Return cusum filtered series
    """
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
    """
    Triple barrier label series
    """

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

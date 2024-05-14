from __future__ import annotations

import pandas as pd
import numpy as np


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


def clean(data: pd.Series) -> pd.Series:
    """
    Clean series
    """
    data[(data == np.inf) | (data == -np.inf)] = np.nan
    data.ffill(inplace=True)
    return data


def normalize(data: pd.Series, method: str = "zeromean") -> pd.Series:
    """
    Normalize series
    """
    if method is "zeromean":
        data = data / np.sqrt((data.abs() ** 2).mean())
        data = data / 2
    elif method is "zscore":
        data = (data - data.mean()) / data.std()
        data = data / 2
    elif method is "minmax":
        data = (data - data.min()) / (data.max() - data.min())
        data = data * 2 - 1
    elif method is "mediqr":
        data = (data - data.median()) / (data.percentile(0.75) - data.percentile(0.25))
        data = data / 2
    return data


def get_entropy(data: pd.Series, n_bins: int = 10) -> float:
    """
    Relative entropy
    """
    bins = pd.cut(data, bins=n_bins, labels=False, include_lowest=True, right=False)
    freq = data.groupby(bins).count()
    prob = freq / len(data)
    entropy = -(prob * np.log(prob)).sum() / np.log(n_bins)
    return entropy

import pandas as pd
import numpy as np


def clean(data: pd.Series) -> pd.Series:
    data[(data == np.inf) | (data == -np.inf)] = np.nan
    data.ffill(inplace=True)
    return data


def normalize(data: pd.Series, method: str = "zeromean") -> pd.Series:
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
    bins = pd.cut(data, bins=n_bins, labels=False, include_lowest=True, right=False)
    freq = data.groupby(bins).count()
    prob = freq / len(data)
    entropy = -(prob * np.log(prob)).sum() / np.log(n_bins)
    return entropy


class Indicator:
    def __init__(self) -> None:
        pass

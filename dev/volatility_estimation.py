from modules.data_collection.data_managers import PretradeDataManager

from modules.tools import StrategyEvaluator

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

print(os.getcwd())

import warnings

warnings.filterwarnings("ignore")

from modules.tools import read_json


ac_info = read_json("config/mt5_account.json")["ftmo-demo"]
db_info = read_json("config/postgres_info.json")["pre-trade"]
params = read_json("config/parameters.json")


from sklearn.cluster import KMeans

pdm = PretradeDataManager(ac_info, db_info, params)

universe = pdm.get_universe()
forex = universe.loc[universe.asset_class == "forex"]
symbols = forex.symbol.reset_index(drop=True).iloc[:]
print(symbols)

all_bars = {
    symbol: pdm.get_bars(symbol).drop(columns=["id", "symbol"]) for symbol in symbols
}


# Ewvol
result = pd.DataFrame()
result["n_period"] = [6000, 12000, 18000, 24000, 36000, 48000]
result["root_mean_sq_err"] = pd.Series()
result["mean_abs_err"] = pd.Series()

for i in result.index:
    n = int(result.loc[i, "n_period"])

    err_l = list()
    for symbol in symbols:
        log_ret = np.log(all_bars[symbol].close).diff()

        abs_ret_forward = log_ret.abs().shift(-1)
        vol = (log_ret**2).ewm(48).mean()**0.5
        long_vol = vol.rolling(n).mean()

        comb_vol = 0.7 * vol + 0.3 * long_vol

        err = (comb_vol - abs_ret_forward) * 100 * np.sqrt(24 * 252)
        err_l.append(err)

    result.loc[i, "root_mean_sq_err"] = (pd.concat(err_l, axis=0) ** 2).mean() ** 0.5
    result.loc[i, "mean_abs_err"] = pd.concat(err_l, axis=0).abs().mean()
    result.loc[i, "sample_size"] = pd.concat(err_l, axis=0).dropna().count()

result.to_csv("others/volatility_estimation.csv")

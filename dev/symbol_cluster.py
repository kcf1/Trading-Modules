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

all_bars = {
    symbol: pdm.get_bars(symbol).drop(columns=["id", "symbol"]) for symbol in symbols
}

points = forex.loc[:, ["symbol", "point"]].set_index("symbol").point
spreads = pd.DataFrame({symbol: all_bars[symbol].spread for symbol in symbols})
closes = pd.DataFrame({symbol: all_bars[symbol].close for symbol in symbols})
spreads_pct = spreads * points / closes

log_ret = np.log(closes).diff()

# Cluster by corr
corr_df = log_ret.corr().abs()

k = 10
kmean = KMeans(n_clusters=10)
kmean.fit(log_ret.dropna().T)
cluster_map = pd.DataFrame(index=log_ret.columns.values)

cluster_map["cluster"] = kmean.labels_

# Map volatility and costs
short_vol = log_ret.ewm(48).std()
long_vol = short_vol.rolling(24000).mean()
comb_vol = 0.7 * short_vol + 0.3 * long_vol

cluster_map["vol"] = comb_vol.iloc[-24000:].mean()*100
cluster_map["spread"] = spreads_pct.iloc[-24000:].mean()*100

cluster_map["risk_adj_spread"] = cluster_map["spread"] / cluster_map["vol"]

cluster_map.to_csv("others/clusters.csv")

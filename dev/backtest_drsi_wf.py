from modules.interfaces.mt5_interface import User
from modules.interfaces.postgres_interface import DataBase
from modules.data_managers.data_managers import PretradeDataManager

from modules.tools import StrategyEvaluator, backtest

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
pre_db_info = read_json("config/postgres_info.json")["pre-trade"]
post_db_info = read_json("config/postgres_info.json")["post-trade"]
params = read_json("config/parameters.json")

user = User().login_by_dict(ac_info)
pre_db = DataBase().connect_by_dict(pre_db_info)
post_db = DataBase().connect_by_dict(post_db_info)

pdm = PretradeDataManager(user, pre_db)


from modules.trading_models_test.rule_based_models.double_rsi import (
    DoubleRSI,
)

universe = pdm.get_universe()
forex = universe.loc[universe.asset_class == "crypto"]
symbols = forex.symbol.sample(10)

symbols = ["NZDUSD"]

print(symbols)


results = pd.Series()

fixed_cost = 0  # .00001
commission = 0.00003
slippage = 0  # .00001


all_bars = {
    symbol: pdm.get_bars(symbol).drop(columns=["id", "symbol"]) for symbol in symbols
}

pnl_df = pd.DataFrame()
pnl_net_df = pd.DataFrame()

train_size = 10000
step_size = 6000

symbol = symbols[0]
print("=========================================================")
print(symbol)

bars = all_bars[symbol]
point = universe.loc[universe.symbol == symbol, "point"].values[0]
spread = bars.spread * point / bars.close

# params_DoubleRSI = params[symbol]["double_rsi"]["params"]
params_DoubleRSI = [
    {
        "fast_rsi_lookback": n,
        "slow_rsi_lookback": n * 4,
        "ema_lookback": n,
        "ema_filter_lookback": n * 8,
    }
    for n in range(120,241,3)
]


weighted_bet_df = pd.DataFrame()
for params in params_DoubleRSI:
    model = DoubleRSI(1, params=params, bars=bars)
    weighted_bet_df[params["ema_lookback"]] = model.walkforward(
        train_size=train_size,
        step_size=step_size,
        max_lookback=max(params.values()),
    ).weighted_bet

pos = weighted_bet_df.fillna(0)
port_pos = weighted_bet_df.mean(axis=1).fillna(0)


def pf(pnl):
    return pnl.clip(lower=0).sum() / -pnl.clip(upper=0).sum()


from modules.trading_models_test.indicators.technical_indicators import (
    TechnicalIndicator,
)

signal = port_pos
price = np.log(bars.close)
vol = price * TechnicalIndicator(close=bars.close).get_combvol(48, 6000)
period = 12
from itertools import product

pf_df = pd.DataFrame()
trades_df = pd.DataFrame()
for a, k in product([3], [1]):
    i, j = a, a * k

    tp = i * vol
    sl = j * vol
    trades = backtest(signal, price, tp, sl, period)
    trades_df[f"{i}_{j}"] = trades
    pf_df.loc[i, j] = pf(trades)
pf_df.columns.name = "Stop-loss"
pf_df.index.name = "Take-profit"

trades_df.cumsum().plot()

pnl = port_pos.mul(np.log(bars.close / bars.open).shift(-1), axis=0)
pnl.cumsum().plot()

# sns.heatmap(pf_df)
"""

pnl = (
    pos.mul(np.log(bars.close / bars.open).shift(-1), axis=0)
    .dropna()
    .resample("d")
    .sum()
)
cost = (
    pos.diff()
    .abs()
    .mul(spread / 2 + commission + slippage, axis=0)
    .dropna()
    .resample("d")
    .sum()
)
port_pnl = (
    port_pos.mul(np.log(bars.close / bars.open).shift(-1), axis=0)
    .dropna()
    .resample("d")
    .sum()
)
port_cost = (
    port_pos.diff()
    .abs()
    .mul(spread / 2 + commission + slippage, axis=0)
    .dropna()
    .resample("d")
    .sum()
)

pnl_net = pnl - cost
port_pnl_net = port_pnl - port_cost


def pf(pnl):
    return pnl.clip(lower=0).sum() / -pnl.clip(upper=0).sum()


pnl_net.cumsum().plot()
port_pnl_net.cumsum().plot(color="black")
sr = port_pnl_net.mean() / port_pnl_net.std() * np.sqrt(252)
plt.title(f'SR={sr}')
e = StrategyEvaluator(port_pnl_net)
print(e.get_summary())

"""
"""
port_pnl = pnl_df.iloc[train_size:].mean(axis=1).resample("d").sum()
port_pnl_net = pnl_net_df.iloc[train_size:].mean(axis=1).resample("d").sum()
sr = port_pnl_net.mean() / port_pnl_net.std() * np.sqrt(252)
e = StrategyEvaluator(port_pnl_net)
print(e.get_summary())

pnl_df.cumsum().plot(title=f"SR = {sr:.2f}")
port_pnl.cumsum().plot(color="black", linestyle="-.")
port_pnl_net.cumsum().plot(color="black")
"""

plt.show()

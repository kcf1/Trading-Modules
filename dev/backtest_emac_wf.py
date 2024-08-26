from modules.interfaces import User, DataBase
from modules.managers import PretradeDataManager
from modules.tools import StrategyEvaluator, backtest, read_json

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

print(os.getcwd())

import warnings

warnings.filterwarnings("ignore")


ac_info = read_json("config/mt5_account.json")["ftmo-demo"]
pre_db_info = read_json("config/postgres_info.json")["pre-trade"]
post_db_info = read_json("config/postgres_info.json")["post-trade"]
params = read_json("config/parameters.json")

user = User().login_by_dict(ac_info)
pre_db = DataBase().connect_by_dict(pre_db_info)
post_db = DataBase().connect_by_dict(post_db_info)

pdm = PretradeDataManager(user, pre_db)


from modules.trading_models_test.rule_based_models import (
    FilteredBollingerBand,
    BollingerBand,
    DonchianBreakout,
    EMACrossover,
)

universe = pdm.get_universe()
forex = universe.loc[universe.asset_class == "crypto"]
symbols = forex.symbol.sample(10)

symbols = ["GBPUSD"]

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

# params_FilteredBollingerBand = params[symbol]["filtered_bollinger_band"]["params"]
params_EMACrossover = [
    {
        "fast_lookback": n,
        "slow_lookback": n * 8,
        "vol_lookback": 48,
        "avg_vol_lookback": 6000,
    }
    for n in range(12, 13, 24)
]

weighted_bet_df = pd.DataFrame()
for params in params_EMACrossover:
    model = EMACrossover(1, params=params, bars=bars)
    weighted_bet_df[f'{params["fast_lookback"]}'] = model.walkforward(
        train_size=train_size,
        step_size=step_size,
        max_lookback=max(params.values()),
    ).weighted_bet

pos = weighted_bet_df.fillna(0).mean(axis=1)

from modules.trading_models_test.indicators import TechnicalIndicator

signal = pos
price = bars.close
vol = price * TechnicalIndicator(close=bars.close).get_combvol(48, 6000)
tp = 6 * vol
sl = 2 * vol
period = 48

pnl = backtest(signal, price, tp, sl, period, trail_sl=True)

pnl.fillna(0).cumsum().plot()

"""
pos_chg = pos.diff()
buy = pos_chg.loc[pos_chg>0]
sell = pos_chg.loc[pos_chg<0]

period = bars.close.loc['2024':].index
buy_markers = bars.close.loc[buy.index.intersection(period)]
sell_markers = bars.close.loc[sell.index.intersection(period)]

ax = bars.close.loc[period].plot(alpha=0.5)
ax.scatter(y=buy_markers,x=buy_markers.index,marker='^',color='g')
ax.scatter(y=sell_markers,x=sell_markers.index,marker='v',color='r')
"""
# pos.mul(np.log(bars.close).diff().shift(-1), axis=0).loc['2024':].cumsum().plot()

# pnl = pos.mul(np.log(bars.close).diff().shift(-1), axis=0)# - pos.diff().abs().mul(spread, axis=0)
# pf = pnl.clip(lower=0).sum() / -pnl.clip(upper=0).sum()
# pf.plot()
# pnl.cumsum().plot()

"""
port_pos = weighted_bet_df.mean(axis=1).fillna(0)
cost = port_pos.abs().mul(spread, axis=0)


def pf(pnl):
    return pnl.clip(lower=0).sum() / -pnl.clip(upper=0).sum()


from modules.trading_models_test.indicators import TechnicalIndicator

signal = port_pos
price = np.log(bars.close)
vol = price * TechnicalIndicator(close=bars.close).get_combvol(48, 6000)
period = 12
from itertools import product

pf_df = pd.DataFrame()
trades_df = pd.DataFrame()
"""
"""
for a, k in product([4], [1]):
    i, j = a, a * k

    tp = i * vol
    sl = j * vol
    trades = backtest(signal, price, tp, sl, period)
    trades_df[f"{i}_{j}"] = trades
    pf_df.loc[a, k] = pf(trades)
pf_df.columns.name = "Stop-loss scalar"
pf_df.index.name = "Take-profit"

print(pf_df)

"""
"""

tp = 20 * vol
sl = 0.5 * vol

trades_df = backtest(signal, price, tp, sl, period)
trades_df = trades_df - cost

trades_df.fillna(0).rolling(12).mean().cumsum().plot()
pnl = port_pos.mul(np.log(bars.close / bars.open).shift(-1), axis=0)
pnl.cumsum().plot()

# sns.heatmap(pf_df)
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

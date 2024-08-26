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
    DonchianBand,
    EMACrossover,
    EMASlope,
)

universe = pdm.get_universe()
forex = universe.loc[universe.asset_class == "crypto"]
symbols = forex.symbol.sample(10)

symbols = ["USDJPY"]

print(symbols)


results = pd.Series()

fixed_cost = 0  # .00001
commission = 0  # .00003
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

params_EMASlope = [
    {
        "ema_lookback": n,
        "slope_lookback": n // 4,
    }
    for n in range(10, 71, 3)
]


weighted_bet_df = pd.DataFrame()
for params in params_EMASlope:
    model = EMASlope(1, params=params, bars=bars)
    weighted_bet_df[f'{params["ema_lookback"]}'] = model.walkforward(
        train_size=train_size,
        step_size=step_size,
        max_lookback=max(params.values()),
    ).weighted_bet

pos = weighted_bet_df.fillna(0).mean(axis=1)

from modules.trading_models_test.indicators import TechnicalIndicator

signal = pos
price = bars.close
vol = price * TechnicalIndicator(close=bars.close).get_combvol(48, 6000)
tp = 3 * vol
sl = 3 * vol
period = 12
cost = pos.abs() * (spread + commission * 2)

pnl = backtest(signal, price, tp, sl, period)
pnl_net = pnl - cost

pnl.fillna(0).rolling(12).mean().cumsum().plot()
pnl_net.fillna(0).rolling(12).mean().cumsum().plot(linestyle="-.", color="black")
price.plot(secondary_y=True)

plt.show()

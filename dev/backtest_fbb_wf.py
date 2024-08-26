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
)
from modules.trading_models_test.indicators import TechnicalIndicator

universe = pdm.get_universe()
forex = universe.loc[universe.asset_class == "forex"]
symbols = forex.symbol  # .sample(10)

# symbols = ["AUDUSD"]

print(symbols)


results = pd.Series()

fixed_cost = 0  # .00001
commission = 0.00003
slippage = 0  # .00001


all_bars = {
    symbol: pdm.get_bars(symbol).drop(columns=["id", "symbol"]) for symbol in symbols
}
trades_df = pd.DataFrame()
for symbol in symbols:
    pnl_df = pd.DataFrame()
    pnl_net_df = pd.DataFrame()

    train_size = 10000
    step_size = 6000

    # symbol = symbols[0]
    print("=========================================================")
    print(symbol)

    bars = all_bars[symbol]
    point = universe.loc[universe.symbol == symbol, "point"].values[0]
    spread = bars.spread * point

    # params_FilteredBollingerBand = params[symbol]["filtered_bollinger_band"]["params"]
    params_FilteredBollingerBand = [
        {
            "ema_lookback": n,
            "vol_lookback": 48,
            "avg_vol_lookback": 6000,
            "width": 1,
            "ema_filter_lookback": n * 8,
        }
        for n in range(30, 91, 3)
    ]

    weighted_bet_df = pd.DataFrame()
    for params in params_FilteredBollingerBand:
        model = FilteredBollingerBand(1, params=params, bars=bars)
        weighted_bet_df[f'{params["ema_lookback"]}_{params["width"]}'] = (
            model.walkforward(
                train_size=train_size,
                step_size=step_size,
                max_lookback=max(params.values()),
            ).weighted_bet
        )

    pos = weighted_bet_df.fillna(0).mean(axis=1)

    signal = pos
    price = bars.close
    vol = price * TechnicalIndicator(close=bars.close).get_combvol(48, 6000)
    period = 12
    tp = 4 * vol
    sl = 4 * vol

    trades = backtest(signal, price, tp, sl, period)
    cost = (spread + commission) * pos.abs()
    trades_df[symbol] = (trades - cost) / vol

trades_df.fillna(0).rolling(12).mean().cumsum().plot()
plt.show()

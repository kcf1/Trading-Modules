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

symbols = ['EURGBP']
symbol = symbols[0]

bars = pdm.get_bars(symbol)
bars.close.plot()

plt.show()

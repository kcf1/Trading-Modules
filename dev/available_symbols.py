from modules.data_collection.data_managers import PretradeDataManager
from modules.tools import read_json

import pandas as pd
import numpy as np

ac_info = read_json("config/mt5_account.json")["ftmo-demo"]
db_info = read_json("config/postgres_info.json")["pre-trade"]
params = read_json("config/parameters.json")

pdm = PretradeDataManager(ac_info, db_info, params)

symbols_df = pdm.user.get_available_symbols()
symbols_df.to_csv("others/available_symbols.csv")


pdm.close()

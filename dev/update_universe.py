from modules.data_collection.data_managers import PretradeDataManager
from modules.tools import read_json

import pandas as pd
import numpy as np


ac_info = read_json("config/mt5_account.json")["ftmo-demo"]
db_info = read_json("config/postgres_info.json")["pre-trade"]
params = read_json("config/parameters.json")

universe_df = pd.read_excel("config/universe.xlsx", header=0, index_col=None)
print(universe_df)

pdm = PretradeDataManager(ac_info, db_info, params)
pdm.init_universe(universe_df)
pdm.set_universe()
pdm.update_universe(universe)

pdm.init_bars()

pdm.close()

from modules.interfaces.mt5_interface import User
from modules.interfaces.postgres_interface import DataBase

from modules.data_managers.data_managers import PretradeDataManager
from modules.tools import read_json


ac_info = read_json("config/mt5_account.json")["ftmo-demo"]
db_info = read_json("config/postgres_info.json")["pre-trade"]
params = read_json("config/parameters.json")

user = User().login_by_dict(ac_info)
db = DataBase().connect_by_dict(db_info)

pdm = PretradeDataManager(user, db)

pdm.set_universe()
pdm.update_bars()

pdm.close()

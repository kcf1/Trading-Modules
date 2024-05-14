from modules.interfaces import User, DataBase
from modules.managers import PretradeDataManager, OrderManager, PortfolioManager

from modules.tools import read_json

from time import sleep

ac_info = read_json("config/mt5_account.json")["ftmo-demo"]
pre_db_info = read_json("config/postgres_info.json")["pre-trade"]
post_db_info = read_json("config/postgres_info.json")["post-trade"]
params = read_json("config/parameters.json")
asset_allocation = read_json("config/asset_allocation.json")

user = User().login_by_dict(ac_info)
pre_db = DataBase().connect_by_dict(pre_db_info)
post_db = DataBase().connect_by_dict(post_db_info)

pdm = PretradeDataManager(user, pre_db)
om = OrderManager(user, post_db)
pm = PortfolioManager(om, pdm, asset_allocation, params)

pdm.set_universe()
pm.init_assets()

while True:
    # Update everything
    pdm.update_bars()
    pm.update_assets_attr()
    pm.update_capital(100000)
    pm.allocate_capital()

    # Send order
    pm.send_order()

    # Sleep 1 hr wait for the next bar
    mins = 60
    print(f"Operation finished, next in {mins} mins")
    sleep(mins * 60)

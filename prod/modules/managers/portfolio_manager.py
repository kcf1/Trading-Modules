from __future__ import annotations

import pandas as pd

from ..managers import PretradeDataManager, OrderManager
from ..portfolio import *


class PortfolioManager:
    def __init__(
        self,
        om: OrderManager,
        pdm: PretradeDataManager,
        asset_allocation: dict,
        params: dict,
    ):
        self.om = om
        self.pdm = pdm
        self.asset_allocation = asset_allocation
        self.params = params

        self.init_assets()
        pass

    def init_assets(self):
        assets = dict()

        forex_symbols = self.asset_allocation["forex"]["symbols"]
        for symbol in forex_symbols:
            assets[symbol] = eval(
                f"{symbol}(self.om, self.pdm, self.params['{symbol}'])"
            )
        self.assets = assets

    def update_assets_attr(self):
        for symbol in self.assets.keys():
            self.assets[symbol].set_spec()
            self.assets[symbol].set_bars()
            self.assets[symbol].set_price()
            self.assets[symbol].set_vol_scalar()
            self.assets[symbol].set_lot_usd()

    def update_capital(self, capital):
        self.capital = capital

    def allocate_capital(self):
        # Simple risk-parity
        assert self.capital is not None, "update_capital() first"
        assert self.assets is not None, "init_assets() first"

        inv_vol = pd.Series(0.0, index=self.assets.keys())
        for symbol in inv_vol.index:
            inv_vol[symbol] = 1.0 / self.assets[symbol].vol

        forex_symbols = self.asset_allocation["forex"]["symbols"]
        forex_inv_vol = inv_vol.loc[forex_symbols]
        forex_weights = forex_inv_vol / forex_inv_vol.sum()
        for symbol in forex_weights.index:
            sub_capital = forex_weights[symbol] * self.capital
            self.assets[symbol].set_capital(sub_capital)

    def send_order(self):
        assert self.assets is not None, "init_assets() first"

        for symbol in self.assets.keys():
            self.assets[symbol].set_position()
            self.assets[symbol].send_order()

    def close_all(self):
        assert self.assets is not None, "init_assets() first"

        for symbol in self.assets.keys():
            self.om.close_all_market(symbol)

from __future__ import annotations

import pandas as pd

from ..managers import OrderManager, PretradeDataManager

from ..trading_models.indicators import TechnicalIndicator


class Asset:
    def __init__(
        self,
        om: OrderManager,
        pdm: PretradeDataManager,
        params: dict
    ):
        self.om = om
        self.pdm = pdm
        self.params = params
        self.capital = None

    def set_capital(self, capital: float) -> None:
        self.capital = capital

    def set_symbol(self, symbol: str) -> None:
        self.symbol = symbol

    def set_usd_rate_name(self, usd_rate_name: str) -> None:
        self.usd_rate_name = usd_rate_name

    def set_spec(self) -> None:
        assert self.symbol is not None, "set_spec() first"
        self.spec = self.pdm.get_spec(self.symbol)

    def set_bars(self) -> None:
        assert self.symbol is not None, "set_symbol() first"
        self.bars = self.pdm.get_bars(self.symbol)

    def set_price(self) -> None:
        assert self.bars is not None, "set_bars() first"
        assert self.spec is not None, "set_spec() first"
        price = self.bars.iloc[-1][["open", "high", "low", "close"]].mean()

        usd_rate_name = self.usd_rate_name
        if usd_rate_name == "USDUSD":
            usd_rate = 1
        elif usd_rate_name[:3] == "USD":
            rates = self.pdm.get_bars(usd_rate_name, last_n_rows=1)
            usd_rate = 1 / rates[["open", "high", "low", "close"]].iloc[-1].mean()
        elif usd_rate_name[3:] == "USD":
            rates = self.pdm.get_bars(usd_rate_name, last_n_rows=1)
            usd_rate = rates[["open", "high", "low", "close"]].iloc[-1].mean()
        self.price = price
        self.price_usd = price * usd_rate

    def set_lot_usd(self) -> None:
        assert self.price_usd is not None, "set_price() first"
        self.lot_usd = self.price_usd * self.spec.trade_contract_size

    def set_vol_scalar(self) -> None:
        assert self.price is not None, "set_price() first"
        self.vol = (
            TechnicalIndicator(close=self.bars.close).get_combvol(48, 6000).iloc[-1]
        )
        self.p_vol = self.price * self.vol

    def set_position(self) -> pd.DataFrame:
        assert self.symbol is not None, "Symbol is not set"
        self.position = self.om.book.get_symbol_pos(self.symbol)
        return self.position

    def send_order(self) -> None:
        pass

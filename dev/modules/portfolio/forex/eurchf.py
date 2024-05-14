from __future__ import annotations

import numpy as np

from datetime import timedelta

from ...managers import OrderManager, PretradeDataManager

from ...trading_models.rule_based_models import (
    FilteredBollingerBand,
    BollingerBand,
)

from ..asset import Asset


class EURCHF(Asset):
    def __init__(self, om: OrderManager, pdm: PretradeDataManager, params: dict):
        super().__init__(om, pdm, params)
        self.set_symbol("EURCHF")
        self.set_usd_rate_name("USDCHF")

    def send_order(self) -> None:
        super().send_order()
        print(f"{str('['+self.symbol+']'):<15} ordering...")
        # self.order_Test(1)
        self.order_FilteredBollingerBand(
            weight=self.params["filtered_bollinger_band"]["weight"]
        )
        self.order_BollingerBand(weight=self.params["bollinger_band"]["weight"])
        print(f"Cleared orders")
        print()

    def order_Test(self, weight: float) -> None:
        params_strat = self.params["filtered_bollinger_band"]
        comment = "M00001"
        strat_pos = self.position.loc[self.position.comment == comment].sort_values(
            "time"
        )
        bet = 1

        if len(strat_pos) <= 0:
            # Open position if no existing positions
            print("Open position")
            symbol = self.symbol
            side = np.sign(bet)
            n_lots = bet * weight * self.capital / self.lot_usd
            comment = comment
            tp = self.price + side * (params_strat["tp"] * self.p_vol)
            sl = self.price - side * (params_strat["sl"] * self.p_vol)

            self.om.open_market(symbol, side, n_lots, comment, tp, sl)

    def order_FilteredBollingerBand(self, weight: float) -> None:
        print("Filtered bollinger band")
        assert self.capital is not None, "set_capital() first"

        params_strat = self.params["filtered_bollinger_band"]
        comment = "M0001"
        holding_period = params_strat["period"]
        holding_time = holding_period * timedelta(hours=1)

        strat_pos = self.position.loc[self.position.comment == comment].sort_values(
            "time"
        )

        bets = list()
        for params in params_strat["params"]:
            model = FilteredBollingerBand(1, params=params, bars=self.bars)
            model.produce_rules()
            model.produce_sides()
            model.produce_sizes()
            model.produce_bets()
            bets.append(model.get_last_bet())
        bet = np.mean(bets)
        if len(strat_pos) > 0:
            # Close all position exceed holding time
            print("Close all position exceed holding time")
            print(self.om.close_exceed_time(self.symbol, holding_time))
        if len(strat_pos) < holding_period and bet != 0:
            # Open position if no existing positions
            print("Open position")
            symbol = self.symbol
            side = np.sign(bet)
            n_lots = bet * weight * self.capital / self.lot_usd
            comment = comment
            tp = self.price + side * (params_strat["tp"] * self.p_vol)
            sl = self.price - side * (params_strat["sl"] * self.p_vol)

            if round(n_lots, 2) >= 0.01:
                self.om.open_market(symbol, side, n_lots, comment, tp, sl)

    def order_BollingerBand(self, weight: float) -> None:
        print("Bollinger band")
        assert self.capital is not None, "set_capital() first"

        params_strat = self.params["bollinger_band"]
        comment = "M0002"
        holding_period = params_strat["period"]
        holding_time = holding_period * timedelta(hours=1)

        strat_pos = self.position.loc[self.position.comment == comment].sort_values(
            "time"
        )

        bets = list()
        for params in params_strat["params"]:
            model = BollingerBand(1, params=params, bars=self.bars)
            model.produce_rules()
            model.produce_sides()
            model.produce_sizes()
            model.produce_bets()
            bets.append(model.get_last_bet())
        bet = np.mean(bets)
        if len(strat_pos) > 0:
            # Close all position exceed holding time
            print("Close all position exceed holding time")
            print(self.om.close_exceed_time(self.symbol, holding_time))
        if len(strat_pos) < holding_period and bet != 0:
            # Open position if no existing positions
            print("Open position")
            symbol = self.symbol
            side = np.sign(bet)
            n_lots = bet * weight * self.capital / self.lot_usd
            comment = comment
            tp = self.price + side * (params_strat["tp"] * self.p_vol)
            sl = self.price - side * (params_strat["sl"] * self.p_vol)

            if round(n_lots, 2) >= 0.01:
                self.om.open_market(symbol, side, n_lots, comment, tp, sl)

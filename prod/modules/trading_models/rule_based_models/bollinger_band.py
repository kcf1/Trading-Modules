from __future__ import annotations

import pandas as pd
import numpy as np

from ..trading_models import TradingModel

from ..indicators import TechnicalIndicator


class BollingerBand(TradingModel):
    def __init__(self, weight: float, params: dict, bars: pd.DataFrame) -> None:
        super().__init__(weight, params, bars)

    def load_models(self, filename: str = None) -> None:
        pass

    def save_models(self, filename: str = None) -> None:
        pass

    def train_models(self) -> None:
        super().train_models()

    def produce_rules(self) -> None:
        super().produce_rules()
        bars = self.bars
        params = self.params
        ti = TechnicalIndicator(
            bars.open,
            bars.high,
            bars.low,
            bars.close,
            bars.tick_volume,
        )

        ema, upper, lower = ti.get_bb(
            params["ema_lookback"],
            params["vol_lookback"],
            params["avg_vol_lookback"],
            params["width"],
        )

        rules = pd.DataFrame()
        rules["close"] = bars.close
        rules["ema"] = ema
        rules["upper"] = upper
        rules["lower"] = lower

        self.rules = rules.dropna()

    def produce_sides(self) -> None:
        super().produce_sides()
        rules = self.rules

        long_entry = np.where((rules.close <= rules.lower), 1, 0)
        short_entry = np.where((rules.close >= rules.upper), -1, 0)
        sides = pd.Series(long_entry + short_entry, index=rules.index)

        self.sides = sides

    def produce_sizes(self) -> None:
        super().produce_sizes()
        sizes = pd.Series(1.0, index=self.sides.index)

        self.sizes = sizes

    def produce_bets(self) -> None:
        super().produce_bets()

    def get_last_bet(self) -> float:
        return super().get_last_bet()

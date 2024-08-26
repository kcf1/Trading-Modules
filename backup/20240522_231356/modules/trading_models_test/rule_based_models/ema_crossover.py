from __future__ import annotations

import pandas as pd
import numpy as np
from ..trading_models import TradingModel

from ..indicators import TechnicalIndicator


class EMACrossover(TradingModel):
    def __init__(self, weight: float, params: dict, bars: pd.DataFrame) -> None:
        super().__init__(weight, params, bars)

    def load_models(self, filename: str = None) -> None:
        pass

    def save_models(self, filename: str = None) -> None:
        pass

    def train_models(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        super().train_models(start_idx, end_idx)
        if start_idx is None:
            self.bars.index[0]
        if end_idx is None:
            self.bars.index[-2]

    def produce_rules(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        super().produce_rules(start_idx, end_idx)
        if start_idx is None:
            self.bars.index[0]
        if end_idx is None:
            self.bars.index[-2]
        bars = self.bars.loc[start_idx:end_idx]
        params = self.params
        ti = TechnicalIndicator(
            bars.open,
            bars.high,
            bars.low,
            bars.close,
            bars.tick_volume,
        )

        fast_ema = ti.get_ema(params["fast_lookback"])
        slow_ema = ti.get_ema(params["slow_lookback"])
        combvol = ti.get_combvol(params["vol_lookback"], params["avg_vol_lookback"])

        rules = pd.DataFrame()
        rules["close"] = bars.close
        rules["fast_ema"] = fast_ema
        rules["slow_ema"] = slow_ema
        rules["combvol"] = combvol

        self.rules = rules.dropna()

    def produce_sides(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        super().produce_sides(start_idx, end_idx)
        if start_idx is None:
            self.bars.index[0]
        if end_idx is None:
            self.bars.index[-1]
        rules = self.rules.loc[start_idx:end_idx]

        long_entry = np.where(rules.fast_ema > rules.slow_ema, 1, 0)
        short_entry = np.where(rules.fast_ema < rules.slow_ema, -1, 0)
        sides = pd.Series(long_entry + short_entry, index=rules.index)

        vol_filter = rules.combvol <= rules.combvol.expanding(6000).quantile(0.25)
        #sides.loc[vol_filter] = 0

        self.sides = sides

    def produce_sizes(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        super().produce_sizes(start_idx, end_idx)
        if start_idx is None:
            self.bars.index[0]
        if end_idx is None:
            self.bars.index[-1]
        sizes = pd.Series(1.0, index=self.sides.index)

        self.sizes = sizes

    def produce_bets(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        super().produce_bets(start_idx, end_idx)

    def walkforward(
        self, train_size: int, step_size: int, max_lookback: int
    ) -> pd.DataFrame:
        return super().walkforward(train_size, step_size, max_lookback)

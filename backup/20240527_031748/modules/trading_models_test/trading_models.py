from __future__ import annotations

import pandas as pd
import numpy as np
from ..tools import load_pickle, save_pickle


class TradingModel:
    def __init__(self, weight: float, params: dict, bars: pd.DataFrame) -> None:
        assert weight >= 0 and weight <= 1, "Weight must between [0,1]"
        self.weight = weight
        self.params = params
        self.bars = bars

        self.models = None
        self.rules = None
        self.sides = None
        self.sizes = None
        self.bets = None
        self.weighted_bets = None

    def load_models(self, filename: str) -> None:
        """
        Load dict of models from pickle file
        """
        self.models = load_pickle(filename)

    def save_models(self, filename: str) -> None:
        """
        Save dict of models into pickle file
        """
        save_pickle(self.models, filename)

    def train_models(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        """
        Train underlying model
        """
        pass

    def produce_rules(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        """
        Rule series
        """
        pass

    def produce_sides(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        """
        Buy/sell condition imposed on the raw forecast
        {-1, 0, +1}
        """
        assert self.rules is not None, "produce_rules() first"
        pass

    def produce_sizes(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        """
        Bet size produced by the model or meta-model
        [0,1]
        """
        assert self.sides is not None, "produce_sides() first"
        pass

    def produce_bets(
        self, start_idx: pd.DatetimeIndex = None, end_idx: pd.DatetimeIndex = None
    ) -> None:
        """
        Bet = side * size
        Weighted bet = weight * bet
        """
        assert self.sides is not None, "produce_sides() first"
        assert self.sizes is not None, "produce_sizes() first"

        if start_idx is None:
            self.bars.index[0]
        if end_idx is None:
            self.bars.index[-1]

        self.bets = (self.sides * self.sizes).loc[start_idx:end_idx]
        self.weighted_bets = self.weight * self.bets

    def walkforward(
        self, train_size: int, step_size: int, max_lookback: int
    ) -> pd.DataFrame:
        """
        Walkforward testing
        Return walkforward_df
        """
        start, end = 0, len(self.bars) - 1
        idx = self.bars.index

        assert end > train_size, "Lack of enough bars to train"
        assert train_size > max_lookback, "Train bars must larger than max_lookback"

        sides_l = list()
        sizes_l = list()
        bets_l = list()
        weighted_bets_l = list()

        while True:
            # start ----- (train_size-1) days -----> end
            start_train = idx[start]
            end_train = idx[start + train_size - 1 - max_lookback]

            # end+1 ----- (step-1) days -----------> end test
            start_lookback = idx[start + train_size - max_lookback]
            start_test = idx[start + train_size]
            end_test = idx[min(start + train_size + step_size - 1, end)]

            # print(f"Training: [{start_train} - {end_train}]")
            self.train_models(start_train, end_train)

            # print(f"Testing: [{start_test} - {end_test}]")
            self.produce_rules(start_lookback, end_test)
            self.produce_sides(start_test, end_test)
            self.produce_sizes(start_test, end_test)
            self.produce_bets(start_test, end_test)

            sides_l.append(self.sides.loc[start_test:end_test])
            sizes_l.append(self.sizes.loc[start_test:end_test])
            bets_l.append(self.bets.loc[start_test:end_test])
            weighted_bets_l.append(self.weighted_bets.loc[start_test:end_test])

            start += step_size
            if start + train_size >= end:
                break

        walkforward_df = pd.DataFrame(
            {
                "side": pd.concat(sides_l),
                "size": pd.concat(sizes_l),
                "bet": pd.concat(bets_l),
                "weighted_bet": pd.concat(weighted_bets_l),
            }
        )

        walkforward_df[walkforward_df == 0] = np.nan

        return walkforward_df

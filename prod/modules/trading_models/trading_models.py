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

    def train_models(self) -> None:
        """
        Train underlying model
        """
        pass

    def produce_rules(self) -> None:
        """
        Rule series
        """
        pass

    def produce_sides(self) -> None:
        """
        Buy/sell condition imposed on the raw forecast
        {-1, 0, +1}
        """
        assert self.rules is not None, "produce_rules() first"
        pass

    def produce_sizes(self) -> None:
        """
        Bet size produced by the model or meta-model
        [0,1]
        """
        assert self.sides is not None, "produce_sides() first"
        pass

    def produce_bets(self) -> None:
        """
        Bet = side * size
        Weighted bet = weight * bet
        """
        assert self.sides is not None, "produce_sides() first"
        assert self.sizes is not None, "produce_sizes() first"

        self.bets = self.sides * self.sizes
        self.weighted_bets = self.weight * self.bets

    def get_last_bet(self) -> float:
        """
        Last weighted bet
        """
        assert self.weighted_bets is not None, "produce_bets() first"
        last_time = self.weighted_bets.index[-1]
        last_bet = self.weighted_bets.iloc[-1]
        # print(last_time)
        return last_bet

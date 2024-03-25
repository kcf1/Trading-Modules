from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from mt5_interface import User
from postgres_interface import DataBase
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database


class PretradeDataUpdater:
    def __init__(self, mt5_config, postgres_config):
        self.user = User().login_by_dict(mt5_config)
        self.db = DataBase().connect_by_dict(postgres_config)
        self.universe = None
        pass

    def set_universe(self) -> None:
        universe = self.db.read_all("universe")
        self.universe = universe

    def init_universe(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1000, 2000, 3000, 4000, 5000],
                "symbol": ["-"] * 5,
                "asset_class": ["equity", "currency", "metal", "agriculture", "crypto"],
                "nominal_currency": ["-"] * 5,
                "description": ["-"] * 5,
            }
        )
        self.db.create_table_from_df("universe", df)
        self.db.commit()
        self.set_universe()

    def update_universe(self, df: pd.DataFrame) -> None:
        """Columns: symbol, asset_class, nominal_currency, description"""

        assert self.universe is not None, "Please set_universe()"

        table_name = "universe"
        last_id_by_class = self.universe.groupby("asset_class")["id"].max()

        unique_symbols = ~df["symbol"].isin(self.universe["symbol"])

        df_with_id = df.loc[unique_symbols]
        df_with_id["id"] = 0
        for i, r in df_with_id.iterrows():
            last_id = last_id_by_class.loc[r.asset_class] + 1
            last_id_by_class.loc[r.asset_class] = last_id
            df_with_id.loc[i, "id"] = last_id

        assert len(df_with_id) + len(self.universe) >= len(df) + len(
            self.universe
        ), "No new symbol is added"

        self.db.insert_rows_from_df(table_name, df_with_id)
        self.db.commit()
        self.set_universe()

    def init_bars(self) -> None:

        assert self.universe is not None, "Please set_universe()"

        table_name = "bars"
        table_created = False
        universe = self.universe.loc[self.universe["symbol"] != "-"]
        for i, r in universe.iterrows():
            symbol_id = r.id
            df = self.user.get_bars(r.symbol, "1h", period_days=4000)
            df["symbol"] = r.symbol
            df["id"] = str(symbol_id) + "-" + df["time"].dt.strftime("%Y%m%d%H%M%S")
            df = df[
                [
                    "id",
                    "symbol",
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "spread",
                    "tick_volume",
                    "real_volume",
                ]
            ]

            if not table_created:
                self.db.create_table_from_df(table_name, df)
                table_created = True
            else:
                self.db.insert_rows_from_df(table_name, df)

        self.db.commit()

    def update_bars(self) -> None:

        assert self.universe is not None, "Please set_universe()"

        table_name = "bars"
        universe = self.universe.loc[self.universe["symbol"] != "-"]
        for i, r in universe.iterrows():
            columns = self.db.read_columns(table_name)
            cmd = f"SELECT * FROM {table_name} WHERE symbol = '{r.symbol}' ORDER BY time DESC LIMIT 1;"
            latest_bar = pd.Series(self.db.execute(cmd)[0], index=columns)
            df = self.user.get_bars(
                r.symbol,
                "1h",
                start=latest_bar.time,
            )
            print(latest_bar.time)
            symbol_id = r.id
            df["symbol"] = r.symbol
            df["id"] = str(symbol_id) + "-" + df["time"].dt.strftime("%Y%m%d%H%M%S")
            df = df[
                [
                    "id",
                    "symbol",
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "spread",
                    "tick_volume",
                    "real_volume",
                ]
            ]

            assert df.iloc[0].id == latest_bar.id, print(
                "Local data not reconciling with MT5 source"
            )

            assert len(df.iloc[1:]) > 0, "No new bars has been updated"

            self.db.insert_rows_from_df("bars", df.iloc[1:])

    def update_technicals(self) -> None:
        pass

    def update_fundamentals(self) -> None:
        pass

    def update_macro(self) -> None:
        pass

    def close(self) -> None:
        self.db.close()
        self.user.shutdown()

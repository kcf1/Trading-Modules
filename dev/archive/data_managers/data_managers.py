from __future__ import annotations

import pandas as pd
import numpy as np
import datetime as dt
from time import sleep

from ..interfaces.mt5_interface import User
from ..interfaces.postgres_interface import DataBase


class PretradeDataManager:
    def __init__(self, user: User, db: DataBase):
        self.user = user
        self.db = db

        self.universe = None
        pass

    def set_universe(self) -> None:
        universe = self.db.read_all("universe")
        self.universe = universe

    def init_universe(self, universe_df: pd.DataFrame) -> None:
        """
        Drop universe and reinitialize
        (improvement needed)
        """

        if (
            input(
                "init_universe() will drop existing universe, confirm? (y/n)\n"
            ).lower()
            != "y"
        ):
            print("Halt operation")
            return

        assert len(universe_df) > 0, "universe_df has no rows!"

        self.db.drop("universe")
        self.db.create_table_from_df("universe", universe_df)
        self.db.commit()
        self.set_universe()

    def update_universe(self, df: pd.DataFrame) -> None:
        """
        Columns: symbol, asset_class, nominal_currency, description
        """

        assert self.universe is not None, "Please set_universe()"

        last_id_by_class = self.universe.groupby("asset_class")["id"].max()

        unique_symbols = ~df["symbol"].isin(self.universe["symbol"])

        df_with_id = df.loc[unique_symbols]
        df_with_id["id"] = 0
        for i, r in df_with_id.iterrows():
            last_id = last_id_by_class.loc[r.asset_class] + 1
            last_id_by_class.loc[r.asset_class] = last_id
            df_with_id.loc[i, "id"] = last_id

        assert len(df_with_id) >= 0, "No new symbol is added"

        self.db.insert_rows_from_df("universe", df_with_id)
        self.db.commit()
        self.set_universe()

    def get_universe(self) -> pd.DataFrame:
        # universe = self.db.read_all("universe")

        cmd = f"SELECT * FROM universe WHERE symbol != '-';"
        universe = pd.DataFrame(
            self.db.execute(cmd), columns=self.db.read_columns("universe")
        )

        return universe

    def get_spec(self, symbol: str) -> pd.Series:
        cmd = f"SELECT * FROM universe WHERE symbol = '{symbol}';"
        spec = pd.Series(
            self.db.execute(cmd)[0], index=self.db.read_columns("universe"), name="spec"
        )
        return spec

    def init_bars(self) -> None:
        """
        Drop all bars and reinitialize
        (improvement needed)
        """

        if (
            input("init_bars() will drop all existing bars, confirm? (y/n)\n").lower()
            != "y"
        ):
            print("Halt operation")
            return

        assert self.universe is not None, "Please set_universe()"

        self.db.drop("bars")
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
                self.db.create_table_from_df("bars", df)
                table_created = True
            else:
                self.db.insert_rows_from_df("bars", df)

        self.db.commit()

    def update_bars(self) -> None:

        assert self.universe is not None, "Please set_universe()"

        universe = self.universe.loc[self.universe["symbol"] != "-"]
        for i, r in universe.iterrows():
            print(f"{str('['+r.symbol+']'):<15} {r.asset_class.upper()}")
            try:
                columns = self.db.read_columns("bars")
                cmd = f"SELECT * FROM bars WHERE symbol = '{r.symbol}' ORDER BY time DESC LIMIT 1;"
                latest_bar = pd.Series(self.db.execute(cmd)[0], index=columns)
                df = self.user.get_bars(
                    r.symbol,
                    "1h",
                    start=latest_bar.time,
                )
                # print(latest_bar.time)
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

                if len(df.iloc[1:]) <= 0:
                    print("No new bars")
                else:
                    print(df.iloc[1:])
                    self.db.insert_rows_from_df("bars", df.iloc[1:])
                    print("Updated")
            except:
                print("Failed to update")
            finally:
                print()
        # Sleep 1s to avoid skipping bars
        sleep(1)

    def get_bars(
        self,
        symbol: str,
        start: str | dt.datetime = None,
        end: str | dt.datetime = None,
        last_n_rows: int = None
    ) -> pd.DataFrame:
        if last_n_rows is None:
            if end is None:
                end_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)
                end = dt.datetime.strftime(end_time, format="%Y-%m-%d %H:%M:%S")
            elif type(end) is str:
                pass
            elif type(end) is dt.datetime:
                end = dt.datetime.strftime(end_time, format="%Y-%m-%d %H:%M:%S")

            if start is None:
                start = "2010-01-01 00:00:00"
            elif type(start) is str:
                pass
            elif type(start) is dt.datetime:
                start = dt.datetime.strftime(start, format="%Y-%m-%d %H:%M:%S")

            cmd = f"SELECT * FROM bars WHERE symbol = '{symbol}' AND time BETWEEN '{start}' AND '{end}'ORDER BY time;"
        else:
            cmd = f"SELECT * FROM bars WHERE symbol = '{symbol}' ORDER BY time DESC LIMIT {last_n_rows};"
        bars = pd.DataFrame(self.db.execute(cmd), columns=self.db.read_columns("bars"))

        bars.set_index("time", inplace=True)
        bars.sort_index(inplace=True)
        return bars

    def update_fundamentals(self) -> None:
        pass

    def update_macro(self) -> None:
        pass

    def close(self) -> None:
        self.db.close()
        self.user.shutdown()

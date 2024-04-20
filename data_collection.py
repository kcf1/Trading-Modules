from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from mt5_interface import User
from postgres_interface import DataBase
from technical_indicators import TechnicalIndicator


class PretradeDataManager:
    def __init__(self, mt5_config: dict, postgres_config: dict, params: dict):
        self.user = User().login_by_dict(mt5_config)
        self.db = DataBase().connect_by_dict(postgres_config)
        self.params = params
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

        self.db.insert_rows_from_df("universe", df_with_id)
        self.db.commit()
        self.set_universe()

    def get_universe(self) -> pd.DataFrame:
        universe = self.db.read_all("universe")
        return universe

    def init_bars(self) -> None:

        assert self.universe is not None, "Please set_universe()"

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
            columns = self.db.read_columns("bars")
            cmd = f"SELECT * FROM bars WHERE symbol = '{r.symbol}' ORDER BY time DESC LIMIT 1;"
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

            if len(df.iloc[1:]) > 0:
                print("No new technicals has been updated")
            else:
                self.db.insert_rows_from_df("technicals", df.iloc[1:])

    def get_bars(
        self,
        symbol: str,
        start: str | dt.datetime = None,
        end: str | dt.datetime = None,
    ) -> pd.DataFrame:

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

        cmd = f"SELECT * FROM bars WHERE symbol = '{symbol}'AND time BETWEEN '{start}' AND '{end}'ORDER BY time;"
        bars = pd.DataFrame(self.db.execute(cmd), columns=self.db.read_columns("bars"))

        bars.set_index("time", inplace=True)
        bars.sort_index(inplace=True)
        return bars

    def init_technicals(self) -> None:

        assert self.universe is not None, "Please set_universe()"

        table_created = False

        lookbacks = self.params["technicals"]["lookbacks"]

        universe = self.universe.loc[self.universe["symbol"] != "-"]
        for i, r in universe.iterrows():
            symbol_id = r.id

            bars = self.get_bars(r.symbol)

            df = TechnicalIndicator(
                open=bars.open,
                high=bars.high,
                low=bars.low,
                close=bars.close,
                volume=bars.tick_volume,
            ).get_all(lookbacks)

            df["time"] = df.index
            df["symbol"] = r.symbol
            df["id"] = str(symbol_id) + "-" + df["time"].dt.strftime("%Y%m%d%H%M%S")

            cols = df.columns.to_list()
            reordered_cols = cols[-3:][::-1] + cols[:-3]
            df = df[reordered_cols]

            if not table_created:
                self.db.create_table_from_df("technicals", df)
                table_created = True
            else:
                self.db.insert_rows_from_df("technicals", df)

            print(f"Initialized technicals - {r.symbol}")

        self.db.commit()

    def update_technicals(self) -> None:

        assert self.universe is not None, "Please set_universe()"

        lookbacks = self.params["technicals"]["lookbacks"]

        universe = self.universe.loc[self.universe["symbol"] != "-"]
        for i, r in universe.iterrows():
            columns = self.db.read_columns("technicals")
            cmd = f"SELECT * FROM technicals WHERE symbol = '{r.symbol}' ORDER BY time DESC LIMIT 1;"
            latest_bar = pd.Series(self.db.execute(cmd)[0], index=columns)

            bars = self.get_bars(r.symbol)

            df = TechnicalIndicator(
                open=bars.open,
                high=bars.high,
                low=bars.low,
                close=bars.close,
                volume=bars.tick_volume,
            ).get_all(lookbacks)

            df = df.loc[df.index >= latest_bar.time]

            print(latest_bar.time)
            symbol_id = r.id
            df["time"] = df.index
            df["symbol"] = r.symbol
            df["id"] = str(symbol_id) + "-" + df["time"].dt.strftime("%Y%m%d%H%M%S")

            cols = df.columns.to_list()
            reordered_cols = cols[-3:][::-1] + cols[:-3]
            df = df[reordered_cols]

            assert df.iloc[0].id == latest_bar.id, print(
                "Local data not reconciling with updated technicals"
            )

            if len(df.iloc[1:]) > 0:
                print("No new technicals has been updated")
            else:
                self.db.insert_rows_from_df("technicals", df.iloc[1:])

    def get_technicals(
        self,
        symbol: str,
        start: str | dt.datetime = None,
        end: str | dt.datetime = None,
    ) -> pd.DataFrame:

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

        cmd = f"SELECT * FROM technicals WHERE symbol = '{symbol}'AND time BETWEEN '{start}' AND '{end}'ORDER BY time;"
        technicals = pd.DataFrame(
            self.db.execute(cmd), columns=self.db.read_columns("technicals")
        )

        technicals.set_index("time", inplace=True)
        technicals.sort_index(inplace=True)
        return technicals

    def update_fundamentals(self) -> None:
        pass

    def update_macro(self) -> None:
        pass

    def close(self) -> None:
        self.db.close()
        self.user.shutdown()

import MetaTrader5 as mt5
import json
import datetime as dt
import pandas as pd
import re


class User:
    def __init__(self) -> None:
        self.account = None
        self.password = None
        self.server = None

    def login_by_dict(self, d: dict) -> None:
        self.account = d["account"]
        self.password = d["password"]
        self.server = d["server"]

        is_init = mt5.initialize(
            login=self.account, password=self.password, server=self.server
        )
        if is_init:
            print(f"Logged in as {self.account}")
        else:
            self.shutdown()
            print(f"Failed to login to {self.account}")

    def shutdown(self) -> None:
        mt5.shutdown()

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: str = None,
        end: str = None,
        period_days: int = 365,
    ) -> None:
        if end is None:
            end_time = dt.datetime.now(dt.timezone.utc)
        else:
            end_time = dt.datetime.strptime(end, format="%Y-%m-%d")
        if start is None:
            start_time = end_time - dt.timedelta(days=period_days)
        else:
            start_time = dt.datetime.strptime(end, format="%Y-%m-%d")
        # parse timeframe
        timeframe_obj_dict = {
            "1m": mt5.TIMEFRAME_M1,
            "2m": mt5.TIMEFRAME_M2,
            "3m": mt5.TIMEFRAME_M3,
            "4m": mt5.TIMEFRAME_M4,
            "5m": mt5.TIMEFRAME_M5,
            "6m": mt5.TIMEFRAME_M6,
            "10m": mt5.TIMEFRAME_M10,
            "12m": mt5.TIMEFRAME_M12,
            "15m": mt5.TIMEFRAME_M15,
            "20m": mt5.TIMEFRAME_M20,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "2h": mt5.TIMEFRAME_H2,
            "3h": mt5.TIMEFRAME_H3,
            "4h": mt5.TIMEFRAME_H4,
            "6h": mt5.TIMEFRAME_H6,
            "8h": mt5.TIMEFRAME_H8,
            "12h": mt5.TIMEFRAME_H12,
            "1d": mt5.TIMEFRAME_D1,
        }
        timeframe_obj = timeframe_obj_dict[timeframe.lower()]
        print(f"Getting {timeframe} bars of {symbol} from {start_time} to {end_time}")
        bars = mt5.copy_rates_range(symbol, timeframe_obj, start_time, end_time)
        bars_df = pd.DataFrame(bars)
        # print(bars_df)
        clean_symbol = re.sub(r"[^a-zA-Z]", "", symbol)
        now_timestamp = int(end_time.timestamp())

        def create_id(time, clean_symbol, now_timestamp):
            id = f"{clean_symbol}-{time}-{now_timestamp}"
            return id

        bars_df["id"] = (
            bars_df["time"]
            .map(lambda time: create_id(time, clean_symbol, now_timestamp))
            .astype(str)
        )
        bars_df["time"] = pd.to_datetime(bars_df["time"], unit="s")
        return bars_df.sort_values("time", ascending=True)

from __future__ import annotations

import MetaTrader5 as mt5
import datetime as dt
import pandas as pd


class User:
    def __init__(self):
        self.account = None
        self.password = None
        self.server = None

    def login_by_dict(self, d: dict) -> User | None:
        self.path = d["path"]
        self.account = d["account"]
        self.password = d["password"]
        self.server = d["server"]

        is_init = mt5.initialize(
            path=self.path,
            login=self.account,
            password=self.password,
            server=self.server,
        )
        if is_init:
            print(f"Logged in as {self.account} ({self.server})")
            return self
        else:
            self.shutdown()
            print(f"Failed to login to {self.account} ({self.server})")
            return None

    def shutdown(self) -> None:
        mt5.shutdown()

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: str | dt.datetime = None,
        end: str | dt.datetime = None,
        period_days: int = 365,
    ) -> pd.DataFrame:
        if end is None:
            end_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)
        elif type(end) is str:
            end_time = dt.datetime.strptime(end, format="%Y-%m-%d %H:%M:%S").replace(
                tzinfo=dt.timezone.utc
            )
        elif type(end) is dt.datetime:
            end_time = end.replace(tzinfo=dt.timezone.utc)

        if start is None:
            start_time = end_time - dt.timedelta(days=period_days)
        elif type(start) is str:
            start_time = dt.datetime.strptime(
                start, format="%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=dt.timezone.utc)
        elif type(start) is dt.datetime:
            start_time = start.replace(tzinfo=dt.timezone.utc)

        assert (
            start_time is not None and end_time is not None
        ), "start_time and end_time is None"

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
        bars_df["time"] = pd.to_datetime(bars_df["time"], unit="s")
        return bars_df.sort_values("time", ascending=True)

    def get_available_symbols(self) -> pd.DataFrame:
        symbols_info = mt5.symbols_get(group="*")
        info_df = pd.DataFrame(
            {info.name: self.parse_symbol_info(info) for info in symbols_info}
        ).T
        return info_df

    def get_symbol(self, symbol: str) -> pd.Series:
        symbol_info = mt5.symbol_info(symbol)
        return self.parse_symbol_info(symbol_info)

    def parse_symbol_info(self, symbol_info: tuple) -> pd.Series:
        d = {
            "ask": symbol_info.ask,
            "askhigh": symbol_info.askhigh,
            "asklow": symbol_info.asklow,
            "bank": symbol_info.bank,
            "basis": symbol_info.basis,
            "bid": symbol_info.bid,
            "bidhigh": symbol_info.bidhigh,
            "bidlow": symbol_info.bidlow,
            "category": symbol_info.category,
            "chart_mode": symbol_info.chart_mode,
            "currency_base": symbol_info.currency_base,
            "currency_margin": symbol_info.currency_margin,
            "currency_profit": symbol_info.currency_profit,
            "custom": symbol_info.custom,
            "description": symbol_info.description,
            "digits": symbol_info.digits,
            "exchange": symbol_info.exchange,
            "expiration_mode": symbol_info.expiration_mode,
            "expiration_time": symbol_info.expiration_time,
            "filling_mode": symbol_info.filling_mode,
            "formula": symbol_info.formula,
            "isin": symbol_info.isin,
            "last": symbol_info.last,
            "lasthigh": symbol_info.lasthigh,
            "lastlow": symbol_info.lastlow,
            "margin_hedged": symbol_info.margin_hedged,
            "margin_hedged_use_leg": symbol_info.margin_hedged_use_leg,
            "margin_initial": symbol_info.margin_initial,
            "margin_maintenance": symbol_info.margin_maintenance,
            "n_fields": symbol_info.n_fields,
            "n_sequence_fields": symbol_info.n_sequence_fields,
            "n_unnamed_fields": symbol_info.n_unnamed_fields,
            "name": symbol_info.name,
            "option_mode": symbol_info.option_mode,
            "option_right": symbol_info.option_right,
            "option_strike": symbol_info.option_strike,
            "order_gtc_mode": symbol_info.order_gtc_mode,
            "order_mode": symbol_info.order_mode,
            "page": symbol_info.page,
            "path": symbol_info.path,
            "point": symbol_info.point,
            "price_change": symbol_info.price_change,
            "price_greeks_delta": symbol_info.price_greeks_delta,
            "price_greeks_gamma": symbol_info.price_greeks_gamma,
            "price_greeks_omega": symbol_info.price_greeks_omega,
            "price_greeks_rho": symbol_info.price_greeks_rho,
            "price_greeks_theta": symbol_info.price_greeks_theta,
            "price_greeks_vega": symbol_info.price_greeks_vega,
            "price_sensitivity": symbol_info.price_sensitivity,
            "price_theoretical": symbol_info.price_theoretical,
            "price_volatility": symbol_info.price_volatility,
            "select": symbol_info.select,
            "session_aw": symbol_info.session_aw,
            "session_buy_orders": symbol_info.session_buy_orders,
            "session_buy_orders_volume": symbol_info.session_buy_orders_volume,
            "session_close": symbol_info.session_close,
            "session_deals": symbol_info.session_deals,
            "session_interest": symbol_info.session_interest,
            "session_open": symbol_info.session_open,
            "session_price_limit_max": symbol_info.session_price_limit_max,
            "session_price_limit_min": symbol_info.session_price_limit_min,
            "session_price_settlement": symbol_info.session_price_settlement,
            "session_sell_orders": symbol_info.session_sell_orders,
            "session_sell_orders_volume": symbol_info.session_sell_orders_volume,
            "session_turnover": symbol_info.session_turnover,
            "session_volume": symbol_info.session_volume,
            "spread": symbol_info.spread,
            "spread_float": symbol_info.spread_float,
            "start_time": symbol_info.start_time,
            "swap_long": symbol_info.swap_long,
            "swap_mode": symbol_info.swap_mode,
            "swap_rollover3days": symbol_info.swap_rollover3days,
            "swap_short": symbol_info.swap_short,
            "ticks_bookdepth": symbol_info.ticks_bookdepth,
            "time": symbol_info.time,
            "trade_accrued_interest": symbol_info.trade_accrued_interest,
            "trade_calc_mode": symbol_info.trade_calc_mode,
            "trade_contract_size": symbol_info.trade_contract_size,
            "trade_exemode": symbol_info.trade_exemode,
            "trade_face_value": symbol_info.trade_face_value,
            "trade_freeze_level": symbol_info.trade_freeze_level,
            "trade_liquidity_rate": symbol_info.trade_liquidity_rate,
            "trade_mode": symbol_info.trade_mode,
            "trade_stops_level": symbol_info.trade_stops_level,
            "trade_tick_size": symbol_info.trade_tick_size,
            "trade_tick_value": symbol_info.trade_tick_value,
            "trade_tick_value_loss": symbol_info.trade_tick_value_loss,
            "trade_tick_value_profit": symbol_info.trade_tick_value_profit,
            "visible": symbol_info.visible,
            "volume": symbol_info.volume,
            "volume_limit": symbol_info.volume_limit,
            "volume_max": symbol_info.volume_max,
            "volume_min": symbol_info.volume_min,
            "volume_real": symbol_info.volume_real,
            "volume_step": symbol_info.volume_step,
            "volumehigh": symbol_info.volumehigh,
            "volumehigh_real": symbol_info.volumehigh_real,
            "volumelow": symbol_info.volumelow,
            "volumelow_real": symbol_info.volumelow_real,
        }
        return pd.Series(d)

    def parse_order_result(self, order_result: mt5.OrderSendResult) -> pd.Series:
        d = {
            "retcode": order_result.retcode,
            "deal": order_result.deal,
            "volume": order_result.volume,
            "price": order_result.price,
            "bid": order_result.bid,
            "ask": order_result.ask,
            "comment": order_result.comment,
            "request_id": order_result.request_id,
            "retcode_external": order_result.retcode_external,
        }
        code, des = d["retcode"], d["comment"]
        print(f"Order status: {des}({code})")
        return pd.Series(d)

    def send_order(self, request: dict) -> pd.Series:
        result = mt5.order_send(request)
        code, des = mt5.last_error()
        print(f"Error status: {des}({code})")
        return self.parse_order_result(result)

    def get_position(self) -> None:
        position = mt5.positions_get()
        position_df = pd.DataFrame(
            position,
            columns=[
                "ticket",
                "time",
                "time_msc",
                "time_update",
                "time_update_msc",
                "type",
                "magic",
                "identifier",
                "reason",
                "volume",
                "price_open",
                "sl",
                "tp",
                "price_current",
                "swap",
                "profit",
                "symbol",
                "comment",
                "external_id",
            ],
        ).drop(
            columns=[
                "time_msc",
                "time_update",
                "time_update_msc",
            ]
        )
        position_df["time"] = pd.to_datetime(position_df["time"], unit="s")
        return position_df.sort_values("time", ascending=True)

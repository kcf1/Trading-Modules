from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import timezone as tz
from datetime import datetime as dt

from ..interfaces import User, DataBase

import MetaTrader5 as mt5


class BookManager:
    def __init__(self, user: User):
        self.user = user
        pass

    def get_book(self) -> pd.DataFrame:
        book = self.user.get_position()
        return book

    def get_strat_pos(self, strat_code: str) -> pd.DataFrame:
        book = self.user.get_position()
        strat_pos = book.loc[book.comment == strat_code]
        return strat_pos

    def get_symbol_pos(self, symbol: str) -> pd.DataFrame:
        book = self.user.get_position()
        symbol_pos = book.loc[book.symbol == symbol]
        return symbol_pos


class OrderManager:
    def __init__(self, user: User, db: DataBase):
        self.user = user
        self.db = db
        self.book = BookManager(self.user)
        pass

    def open_market(
        self,
        symbol: str,
        side: int,
        n_lots: float,
        comment: str,
        tp: float = None,
        sl: float = None,
    ) -> pd.Series:
        assert round(n_lots, 2) != 0, "Minimum 0.01 lots"
        assert len(comment) > 0, "Please input comment"
        info = self.user.get_symbol(symbol)
        digits, price = info.digits, (info.bid + info.ask) / 2
        assert (
            round(sl, digits) - price != 0 and round(tp, digits) - price != 0
        ), "Stop-loss, take-profit cannot be the same as the close price"
        assert side == 1 or side == -1, "Side must be in {1,-1}"

        print(
            f"    side: {int(side):+d}, n_lots: {n_lots:.3f}, tp: {round(tp,digits)}, sl: {round(sl,digits)}, ticket: -"
        )
        print(f"    comment: {comment}")

        if side == 1:
            order_type = mt5.ORDER_TYPE_BUY
        elif side == -1:
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(n_lots, 2),
            "type": order_type,
            "comment": comment,
        }
        if sl is not None:
            request["sl"] = round(sl, digits)
        if tp is not None:
            request["tp"] = round(tp, digits)

        result = self.user.send_order(request)
        print(f"    price: {result.price}, bid: {result.bid}, ask: {result.ask}")
        print(f"    comment: {result.comment}")
        return result

    def close_market(
        self,
        symbol: str,
        ticket: int,
        side: int,
        n_lots: float,
        comment: str,
    ) -> pd.Series:
        assert round(n_lots, 2) != 0, "Minimum 0.01 lots"
        assert len(comment) > 0, "Please input comment"
        info = self.user.get_symbol(symbol)
        # digits, price = info.digits, (info.bid + info.ask) / 2
        assert side == 1 or side == -1, "Side must be in {1,-1}"

        print(
            f"    side: {int(side)}, n_lots: {n_lots:.3f}, tp: -, sl: -, ticket: {ticket}"
        )
        print(f"    comment: {comment}")

        if side == 1:
            order_type = mt5.ORDER_TYPE_BUY
        elif side == -1:
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": ticket,
            "volume": round(n_lots, 2),
            "type": order_type,
            "comment": comment,
        }

        result = self.user.send_order(request)
        print(f"    price: {result.price}, bid: {result.bid}, ask: {result.ask}")
        print(f"    comment: {result.comment}")
        return result

    def close_exceed_time(self, symbol: str, holding_time: timedelta) -> pd.DataFrame:
        pos = self.book.get_symbol_pos(symbol)
        now = dt.now(tz.utc).replace(tzinfo=None)
        pos["held_time"] = pos.time.map(lambda t: now - t)

        pos_exceed_time = pos.loc[pos.held_time >= holding_time]

        if len(pos_exceed_time)>0:
            result_l = list()
            for i, r in pos_exceed_time.iterrows():
                symbol = symbol
                ticket = r.ticket
                # parse type to opposite side
                side = r.type * 2 - 1
                n_lots = r.volume
                comment = "999999 - Close all"
                print(ticket, side, n_lots, comment)
                result_l.append(self.close_market(symbol, ticket, side, n_lots, comment))
            result = pd.concat(result_l, axis=1).T
        else:
            result = None
        return result

    def close_all_market(self, symbol: str) -> pd.DataFrame:
        pos = self.book.get_symbol_pos(symbol)
        result_l = list()
        for i, r in pos.iterrows():
            symbol = symbol
            ticket = r.ticket
            # parse type to opposite side
            side = r.type * 2 - 1
            n_lots = r.volume
            comment = "999999"
            print(ticket, side, n_lots, comment)
            result_l.append(self.close_market(symbol, ticket, side, n_lots, comment))
        result = pd.concat(result_l, axis=1).T
        return result

    def close(self) -> None:
        self.db.close()
        self.user.shutdown()

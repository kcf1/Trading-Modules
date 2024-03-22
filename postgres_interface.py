import pandas as pd
import numpy as np
import psycopg2 as pg
import time


class DataBase:
    def __init__(self):
        self.host = None
        self.database = None
        self.port = None
        self.user = None
        self.password = None
        self.conn = None
        self.cur = None
        pass

    def connect_by_dict(self, d: dict) -> None:
        self.host = d["host"]
        self.database = d["database"]
        self.port = d["port"]
        self.user = d["user"]
        self.password = d["password"]

        try:
            self.conn = pg.connect(
                host=self.host,
                database=self.database,
                port=self.port,
                user=self.user,
                password=self.password,
            )
            self.cur = self.conn.cursor()
            print(f"Connected to {self.database}")
        except Exception as e:
            print(f"Failed to connect to {self.database}")
            print(e)

    def create_table_from_df(self, table_name: str, df: pd.DataFrame) -> None:
        col_type = df.dtypes.astype(str)
        col_spec = col_type.map(self.map_type).to_dict()
        self.create_table(table_name, col_spec)
        self.insert_rows_from_df(table_name, df)

    def create_table(self, table_name: str, col_spec: dict) -> None:
        col_cmds = ""
        for col, spec in col_spec.items():
            if col == "id":
                col_cmds += f"  {col} {spec} PRIMARY KEY,\n"
            else:
                col_cmds += f"  {col} {spec},\n"
        cmd = f"""
CREATE TABLE {table_name} (
{col_cmds[:-2]}
);
        """
        self.cur.execute(cmd.lower())

    def map_type(self, dtype: str) -> str:
        mapper = {
            "datetime64[ns]": "timestamp",
            "float64": "double precision",
            "uint64": "bigint",
            "int32": "integer",
            "object": "text",
        }
        return mapper[dtype]

    def cast_type(self, column: pd.Series) -> pd.Series:
        dtype = self.map_type(str(column.dtype))
        casted_column = column.map(lambda v: f"'{str(v)}'::{dtype}")
        return casted_column

    def insert_rows_from_df(self, table_name: str, df: pd.DataFrame) -> None:
        columns = df.columns.to_list()

        df_casted = df.apply(self.cast_type)

        for row in df_casted.itertuples(index=False, name=None):
            self.insert_row(table_name, columns, list(row))

    def insert_row(self, table_name: str, columns: list, values: list) -> None:
        column_value = ",".join(columns)
        value_value = ",".join(values)

        cmd = f"""
INSERT INTO {table_name} ({column_value})
SELECT {value_value};
      """
        # print(cmd)
        self.cur.execute(cmd.lower())

    def read_columns(self, table_name: str) -> list:
        cmd = f"""
SELECT column_name
FROM information_schema.columns
WHERE table_name = '{table_name}';
        """
        self.cur.execute(cmd.lower())
        # time.sleep(1)
        columns = pd.Series(self.cur.fetchall()).map(lambda t: t[0]).to_list()
        return columns

    def read_all(self, table_name: str) -> pd.DataFrame:
        columns = self.read_columns(table_name)
        cmd = f"""
SELECT * FROM {table_name};
        """
        self.cur.execute(cmd.lower())
        # time.sleep(1)
        df = pd.DataFrame(self.cur.fetchall(), columns=columns)
        return df

    def read_tail(self, table_name: str, n_rows: int = 5) -> pd.DataFrame:
        columns = self.read_columns(table_name)
        # print(columns)
        cmd = f"""
SELECT * FROM {table_name} ORDER BY id DESC LIMIT {n_rows};
        """
        self.cur.execute(cmd.lower())
        df = pd.DataFrame(self.cur.fetchall(), columns=columns)
        return df

    def execute(self, sql: str) -> None:
        self.cur.execute(sql)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.cur.close()
        self.conn.close()

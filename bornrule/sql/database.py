import pandas as pd
from uuid import uuid1
from hashlib import md5
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine, inspect, text, MetaData, Table, Column, Integer, Float, Boolean


class Database:

    LOG = 'LOG'
    SUM = 'SUM'
    POW = 'POW'

    TABLE_PARAMS = "params"
    TABLE_CORPUS = "corpus"
    TABLE_WEIGHTS = "weights"

    n = FIELD_ITEM = "item"
    i = FIELD_LABEL = "label"
    j = FIELD_FEATURE = "feature"
    w = FIELD_WEIGHT = "weight"

    def __init__(self, engine: Engine, prefix, type_features, type_labels):
        self.prefix = prefix
        self.type_features = type_features
        self.type_labels = type_labels

        if isinstance(engine, str):
            self.engine = create_engine(engine, echo=False)
        else:
            self.engine = engine

        self.table_params = Table(
            f"{self.prefix}_{self.TABLE_PARAMS}", MetaData(),
            Column('a', Float),
            Column('b', Float),
            Column('h', Float),
            Column('deployed', Boolean, primary_key=True),
        )

        self.table_corpus = Table(
            f"{self.prefix}_{self.TABLE_CORPUS}", MetaData(),
            Column(self.FIELD_LABEL, self.type_labels, primary_key=True),
            Column(self.FIELD_FEATURE, self.type_features, primary_key=True),
            Column(self.FIELD_WEIGHT, Float),
        )

        self.table_weights = Table(
            f"{self.prefix}_{self.TABLE_WEIGHTS}", MetaData(),
            Column(self.FIELD_LABEL, self.type_labels, primary_key=True),
            Column(self.FIELD_FEATURE, self.type_features, primary_key=True),
            Column(self.FIELD_WEIGHT, Float),
        )

        self.deployed = self.exists(self.table_weights)

    def VAL(self):
        return f"""
            SELECT 
                a, b, h
            FROM
                {self.table_params}
            WHERE
                deployed = {self.deployed}
            """

    def LN(self):
        return f"""
            SELECT 
                {self.LOG}(COUNT(*)) AS {self.w}
            FROM 
                C_i
            """

    def C_i(self):
        return f"""
            SELECT 
                {self.i}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                {self.table_corpus}
            GROUP BY 
                {self.i}
            """

    def C_j(self):
        return f"""
            SELECT 
                {self.j}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                {self.table_corpus}
            GROUP BY 
                {self.j}
            """

    def C_ij(self):
        return f"""
            SELECT 
                {self.table_corpus}.{self.i}, 
                {self.table_corpus}.{self.j}, 
                {self.table_corpus}.{self.w} 
                    * {self.POW}(C_i.{self.w}, - VAL.b) 
                    * {self.POW}(C_j.{self.w}, VAL.b - 1) 
                    AS {self.w}
            FROM 
                {self.table_corpus}, C_i, C_j, VAL
            WHERE 
                {self.table_corpus}.{self.i} = C_i.{self.i} AND 
                {self.table_corpus}.{self.j} = C_j.{self.j}
            """

    def P_j(self):
        return f"""
            SELECT 
                {self.j}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                C_ij
            GROUP BY 
                {self.j}
            """

    def P_ij(self):
        return f"""
            SELECT 
                C_ij.{self.i}, 
                C_ij.{self.j}, 
                C_ij.{self.w} / P_j.{self.w} AS {self.w}
            FROM 
                C_ij, P_j
            WHERE 
                C_ij.{self.j} = P_j.{self.j}
            """

    def H_j(self):
        return f"""
            SELECT 
                P_ij.{self.j}, 
                1 + {self.SUM}(
                    P_ij.{self.w} * {self.LOG}(P_ij.{self.w}) / LN.{self.w}
                ) AS {self.w}
            FROM 
                P_ij, LN
            GROUP BY 
                {self.j}
            """

    def W_ij(self):
        return f"""
            SELECT 
                C_ij.{self.i}, 
                C_ij.{self.j}, 
                {self.POW}(C_ij.{self.w}, VAL.a) 
                    * {self.POW}(H_j.{self.w}, VAL.h) 
                    AS {self.w}
            FROM 
                C_ij, H_j, VAL
            WHERE 
                C_ij.{self.j} = H_j.{self.j}
            """

    def X_ni(self, table_items):
        return f"""
            SELECT 
                {table_items}.{self.n}, 
                W_ij.{self.i}, 
                {self.POW}(
                    {self.SUM}(
                        W_ij.{self.w} * {self.POW}({table_items}.{self.w}, VAL.a)
                    )
                    , 1 / VAL.a
                ) AS {self.w} 
            FROM 
                {table_items}, W_ij, VAL
            WHERE 
                {table_items}.{self.j} = W_ij.{self.j}
            GROUP BY 
                {table_items}.{self.n}, 
                W_ij.{self.i},
                VAL.a
            """

    def X_n(self):
        return f"""
            SELECT 
                X_ni.{self.n}, 
                {self.SUM}(X_ni.{self.w}) AS {self.w}
            FROM 
                X_ni
            GROUP BY 
                X_ni.{self.n}
            """

    def WITH(self):
        if self.deployed:
            sql = f"""
                WITH 
                    VAL AS ({self.VAL()}),
                    W_ij AS (SELECT * FROM {self.table_weights})
                """

        else:
            sql = f"""
                WITH 
                    VAL AS ({self.VAL()}),
                    C_i AS ({self.C_i()}), 
                    C_j AS ({self.C_j()}), 
                    C_ij AS ({self.C_ij()}), 
                    P_j AS ({self.P_j()}), 
                    P_ij AS ({self.P_ij()}),
                    LN AS ({self.LN()}),  
                    H_j AS ({self.H_j()}), 
                    W_ij AS ({self.W_ij()})
                """

        return sql

    def predict(self, table_items):
        return f"""
            {self.WITH()}, 
                X_ni AS ({self.X_ni(table_items)}), 
                R_ni AS (
                    SELECT 
                        {self.n}, 
                        {self.i}, 
                        ROW_NUMBER() OVER(PARTITION BY {self.n} ORDER BY {self.w} DESC) AS idx
                    FROM 
                        X_ni
                )
            SELECT 
                {self.n}, 
                {self.i}
            FROM 
                R_ni 
            WHERE 
                idx = 1
            """

    def predict_proba(self, table_items):
        return f"""
            {self.WITH()}, 
                X_ni AS ({self.X_ni(table_items)}), 
                X_n AS ({self.X_n()}) 
            SELECT 
                X_ni.{self.n}, 
                X_ni.{self.i}, 
                X_ni.{self.w} / X_n.{self.w} AS {self.w}
            FROM 
                X_ni, X_n
            WHERE 
                X_ni.{self.n} = X_n.{self.n}
            """

    def explain(self, table_item):
        if table_item is None:
            sql = f"""
                {self.WITH()} 
                SELECT 
                    W_ij.{self.i}, 
                    W_ij.{self.j},
                    W_ij.{self.w}
                FROM 
                    W_ij
                """

        else:
            sql = f"""
                {self.WITH()}
                SELECT 
                    W_ij.{self.i}, 
                    W_ij.{self.j},
                    {self.POW}({table_item}.{self.w}, VAL.a) * W_ij.{self.w} AS {self.w}
                FROM 
                    {table_item}, W_ij, VAL
                WHERE 
                    {table_item}.{self.j} = W_ij.{self.j}
                """

        return sql

    def connect(self):
        return self.engine.connect()

    def exists(self, table):
        return inspect(self.engine).has_table(table.name)

    @staticmethod
    def drop(con, table):
        table.drop(con, checkfirst=True)

    @staticmethod
    def read(con, sql):
        return pd.read_sql(sql, con)

    def read_params(self, con):
        return self.read(con, f"SELECT a, b, h FROM {self.table_params} WHERE deployed = {self.deployed}")

    def write(self, con, table, values=None, if_exists='fail', **kwargs):
        assert if_exists in {'fail', 'replace', 'insert', 'insert_or_ignore', 'insert_or_replace', 'insert_or_sum'}

        if not self.exists(table):
            table.create(con)

        else:
            if if_exists == 'fail':
                raise ValueError(
                    f"Table {table} already exists."
                )

            if if_exists == 'replace':
                table.drop(con)
                table.create(con)

        if values is not None:
            insert = getattr(self, "insert" if if_exists in ['fail', 'replace'] else if_exists)
            insert(con, table, values, **kwargs)

    def write_params(self, con, a, b, h):
        if_exists = {
            'if_exists': 'insert_or_replace',
            'conflict': ['deployed'],
            'replace': ['a', 'b', 'h']
        }

        self.write(con, table=self.table_params, values=[{'a': a, 'b': b, 'h': h, 'deployed': False}], **if_exists)

    def write_corpus(self, con, values):
        if_exists = {
            'if_exists': 'insert_or_sum',
            'confilct': [self.FIELD_LABEL, self.FIELD_FEATURE],
            'sum': [self.FIELD_WEIGHT]
        }

        self.write(con, table=self.table_corpus, values=values, **if_exists)

    @staticmethod
    def insert(con, table, values):
        keys = values[0].keys()
        sql = f"""
            INSERT INTO {table} ({','.join(keys)}) 
            VALUES(:{', :'.join(keys)})"""

        return con.execute(text(sql), values)

    @staticmethod
    def insert_or_ignore(con, table, values):
        keys = values[0].keys()
        sql = f"""
            INSERT INTO {table} ({','.join(keys)}) 
            VALUES(:{', :'.join(keys)}) 
            ON CONFLICT DO NOTHING
            """

        return con.execute(text(sql), values)

    @staticmethod
    def insert_or_replace(con, table, values, conflict, replace):
        keys = values[0].keys()
        sql = f"""
            INSERT INTO {table} ({','.join(keys)}) 
            VALUES(:{', :'.join(keys)})
            ON CONFLICT ({','.join(conflict)})
            DO UPDATE SET ({','.join(replace)}) = (:{', :'.join(replace)})
            """

        return con.execute(text(sql), values)

    @staticmethod
    def insert_or_sum(con, table, values, confilct, sum):
        keys = values[0].keys()
        sql = f"""
            INSERT INTO {table} ({','.join(keys)}) 
            VALUES(:{', :'.join(keys)}) 
            ON CONFLICT ({','.join(confilct)}) 
            DO UPDATE SET {", ".join([f"{s} = {table}.{s} + excluded.{s}" for s in sum])}
            """

        return con.execute(text(sql), values)

    def tmp_key(self):
        return f"tmp_{self.prefix}_{md5(str(uuid1()).encode()).hexdigest()[:12]}"

    def tmp_item(self):
        return Table(
            self.tmp_key(), MetaData(),
            Column(self.FIELD_FEATURE, self.type_features, primary_key=True),
            Column(self.FIELD_WEIGHT, Float),
            prefixes=["TEMPORARY"],
        )

    def tmp_items(self):
        return Table(
            self.tmp_key(), MetaData(),
            Column(self.FIELD_ITEM, Integer, primary_key=True),
            Column(self.FIELD_FEATURE, self.type_features, primary_key=True),
            Column(self.FIELD_WEIGHT, Float),
            prefixes=["TEMPORARY"],
        )

    def deploy(self):
        if not self.exists(self.table_corpus):
            raise ValueError(
                "This instance is not fitted yet. Cannot deploy an empty instance."
            )

        if self.deployed:
            raise ValueError(
                "This instance is already deployed. Nothing to do."
            )

        with self.connect() as con:
            with con.begin():
                self.write(con, table=self.table_weights)
                con.execute(f"{self.WITH()} INSERT INTO {self.table_weights} SELECT * FROM W_ij")
                con.execute(f"UPDATE {self.table_params} SET deployed = true WHERE deployed = false")

        self.deployed = True

    def undeploy(self):
        if not self.deployed:
            raise ValueError(
                "This instance is already undeployed. Nothing to do."
            )

        with self.connect() as con:
            with con.begin():
                self.drop(con, table=self.table_weights)
                con.execute(f"UPDATE {self.table_params} SET deployed = false WHERE deployed = true")

        self.deployed = False

    def check_undeployed(self):
        if self.deployed:
            raise ValueError(
                "Cannot modify a deployed instance."
            )

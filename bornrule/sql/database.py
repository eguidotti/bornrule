import pandas as pd
from uuid import uuid1
from hashlib import md5
from collections import defaultdict
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine, inspect, text, MetaData, Table, Column, Integer, Float, String


class Database:

    LOG = 'LOG'
    SUM = 'SUM'
    POW = 'POW'

    def __init__(self,
                 id,
                 engine,
                 type_feature,
                 type_class,
                 field_id,
                 field_item,
                 field_feature,
                 field_class,
                 field_weight,
                 table_params,
                 table_corpus,
                 table_weights):

        self.engine = engine

        self.id = id
        self.field_id = field_id

        self.type_feature = type_feature
        self.type_class = type_class

        self.j = self.field_feature = field_feature
        self.k = self.field_class = field_class
        self.n = self.field_item = field_item
        self.w = self.field_weight = field_weight

        self.table_params = Table(
            table_params, MetaData(),
            Column(self.field_id, String, primary_key=True),
            Column('a', Float, nullable=False),
            Column('b', Float, nullable=False),
            Column('h', Float, nullable=False),
        )

        self.table_corpus = Table(
            f"{self.id}_{table_corpus}", MetaData(),
            Column(self.field_feature, self.type_feature, primary_key=True),
            Column(self.field_class, self.type_class, primary_key=True),
            Column(self.field_weight, Float, nullable=False),
        )

        self.table_weights = Table(
            f"{self.id}_{table_weights}", MetaData(),
            Column(self.field_feature, self.type_feature, primary_key=True),
            Column(self.field_class, self.type_class, primary_key=True),
            Column(self.field_weight, Float, nullable=False),
        )

        self.table_temp = lambda *args : Table(
            f"temp_{md5(str(uuid1()).encode()).hexdigest()[:12]}", MetaData(), *args, prefixes=["TEMPORARY"]
        )

    def connect(self):
        return self.engine.connect()

    @staticmethod
    def exists(con, table):
        return inspect(con).has_table(table.name)

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
    def insert_or_sum(con, table, values, conflict, sum):
        keys = values[0].keys()
        sql = f"""
            INSERT INTO {table} ({','.join(keys)}) 
            VALUES(:{', :'.join(keys)}) 
            ON CONFLICT ({','.join(conflict)}) 
            DO UPDATE SET {", ".join([f"{s} = {table}.{s} + excluded.{s}" for s in sum])}
            """

        return con.execute(text(sql), values)

    def read_params(self, con):
        if self.exists(con, self.table_params):
            sql = f"""
                SELECT * 
                FROM {self.table_params} 
                WHERE {self.field_id}='{self.id}'
                """

            cursor = con.execute(text(sql))
            values = cursor.fetchone()
            keys = cursor.keys()

            if values:
                params = dict(zip(keys, values))
                params.pop(self.field_id)
                return params

        return None

    def read_sql(self, sql, con):
        cur = con.execute(text(sql) if isinstance(sql, str) else sql)
        data = [tuple(row) for row in cur.fetchall()]
        columns = cur.keys()

        return pd.DataFrame(data, columns=columns)

    def write(self, con, table, values=None, if_exists='fail', **kwargs):
        assert if_exists in {
            'fail',
            'replace',
            'insert',
            'insert_or_ignore',
            'insert_or_replace',
            'insert_or_sum'
        }

        if not self.exists(con, table):
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

        return table

    def write_params(self, con, **kwargs):
        if_exists = {
            'if_exists': 'insert_or_replace',
            'conflict': [self.field_id],
            'replace': list(kwargs.keys())
        }

        return self.write(con, table=self.table_params, values=[{self.field_id: self.id, **kwargs}], **if_exists)

    def write_corpus(self, con, X, y, sample_weight):
        if_exists = {
            'if_exists': 'insert_or_sum',
            'conflict': [self.field_class, self.field_feature],
            'sum': [self.field_weight]
        }

        corpus = defaultdict(lambda: defaultdict(int))
        for x, y, w in zip(X, y, sample_weight):
            if not isinstance(y, dict):
                y = {y: 1}

            n = sum(x.values())
            for k, p in y.items():
                for f, v in x.items():
                    corpus[k][f] += w * p * v / n

        values = []
        for c, d in corpus.items():
            for f, w in d.items():
                values.append({self.field_feature: f, self.field_class: c, self.field_weight: w})

        return self.write(con, table=self.table_corpus, values=values, **if_exists)

    def write_items(self, con, X):
        table = self.table_temp(
            Column(self.field_item, Integer, primary_key=True),
            Column(self.field_feature, self.type_feature, primary_key=True),
            Column(self.field_weight, Float),
        )

        values = [
            {self.field_item: i, self.field_feature: f, self.field_weight: w}
            for i, x in enumerate(X)
            for f, w in x.items()
        ]

        return self.write(con, table=table, values=values)

    def is_params(self, con):
        return self.read_params(con) is not None

    def is_corpus(self, con):
        return self.exists(con, self.table_corpus)

    def is_deployed(self, con):
        return self.exists(con, self.table_weights)

    def is_fitted(self, con):
        return self.is_corpus(con) or self.is_deployed(con)

    def check_fitted(self, con):
        if not self.is_fitted(con):
            raise ValueError(
                f"This instance is not fitted yet."
            )

    def check_editable(self, con):
        if self.is_deployed(con):
            raise ValueError(
                "Cannot modify a deployed instance."
            )

    def deploy(self, con, deep):
        self.check_fitted(con)

        if self.is_deployed(con):
            raise ValueError(
                "This instance is already deployed. Nothing to do."
            )

        sql = f"""
            INSERT INTO  {self.table_weights} ({self.j}, {self.k}, {self.w})
            {self._sql_WITH(cache=False)} 
            SELECT {self.j}, {self.k}, {self.w} 
            FROM HW_jk
            """

        self.write(con, table=self.table_weights)
        con.execute(text(sql))

        if deep:
            self.table_corpus.drop(con)

    def undeploy(self, con, deep):
        if not deep and not self.exists(con, self.table_corpus):
            raise ValueError(
                "This instance has no corpus and the model would be lost. "
                "Set deep=True to force undeploy."
            )

        if not deep and not self.is_deployed(con):
            raise ValueError(
                "This instance is already undeployed. Nothing to do."
            )

        self.table_weights.drop(con, checkfirst=True)

        if deep:
            self.table_corpus.drop(con, checkfirst=True)

            if self.is_params(con):
                con.execute(text(f"DELETE FROM {self.table_params} WHERE {self.field_id}='{self.id}'"))

                if con.execute(text(f"SELECT COUNT(*) FROM {self.table_params}")).fetchone()[0] == 0:
                    self.table_params.drop(con)

    def predict(self, con, X):
        cache = self.is_deployed(con)
        items = self.write_items(con, X)
        sql = self._sql_predict(cache, items)

        return self.read_sql(sql, con)

    def predict_proba(self, con, X):
        cache = self.is_deployed(con)
        items = self.write_items(con, X)
        sql = self._sql_predict_proba(cache, items)

        return self.read_sql(sql, con)

    def explain(self, con, X=None):
        cache = self.is_deployed(con)
        items = self.write_items(con, X) if X else None
        sql = self._sql_explain(cache, items)

        return self.read_sql(sql, con)

    def _sql_predict(self, cache, items):
        return f"""
            {self._sql_WITH(cache)}, 
                X_njk AS ({self._sql_X_njk(items)}), 
                X_nk AS ({self._sql_X_nk()}), 
                R_nk AS (
                    SELECT 
                        {self.n}, 
                        {self.k}, 
                        ROW_NUMBER() OVER(PARTITION BY {self.n} ORDER BY {self.w} DESC) AS idx
                    FROM 
                        X_nk
                )
            SELECT 
                {self.n}, 
                {self.k}
            FROM 
                R_nk
            WHERE 
                idx = 1
            """

    def _sql_predict_proba(self, cache, items):
        return f"""
            {self._sql_WITH(cache)}, 
                X_njk AS ({self._sql_X_njk(items)}), 
                X_nk AS ({self._sql_X_nk()}), 
                U_nk AS ({self._sql_U_nk()}), 
                U_n AS ({self._sql_U_n()}),
                Y_nk AS ({self._sql_Y_nk()})
            SELECT 
                Y_nk.{self.n}, 
                Y_nk.{self.k}, 
                Y_nk.{self.w}
            FROM 
                Y_nk
            """

    def _sql_explain(self, cache, items=None):
        if items is None:
            return f"""
                {self._sql_WITH(cache)} 
                SELECT 
                    {self.j},
                    {self.k},
                    {self.w}
                FROM 
                    HW_jk
                """

        return f"""
            {self._sql_WITH(cache)}, 
                X_njk AS ({self._sql_X_njk(items)})
            SELECT 
                {self.j},
                {self.k},
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                X_njk
            GROUP BY
                {self.j},
                {self.k}
            """

    def _sql_WITH(self, cache):
        if cache:
            sql = f"""
                WITH 
                    ABH AS ({self._sql_ABH()}),
                    HW_jk AS (SELECT * FROM {self.table_weights})
                """

        else:
            sql = f"""
                WITH 
                    ABH AS ({self._sql_ABH()}),
                    P_j AS ({self._sql_P_j()}), 
                    P_k AS ({self._sql_P_k()}), 
                    W_jk AS ({self._sql_W_jk()}), 
                    W_j AS ({self._sql_W_j()}), 
                    H_jk AS ({self._sql_H_jk()}),
                    LN AS ({self._sql_LN()}),
                    H_j AS ({self._sql_H_j()}), 
                    HW_jk AS ({self._sql_HW_jk()})
                """

        return sql

    def _sql_ABH(self):
        return f"""
            SELECT 
                a, b, h
            FROM
                {self.table_params}
            WHERE
                {self.field_id} = '{self.id}'
            """

    def _sql_LN(self):
        return f"""
            SELECT 
                CASE 
                    WHEN COUNT(*) > 1 
                    THEN {self.LOG}(COUNT(*))
                    ELSE 1
                END AS {self.w}
            FROM 
                P_k
            """

    def _sql_P_k(self):
        return f"""
            SELECT 
                {self.k}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                {self.table_corpus}
            GROUP BY 
                {self.k}
            """

    def _sql_P_j(self):
        return f"""
            SELECT 
                {self.j}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                {self.table_corpus}
            GROUP BY 
                {self.j}
            """

    def _sql_W_jk(self):
        return f"""
            SELECT 
                {self.table_corpus}.{self.j}, 
                {self.table_corpus}.{self.k}, 
                {self.table_corpus}.{self.w} 
                    * {self.POW}(P_k.{self.w}, - ABH.b) 
                    * {self.POW}(P_j.{self.w}, ABH.b - 1) 
                    AS {self.w}
            FROM 
                {self.table_corpus}, P_j, P_k, ABH
            WHERE 
                {self.table_corpus}.{self.j} = P_j.{self.j} AND
                {self.table_corpus}.{self.k} = P_k.{self.k}  
            """

    def _sql_W_j(self):
        return f"""
            SELECT 
                {self.j}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                W_jk
            GROUP BY 
                {self.j}
            """

    def _sql_H_jk(self):
        return f"""
            SELECT 
                W_jk.{self.j}, 
                W_jk.{self.k}, 
                W_jk.{self.w} / W_j.{self.w} AS {self.w}
            FROM 
                W_jk, W_j
            WHERE 
                W_jk.{self.j} = W_j.{self.j}
            """

    def _sql_H_j(self):
        return f"""
            SELECT 
                H_jk.{self.j}, 
                1 + {self.SUM}(
                    H_jk.{self.w} * {self.LOG}(H_jk.{self.w}) / LN.{self.w}
                ) AS {self.w}
            FROM 
                H_jk, LN
            GROUP BY 
                {self.j}
            """

    def _sql_HW_jk(self):
        return f"""
            SELECT 
                W_jk.{self.j}, 
                W_jk.{self.k}, 
                {self.POW}(W_jk.{self.w}, ABH.a) 
                    * {self.POW}(H_j.{self.w}, ABH.h) 
                    AS {self.w}
            FROM 
                W_jk, H_j, ABH
            WHERE 
                W_jk.{self.j} = H_j.{self.j}
            """

    def _sql_X_njk(self, items):
        return f"""
            SELECT 
                {items}.{self.n}, 
                HW_jk.{self.j},
                HW_jk.{self.k}, 
                HW_jk.{self.w} * {self.POW}({items}.{self.w}, ABH.a) AS {self.w} 
            FROM 
                {items}, HW_jk, ABH
            WHERE 
                {items}.{self.j} = HW_jk.{self.j}
            """

    def _sql_X_nk(self):
        return f"""
            SELECT 
                {self.n}, 
                {self.k}, 
                {self.SUM}({self.w}) AS {self.w} 
            FROM 
                X_njk
            GROUP BY
                {self.n}, {self.k}
           """

    def _sql_U_nk(self):
        return f"""
            SELECT 
                X_nk.{self.n}, 
                X_nk.{self.k}, 
                {self.POW}(X_nk.{self.w}, 1 / ABH.a) AS {self.w} 
            FROM 
                X_nk, ABH
            """

    def _sql_U_n(self):
        return f"""
            SELECT 
                {self.n}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                U_nk
            GROUP BY 
                {self.n}
            """

    def _sql_Y_nk(self):
        return f"""
            SELECT 
                U_nk.{self.n}, 
                U_nk.{self.k}, 
                U_nk.{self.w} / U_n.{self.w} AS {self.w}
            FROM 
                U_nk, U_n
            WHERE 
                U_nk.{self.n} = U_n.{self.n}
            """

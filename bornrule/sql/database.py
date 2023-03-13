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

    def __init__(self, engine: Engine, prefix, type_features, type_classes,
                 field_features, field_classes, field_items, field_weights,
                 table_params, table_corpus, table_weights):

        self.prefix = prefix

        self.type_features = type_features
        self.type_classes = type_classes

        self.j = self.field_features = field_features
        self.k = self.field_classes = field_classes
        self.n = self.field_items = field_items
        self.w = self.field_weights = field_weights

        if isinstance(engine, str):
            self.engine = create_engine(engine, echo=False)
        else:
            self.engine = engine

        self.table_params = Table(
            f"{self.prefix}_{table_params}", MetaData(),
            Column('prefix', String, primary_key=True),
            Column('a', Float, nullable=False),
            Column('b', Float, nullable=False),
            Column('h', Float, nullable=False),
        )

        self.table_corpus = Table(
            f"{self.prefix}_{table_corpus}", MetaData(),
            Column(self.field_features, self.type_features, primary_key=True),
            Column(self.field_classes, self.type_classes, primary_key=True),
            Column(self.field_weights, Float, nullable=False),
        )

        self.table_weights = Table(
            f"{self.prefix}_{table_weights}", MetaData(),
            Column(self.field_features, self.type_features, primary_key=True),
            Column(self.field_classes, self.type_classes, primary_key=True),
            Column(self.field_weights, Float, nullable=False),
        )

    def connect(self):
        return self.engine.connect()

    def exists(self, con, table):
        return inspect(con).has_table(table.name)

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
        sql = f"""
            SELECT * 
            FROM {self.table_params} 
            WHERE prefix='{self.prefix}'
            """

        cursor = con.execute(text(sql))
        values = cursor.fetchone()
        keys = cursor.keys()

        params = dict(zip(keys, values))
        params.pop('prefix')

        return params

    def read_sql(self, sql, con):
        cur = con.execute(text(sql) if isinstance(sql, str) else sql)
        data = [tuple(row) for row in cur.fetchall()]
        columns = cur.keys()

        return pd.DataFrame(data, columns=columns)

    def write_params(self, con, **kwargs):
        if_exists = {
            'if_exists': 'insert_or_replace',
            'conflict': ['prefix'],
            'replace': list(kwargs.keys())
        }

        return self.write(con, table=self.table_params, values=[dict(prefix=self.prefix, **kwargs)], **if_exists)

    def write_corpus(self, con, X, y, sample_weight):
        if_exists = {
            'if_exists': 'insert_or_sum',
            'conflict': [self.field_classes, self.field_features],
            'sum': [self.field_weights]
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
                values.append({self.field_features: f, self.field_classes: c, self.field_weights: w})

        return self.write(con, table=self.table_corpus, values=values, **if_exists)

    def write_items(self, con, X):
        table = Table(
            f"{self.prefix}_items_{md5(str(uuid1()).encode()).hexdigest()[:12]}", MetaData(),
            Column(self.field_items, Integer, primary_key=True),
            Column(self.field_features, self.type_features, primary_key=True),
            Column(self.field_weights, Float),
            prefixes=["TEMPORARY"],
        )

        values = [
            {self.field_items: i, self.field_features: f, self.field_weights: w}
            for i, x in enumerate(X)
            for f, w in x.items()
        ]

        return self.write(con, table=table, values=values)

    def write_sample_weight(self, con, sample_weight):
        table = Table(
            f"{self.prefix}_sample_weight_{md5(str(uuid1()).encode()).hexdigest()[:12]}", MetaData(),
            Column(self.field_items, Integer, primary_key=True),
            Column(self.field_weights, Float),
            prefixes=["TEMPORARY"],
        )

        values = [
            {self.field_items: i, self.field_weights: w}
            for i, w in enumerate(sample_weight)
        ]

        return self.write(con, table=table, values=values)

    def is_params(self, con):
        return self.exists(con, self.table_params)

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
            INSERT INTO  {self.table_weights} 
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
            self.table_params.drop(con, checkfirst=True)
            self.table_corpus.drop(con, checkfirst=True)

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

    def explain(self, con, X=None, sample_weight=None):
        cache = self.is_deployed(con)

        if X is None:
            sql = self._sql_explain(cache)

        else:
            norm = [sum(v for k, v in x.items()) for x in X]

            if sample_weight is None:
                X = [{k: v / n for k, v in x.items()} for x, n in zip(X, norm) if n > 0]
            else:
                p = 1. / self.read_params(con)['a']
                X = [{k: pow(w, p) * v / n for k, v in x.items()} for x, n, w in zip(X, norm, sample_weight) if n > 0]

            items = self.write_items(con, X)
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

        # return f"""
        #     {self._sql_WITH(cache)},
        #         X_njk AS ({self._sql_X_njk(items)}),
        #         X_nk AS ({self._sql_X_nk()}),
        #         U_nk AS ({self._sql_U_nk()}),
        #         U_n AS ({self._sql_U_n()}),
        #         Y_nk AS ({self._sql_Y_nk()}),
        #         U_njk AS ({self._sql_U_njk(items)}),
        #         U_nj AS ({self._sql_U_nj()}),
        #         Y_njk AS ({self._sql_Y_njk()})
        #     SELECT
        #         Y_njk.{self.j},
        #         Y_njk.{self.k},
        #         {self.SUM}(
        #             (Y_nk.{self.w} - Y_njk.{self.w})
        #             {'' if weights is None else f' * {weights}.{self.w}'}
        #         ) AS {self.w}
        #     FROM
        #         Y_njk, Y_nk {'' if weights is None else f', {weights}'}
        #     WHERE
        #         Y_njk.{self.n} = Y_nk.{self.n} AND
        #         Y_njk.{self.k} = Y_nk.{self.k}
        #         {'' if weights is None else f'AND Y_nk.{self.n} = {weights}.{self.n}'}
        #     GROUP BY
        #         Y_njk.{self.j},
        #         Y_njk.{self.k}
        #     """

    def _sql_WITH(self, cache):
        if cache:
            sql = f"""
                WITH 
                    VAL AS ({self._sql_VAL()}),
                    HW_jk AS (SELECT * FROM {self.table_weights})
                """

        else:
            sql = f"""
                WITH 
                    VAL AS ({self._sql_VAL()}),
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

    def _sql_VAL(self):
        return f"""
            SELECT 
                a, b, h
            FROM
                {self.table_params}
            WHERE
                prefix = '{self.prefix}'
            """

    def _sql_LN(self):
        return f"""
            SELECT 
                {self.LOG}(COUNT(*)) AS {self.w}
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
                    * {self.POW}(P_k.{self.w}, - VAL.b) 
                    * {self.POW}(P_j.{self.w}, VAL.b - 1) 
                    AS {self.w}
            FROM 
                {self.table_corpus}, P_j, P_k, VAL
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
                {self.POW}(W_jk.{self.w}, VAL.a) 
                    * {self.POW}(H_j.{self.w}, VAL.h) 
                    AS {self.w}
            FROM 
                W_jk, H_j, VAL
            WHERE 
                W_jk.{self.j} = H_j.{self.j}
            """

    def _sql_X_njk(self, items):
        return f"""
            SELECT 
                {items}.{self.n}, 
                HW_jk.{self.j},
                HW_jk.{self.k}, 
                HW_jk.{self.w} * {self.POW}({items}.{self.w}, VAL.a) AS {self.w} 
            FROM 
                {items}, HW_jk, VAL
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
                {self.POW}(X_nk.{self.w}, 1 / VAL.a) AS {self.w} 
            FROM 
                X_nk, VAL
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

    def _sql_U_njk(self, items):
        return f"""
            SELECT 
                X_nk.{self.n}, 
                {items}.{self.j}, 
                X_nk.{self.k}, 
                {self.POW}(X_nk.{self.w} - COALESCE(X_njk.{self.w}, 0), 1 / VAL.a) AS {self.w} 
            FROM 
                X_nk 
                JOIN {items} ON 
                    X_nk.{self.n} = {items}.{self.n} 
                LEFT JOIN X_njk ON 
                    X_njk.{self.n} = X_nk.{self.n} AND 
                    X_njk.{self.k} = X_nk.{self.k} AND 
                    X_njk.{self.j} = {items}.{self.j}  
                JOIN VAL
            """

    def _sql_U_nj(self):
        return f"""
            SELECT 
                {self.n}, 
                {self.j}, 
                {self.SUM}({self.w}) AS {self.w}
            FROM 
                U_njk
            GROUP BY 
                {self.n},
                {self.j}
            """

    def _sql_Y_njk(self):
        return f"""
            SELECT 
                U_njk.{self.n}, 
                U_njk.{self.j}, 
                U_njk.{self.k}, 
                U_njk.{self.w} / U_nj.{self.w} AS {self.w}
            FROM 
                U_njk, U_nj
            WHERE 
                U_njk.{self.n} = U_nj.{self.n} AND
                U_njk.{self.j} = U_nj.{self.j}
            """

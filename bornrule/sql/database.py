import pandas as pd
from uuid import uuid1
from hashlib import md5
from collections import defaultdict
from sqlalchemy import inspect, text, MetaData, Table, Column, Integer, Float, String


class Database:

    LOG = 'LOG'
    SUM = 'SUM'
    POW = 'POW'

    CONCAT_FUN = ''
    CONCAT_SEP = '||'

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

        self.default_params = {
            'a': 0.5,
            'b': 1,
            'h': 1,
        }

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

        self.table_temp = lambda *args: Table(
            f"temp_{md5(str(uuid1()).encode()).hexdigest()[:12]}", MetaData(), *args, prefixes=["TEMPORARY"]
        )

    def connect(self):
        return self.engine.connect()

    @staticmethod
    def exists(con, table):
        return inspect(con).has_table(table.name)

    @staticmethod
    def insert(con, table, values, columns=None):
        from_select = isinstance(values, str)
        keys = columns if from_select else values[0].keys()
        records = values if from_select else f"VALUES(:{', :'.join(keys)})"
        params = None if from_select else values
        sql = f"""
            INSERT INTO {table} ({','.join(keys)})
            {records}
            """

        return con.execute(text(sql), params)

    @staticmethod
    def insert_or_ignore(con, table, values, columns=None):
        from_select = isinstance(values, str)
        keys = columns if from_select else values[0].keys()
        records = values if from_select else f"VALUES(:{', :'.join(keys)})"
        params = None if from_select else values
        sql = f"""
            INSERT INTO {table} ({','.join(keys)})
            {records}
            ON CONFLICT DO NOTHING
            """

        return con.execute(text(sql), params)

    @staticmethod
    def insert_or_replace(con, table, values, conflict, replace, columns=None):
        from_select = isinstance(values, str)
        keys = columns if from_select else values[0].keys()
        records = values if from_select else f"VALUES(:{', :'.join(keys)})"
        params = None if from_select else values
        sql = f"""
            INSERT INTO {table} ({','.join(keys)})
            {records}
            ON CONFLICT ({','.join(conflict)})
            DO UPDATE SET ({','.join(replace)}) = (:{', :'.join(replace)})
            """

        return con.execute(text(sql), params)

    @staticmethod
    def insert_or_sum(con, table, values, conflict, sum, columns=None):
        from_select = isinstance(values, str)
        keys = columns if from_select else values[0].keys()
        records = values if from_select else f"VALUES(:{', :'.join(keys)})"
        params = None if from_select else values
        sql = f"""
            INSERT INTO {table} ({','.join(keys)})
            {records}
            ON CONFLICT ({','.join(conflict)}) 
            DO UPDATE SET {", ".join([f"{s} = {table}.{s} + excluded.{s}" for s in sum])}
            """

        return con.execute(text(sql), params)

    def read_params(self, con):
        if self.exists(con, self.table_params):
            sql = f"""
                SELECT {','.join(self.default_params.keys())} 
                FROM {self.table_params} 
                WHERE {self.field_id}='{self.id}'
                """

            cursor = con.execute(text(sql))
            values = cursor.fetchone()
            keys = cursor.keys()

            if values:
                return dict(zip(keys, values))

        return self.default_params

    def read_sql(self, sql, con):
        cur = con.execute(text(sql) if isinstance(sql, str) else sql)
        data = [tuple(row) for row in cur.fetchall()]
        columns = cur.keys()

        return pd.DataFrame(data, columns=columns)

    def write(self, con, table, values=None, if_exists='fail', **kwargs):
        assert if_exists in {
            'fail',
            'replace',
            'ignore',
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

            if if_exists == 'ignore':
                values = None

        if values is not None:
            insert = getattr(self, "insert" if if_exists in ['fail', 'replace', 'ignore'] else if_exists)
            insert(con, table, values, **kwargs)

        return table

    def write_params(self, con, **params):
        if_exists = {
            'if_exists': 'insert_or_replace',
            'conflict': [self.field_id],
            'replace': list(params.keys())
        }

        return self.write(con, table=self.table_params, values=[{self.field_id: self.id, **params}], **if_exists)

    def write_corpus(self, con, X, y, sample_weight):
        if_exists = {
            'if_exists': 'insert_or_sum',
            'columns': [self.field_feature, self.field_class, self.field_weight],
            'conflict': [self.field_class, self.field_feature],
            'sum': [self.field_weight]
        }

        if isinstance(X, dict) and y is None:
            values = self._sql_partial_fit(items=X, sample_weight=sample_weight)

        else:
            if sample_weight is None:
                sample_weight = [1] * len(X)

            corpus = defaultdict(lambda: defaultdict(int))
            for x, y, w in zip(X, y, sample_weight):
                if not isinstance(y, dict):
                    y = {y: 1}

                n = sum(x.values()) * sum(y.values())
                if n != 0:
                    for k, p in y.items():
                        for f, v in x.items():
                            corpus[k][f] += w * p * v / n

            values = []
            for c, d in corpus.items():
                for f, w in d.items():
                    values.append({self.field_feature: f, self.field_class: c, self.field_weight: w})

        self.write(con, table=self.table_corpus, values=values, **if_exists)
        
        sql = f"DELETE FROM {self.table_corpus} WHERE {self.w} = 0"
        con.execute(text(sql))
        
        sql = f"SELECT * FROM {self.table_corpus} WHERE {self.w} < 0"
        if con.execute(text(sql)).fetchone():
            raise ValueError(
                f"Negative values are not allowed in the corpus."
            )

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
        sql = f"SELECT COUNT(*) FROM {self.table_params} WHERE {self.field_id}='{self.id}'"
        return self.exists(con, self.table_params) and con.execute(text(sql)).fetchone()[0]

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
                "This instance is already deployed."
            )

        self.write(
            con,
            table=self.table_weights,
            columns=[self.j, self.k, self.w],
            values=f"{self._sql_with_HW_jk(cache=False)} SELECT {self.j}, {self.k}, {self.w} FROM HW_jk"
        )

        if deep:
            self.table_corpus.drop(con)

    def undeploy(self, con, deep):
        if not deep and not self.exists(con, self.table_corpus):
            raise ValueError(
                "This instance has no corpus and the model would be lost."
            )

        if not deep and not self.is_deployed(con):
            raise ValueError(
                "This instance is not deployed."
            )

        self.table_weights.drop(con, checkfirst=True)

        if deep:
            self.table_corpus.drop(con, checkfirst=True)

            if self.is_params(con):
                con.execute(text(f"DELETE FROM {self.table_params} WHERE {self.field_id}='{self.id}'"))

                if con.execute(text(f"SELECT COUNT(*) FROM {self.table_params}")).fetchone()[0] == 0:
                    self.table_params.drop(con)

    def partial_fit(self, con, X, y, sample_weight):
        self.write_corpus(con, X=X, y=y, sample_weight=sample_weight)
        if not self.is_params(con):
            self.write_params(con, **self.default_params)

    def predict(self, con, X):
        cache = self.is_deployed(con)
        items = X if isinstance(X, dict) else self.write_items(con, X)
        sql = self._sql_predict(items, cache)

        return self.read_sql(sql, con)

    def predict_proba(self, con, X):
        cache = self.is_deployed(con)
        items = X if isinstance(X, dict) else self.write_items(con, X)
        sql = self._sql_predict_proba(items, cache)

        return self.read_sql(sql, con)

    def explain(self, con, X, sample_weight):
        cache = self.is_deployed(con)
        items = X if isinstance(X, dict) else self.write_items(con, X) if X else None
        sql = self._sql_explain(items, sample_weight, cache)

        return self.read_sql(sql, con)
    
    def _sql_transform(self, items, concat, name):
        if isinstance(items, str):
            return items
        
        table, item, field = items
        if concat:
            field = f"'{table}:{field}:'" + self.CONCAT_SEP + field
            if self.CONCAT_FUN:
                field = f"{self.CONCAT_FUN}({field})"

        return f"""
            SELECT 
                {item} AS {self.n}, 
                {field} AS {name}, 
                1.0 AS {self.w}
            FROM 
                {table}
            """
        
    def _sql_partial_fit(self, items, sample_weight):
        return f"""
            WITH
                {self._sql_X_nj(items)},
                {self._sql_C_nk(items)},
                {self._sql_X_n()},
                {self._sql_C_n()},
                {self._sql_Z_nj(sample_weight)},
                {self._sql_Y_nk()},
                {self._sql_ZY_njk()}
            SELECT 
                {self.j},
                {self.k},
                {self.SUM}({self.w}) AS {self.w}
            FROM
                ZY_njk
            GROUP BY 
                ZY_njk.{self.j}, 
                ZY_njk.{self.k}
            """

    def _sql_predict(self, items, cache):
        return f"""
            {self._sql_with_HW_jk(cache)}, 
                {self._sql_X_nj(items)},
                {self._sql_U_nk()}, 
                {self._sql_R_nk()}
            SELECT 
                {self.n}, 
                {self.k}
            FROM 
                R_nk
            WHERE 
                {self.w} = 1
            """

    def _sql_predict_proba(self, items, cache):
        return f"""
            {self._sql_with_HW_jk(cache)}, 
                {self._sql_X_nj(items)},
                {self._sql_U_nk()}, 
                {self._sql_U_n()}
            SELECT 
                U_nk.{self.n}, 
                U_nk.{self.k}, 
                U_nk.{self.w} / U_n.{self.w} AS {self.w}
            FROM 
                U_nk, U_n
            WHERE 
                U_nk.{self.n} = U_n.{self.n}
            """

    def _sql_explain(self, items, sample_weight, cache):
        if items is None:
            return f"{self._sql_with_HW_jk(cache)} SELECT * FROM HW_jk"
                
        return f"""
            {self._sql_with_HW_jk(cache)}, 
                {self._sql_X_nj(items)},
                {self._sql_X_n()},
                {self._sql_Z_nj(sample_weight)},
                {self._sql_Z_j()}
            SELECT 
                HW_jk.{self.j},
                HW_jk.{self.k}, 
                HW_jk.{self.w} * {self.POW}(Z_j.{self.w}, ABH.a) AS {self.w} 
            FROM 
                Z_j, HW_jk, ABH
            WHERE 
                Z_j.{self.j} = HW_jk.{self.j}
            """

    def _sql_with_HW_jk(self, cache):
        if cache:
            return f"""
                WITH 
                    {self._sql_ABH()},
                    HW_jk AS (SELECT * FROM {self.table_weights})
                """

        return f"""
            WITH 
                {self._sql_ABH()},
                {self._sql_P_j()}, 
                {self._sql_P_k()}, 
                {self._sql_W_jk()}, 
                {self._sql_W_j()}, 
                {self._sql_H_jk()},
                {self._sql_LN()},
                {self._sql_H_j()}, 
                {self._sql_HW_jk()}
            """

    def _sql_C_nk(self, items):
        return f"""
            C AS (
                {self._sql_transform(items['class'], concat=False, name=self.k)}
            ),
            C_nk AS (
                SELECT * FROM C
                {f"WHERE {self.n} IN ({items['where']})" if items.get('where') else ''}
            )
            """

    def _sql_X_nj(self, items):
        if not isinstance(items, dict):
            return f"X_nj AS (SELECT * FROM {items})"

        return f"""
            X AS (
                {' UNION ALL '.join([
                    self._sql_transform(feature, concat=True, name=self.j) 
                    for feature in items['features']
                ])}
            ),
            X_nj AS (
                SELECT * FROM X
                {f"WHERE {self.n} IN ({items['where']})" if items.get('where') else ''}
            )
            """

    def _sql_X_n(self):
        return f"""
            X_n AS (
                SELECT 
                    {self.n},
                    {self.SUM}({self.w}) AS {self.w}
                FROM 
                    X_nj
                GROUP BY
                    {self.n}
            )
            """
    
    def _sql_C_n(self):
        return f"""
            C_n AS (
                SELECT 
                    {self.n},
                    {self.SUM}({self.w}) AS {self.w}
                FROM 
                    C_nk
                GROUP BY
                    {self.n}
            )
            """

    def _sql_Z_nj(self, sample_weight):
        if isinstance(sample_weight, str):
            if sample_weight == 'norm':
                return "Z_nj AS (SELECT * FROM X_nj)"

            return f"""
                S_n AS (
                    {sample_weight}
                ),
                Z_nj AS (
                    SELECT 
                        X_nj.{self.n},
                        X_nj.{self.j},
                        X_nj.{self.w} / X_n.{self.w} * S_n.{self.w} AS {self.w}
                    FROM
                        X_nj, X_n, S_n
                    WHERE
                        X_nj.{self.n} = X_n.{self.n} AND
                        X_nj.{self.n} = S_n.{self.n}
                )
                """

        return f"""
            Z_nj AS (
                SELECT 
                    X_nj.{self.n},
                    X_nj.{self.j},
                    X_nj.{self.w} / X_n.{self.w} 
                        {f'* {sample_weight}' if sample_weight is not None else ''} 
                        AS {self.w}
                FROM
                    X_nj, X_n
                WHERE
                    X_nj.{self.n} = X_n.{self.n}
            )
            """

    def _sql_Z_j(self):
        return f"""
            Z_j AS (
                SELECT 
                    {self.j},
                    {self.SUM}({self.w}) AS {self.w}
                FROM 
                    Z_nj
                GROUP BY
                    {self.j}
            )
            """
    
    def _sql_Y_nk(self):
        return f"""
            Y_nk AS (
                SELECT 
                    C_nk.{self.n},
                    C_nk.{self.k},
                    C_nk.{self.w} / C_n.{self.w} AS {self.w}
                FROM
                    C_nk, C_n
                WHERE
                    C_nk.{self.n} = C_n.{self.n}
            )
            """

    def _sql_ZY_njk(self):
        return f"""
            ZY_njk AS (
                SELECT
                    Z_nj.{self.n},
                    Z_nj.{self.j},
                    Y_nk.{self.k},
                    Y_nk.{self.w} * Z_nj.{self.w} AS {self.w}
                FROM
                    Z_nj, Y_nk
                WHERE
                    Z_nj.{self.n} = Y_nk.{self.n}
            )
            """

    def _sql_ABH(self):
        return f"""
            ABH AS (
                SELECT 
                    a, b, h
                FROM
                    {self.table_params}
                WHERE
                    {self.field_id} = '{self.id}'
            )
            """

    def _sql_LN(self):
        return f"""
            LN AS (
                SELECT 
                    CASE 
                        WHEN COUNT(*) > 1 
                        THEN {self.LOG}(COUNT(*))
                        ELSE 1
                    END AS {self.w}
                FROM 
                    P_k
            )
            """

    def _sql_P_k(self):
        return f"""
            P_k AS (
                SELECT 
                    {self.k}, 
                    {self.SUM}({self.w}) AS {self.w}
                FROM 
                    {self.table_corpus}
                GROUP BY 
                    {self.k}
            )
            """

    def _sql_P_j(self):
        return f"""
            P_j AS (
                SELECT 
                    {self.j}, 
                    {self.SUM}({self.w}) AS {self.w}
                FROM 
                    {self.table_corpus}
                GROUP BY 
                    {self.j}
            )
            """

    def _sql_W_jk(self):
        return f"""
            W_jk AS (
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
            )
            """

    def _sql_W_j(self):
        return f"""
            W_j AS (
                SELECT 
                    {self.j}, 
                    {self.SUM}({self.w}) AS {self.w}
                FROM 
                    W_jk
                GROUP BY 
                    {self.j}
            )
            """

    def _sql_H_jk(self):
        return f"""
            H_jk AS (
                SELECT 
                    W_jk.{self.j}, 
                    W_jk.{self.k}, 
                    W_jk.{self.w} / W_j.{self.w} AS {self.w}
                FROM 
                    W_jk, W_j
                WHERE 
                    W_jk.{self.j} = W_j.{self.j}
            )
            """

    def _sql_H_j(self):
        return f"""
            H_j AS (
                SELECT 
                    H_jk.{self.j}, 
                    1 + {self.SUM}(
                        H_jk.{self.w} * {self.LOG}(H_jk.{self.w}) / LN.{self.w}
                    ) AS {self.w}
                FROM 
                    H_jk, LN
                GROUP BY 
                    {self.j}
            )
            """

    def _sql_HW_jk(self):
        return f"""
            HW_jk AS (
                SELECT 
                    W_jk.{self.j}, 
                    W_jk.{self.k}, 
                    {self.POW}(W_jk.{self.w}, ABH.a) * {self.POW}(H_j.{self.w}, ABH.h) AS {self.w}
                FROM 
                    W_jk, H_j, ABH
                WHERE 
                    W_jk.{self.j} = H_j.{self.j}
            )
            """

    def _sql_U_nk(self):
        return f"""
            U_nk AS (
                SELECT 
                    X_nj.{self.n}, 
                    HW_jk.{self.k}, 
                    {self.POW}(
                        {self.SUM}(HW_jk.{self.w} * {self.POW}(X_nj.{self.w}, ABH.a)),
                        1 / ABH.a
                    ) AS {self.w} 
                FROM 
                    X_nj, HW_jk, ABH
                WHERE 
                    X_nj.{self.j} = HW_jk.{self.j}
                GROUP BY
                    {self.n}, {self.k}
            )
            """

    def _sql_U_n(self):
        return f"""
            U_n AS (
                SELECT 
                    {self.n}, 
                    {self.SUM}({self.w}) AS {self.w}
                FROM 
                    U_nk
                GROUP BY 
                    {self.n}
            )
            """
    
    def _sql_R_nk(self):
        return f"""
            R_nk AS (
                SELECT 
                    {self.n}, 
                    {self.k}, 
                    ROW_NUMBER() OVER(PARTITION BY {self.n} ORDER BY {self.w} DESC) AS {self.w}
                FROM 
                    U_nk
            )
            """

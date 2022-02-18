import pandas as pd
from uuid import uuid1
from hashlib import md5
from sqlalchemy import text, MetaData, Table, Column, Integer, Float


class Database:

    LOG = 'LOG'
    SUM = 'SUM'
    POW = 'POW'

    def __init__(self, a, b, h, conn, corpus_table, weights_table,
                 items_field, weights_field, classes_field, features_field, features_type):
        # parameters
        self.a, self.b, self.h = a, b, h
        # connection
        self.conn = conn
        # types
        self.features_type = features_type
        # tables
        self.corpus_table, self.weights_table = corpus_table, weights_table
        # fields
        self.i, self.j, self.k, self.w = classes_field, features_field, items_field, weights_field

    def C_i(self):
        return f"""
            SELECT {self.i}, {self.SUM}({self.w}) AS {self.w}
            FROM {self.corpus_table}
            GROUP BY {self.i}
            """

    def C_j(self):
        return f"""
            SELECT {self.j}, {self.SUM}({self.w}) AS {self.w}
            FROM {self.corpus_table}
            GROUP BY {self.j}
            """

    def C_ij(self):
        return f"""
            SELECT {self.corpus_table}.{self.i}, {self.corpus_table}.{self.j}, {self.corpus_table}.{self.w} 
                * {self.POW}(C_i.{self.w}, {-self.b}) 
                * {self.POW}(C_j.{self.w}, {self.b-1}) AS {self.w}
            FROM {self.corpus_table}, C_i, C_j
            WHERE {self.corpus_table}.{self.i}=C_i.{self.i} AND {self.corpus_table}.{self.j}=C_j.{self.j}
            """

    def P_j(self):
        return f"""
            SELECT {self.j}, {self.SUM}({self.w}) AS {self.w}
            FROM C_ij
            GROUP BY {self.j}
            """

    def P_ij(self):
        return f"""
            SELECT C_ij.{self.i}, C_ij.{self.j}, C_ij.{self.w} / P_j.{self.w} AS {self.w}
            FROM C_ij, P_j
            WHERE C_ij.{self.j}=P_j.{self.j}
            """

    def N_i(self):
        return f"""
            SELECT COUNT(*) AS {self.w}
            FROM C_i
            """

    def H_j(self):
        return f"""
            SELECT P_ij.{self.j}, 1 + {self.SUM}(P_ij.{self.w} * {self.LOG}(P_ij.{self.w})) 
                / {self.LOG}(N_i.{self.w}) AS {self.w}
            FROM P_ij, N_i
            GROUP BY {self.j}
            """

    def W_ij(self):
        return f"""
            SELECT C_ij.{self.i}, C_ij.{self.j}, {self.POW}(C_ij.{self.w}, {self.a}) 
                * {self.POW}(H_j.{self.w}, {self.h}) AS {self.w}
            FROM C_ij, H_j
            WHERE C_ij.{self.j}=H_j.{self.j}
            """

    def X_ik(self, items):
        return f"""
            SELECT {items}.{self.k}, W_ij.{self.i}, {self.POW}(
                {self.SUM}({self.POW}({items}.{self.w}, {self.a}) * W_ij.{self.w}), 
                {1. / self.a}) AS {self.w} 
            FROM {items}, W_ij
            WHERE {items}.{self.j}=W_ij.{self.j}
            GROUP BY {items}.{self.k}, W_ij.{self.i}
            """

    def X_k(self):
        return f"""
            SELECT X_ik.{self.k}, {self.SUM}(X_ik.{self.w}) AS {self.w}
            FROM X_ik
            GROUP BY X_ik.{self.k}
            """

    def WITH(self, cache):
        if cache:
            return f"WITH W_ij AS (SELECT * FROM {self.weights_table})"
        return f"""
            WITH C_i AS ({self.C_i()}), C_j AS ({self.C_j()}), C_ij AS ({self.C_ij()}), P_j AS ({self.P_j()}), 
                P_ij AS ({self.P_ij()}), N_i AS ({self.N_i()}), H_j AS ({self.H_j()}), W_ij AS ({self.W_ij()})
            """

    def predict(self, items, cache):
        sql = f"""
            {self.WITH(cache)}, X_ik AS ({self.X_ik(items)}), R_ik AS 
                (
                    SELECT {self.k}, {self.i}, ROW_NUMBER() OVER(PARTITION BY {self.k} ORDER BY {self.w} DESC) AS idx
                    FROM X_ik
                )
            SELECT {self.k}, {self.i}
            FROM R_ik 
            WHERE idx=1
            """
        return pd.read_sql(sql, self.conn)

    def predict_proba(self, items, cache):
        sql = f"""
            {self.WITH(cache)}, X_ik AS ({self.X_ik(items)}), X_k AS ({self.X_k()}) 
            SELECT X_ik.{self.k}, X_ik.{self.i}, X_ik.{self.w} / X_k.{self.w} AS {self.w}
            FROM X_ik, X_k
            WHERE X_ik.{self.k}=X_k.{self.k}
            """
        return pd.read_sql(sql, self.conn)

    def predict_weights(self, items, cache):
        sql = f"""
            {self.WITH(cache)}
            SELECT {items}.{self.k}, W_ij.{self.i}, W_ij.{self.j},
                {self.POW}({items}.{self.w}, {self.a}) * W_ij.{self.w} AS {self.w}
            FROM {items}, W_ij
            WHERE {items}.{self.j}=W_ij.{self.j}
            """
        return pd.read_sql(sql, self.conn)

    def get_weights(self, cache):
        if cache:
            sql = f"SELECT {self.i}, {self.j}, {self.w} FROM {self.weights_table}"
        else:
            sql = f"{self.WITH(cache)} SELECT * FROM W_ij"
        return pd.read_sql(sql, self.conn)

    def cache_weights(self, table):
        sql = f"INSERT INTO {table} {self.WITH(cache=False)} SELECT * FROM W_ij"
        return self.conn.execute(sql)

    def insert(self, table, values):
        keys = values[0].keys()
        cols, pars = ','.join(keys), ', :'.join(keys)
        sql = f"INSERT INTO {table} ({cols}) VALUES(:{pars})"
        self.conn.execute(text(sql), values)

    def insert_or_ignore(self, table, values):
        keys = values[0].keys()
        cols, pars = ','.join(keys), ', :'.join(keys)
        sql = f"INSERT INTO {table} ({cols}) VALUES(:{pars}) ON CONFLICT DO NOTHING"
        self.conn.execute(text(sql), values)

    def insert_or_replace(self, table, values):
        keys = values[0].keys()
        cols, pars = ','.join(keys), ', :'.join(keys)
        sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES(:{pars})"
        self.conn.execute(text(sql), values)

    def insert_or_sum(self, table, values):
        keys = values[0].keys()
        cols, pars = ','.join(keys), ', :'.join(keys)
        sql = f"""
            INSERT INTO {table} ({cols}) VALUES(:{pars}) 
            ON CONFLICT DO UPDATE SET {self.w}={table}.{self.w}+excluded.{self.w}
            """
        self.conn.execute(text(sql), values)

    def read_table(self, table):
        return self.conn.execute(table.select())

    @staticmethod
    def tmp_key():
        return f"tmp_{md5(str(uuid1()).encode()).hexdigest()[:12]}"

    def tmp_table(self):
        return Table(
            self.tmp_key(), MetaData(),
            Column(self.k, Integer, primary_key=True),
            Column(self.j, self.features_type, primary_key=True),
            Column(self.w, Float),
            prefixes=["TEMPORARY"],
        )

import numpy as np
import pandas as pd
from collections import defaultdict
from .sqlite import SQLite
from .postgresql import PostgreSQL

try:
    from sqlalchemy import create_engine, String, Integer
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SQLAlchemy required but not installed. "
        "Please install SQLAlchemy with: pip install sqlalchemy")


class BornClassifierSQL:

    def __init__(self, engine='sqlite:///', prefix='bc', type_features=String, type_labels=Integer):

        if isinstance(engine, str):
            engine = create_engine(engine, echo=False)

        kwargs = {
            'engine': engine,
            'prefix': prefix,
            'type_features': type_features,
            'type_labels': type_labels
        }

        slug = engine.url.get_dialect().name
        if slug == 'sqlite':
            self.db = SQLite(**kwargs)

        elif slug == 'postgresql':
            self.db = PostgreSQL(**kwargs)

        else:
            raise ValueError(
                f"Backend {slug} is not implemented yet. Please open an issue at "
                f"https://github.com/eguidotti/bornrule/issues "
                f"to add support for {slug}."
            )

        if not self.db.deployed:
            with self.db.connect() as con:
                params = [{'a': 0.5, 'b': 1, 'h': 1, 'deployed': False}]
                self.db.write(con, self.db.table_params, values=params, if_exists='insert_or_ignore')

        self.deploy = self.db.deploy
        self.undeploy = self.db.undeploy
        self.check_undeployed = self.db.check_undeployed

    def fit(self, X, y, sample_weight=None):
        self.clean()
        return self.partial_fit(X, y, sample_weight=sample_weight)

    def clean(self):
        self.check_undeployed()

        with self.db.connect() as con:
            self.db.drop(con, table=self.db.table_corpus)

    def partial_fit(self, X, y, sample_weight=None):
        self.check_undeployed()

        if not isinstance(sample_weight, list):
            sample_weight = [1] * len(X)

        corpus = defaultdict(lambda: defaultdict(int))
        for x, y, w in zip(X, y, sample_weight):
            if not isinstance(y, list):
                y = [y]

            n = sum(x.values()) * len(y)
            for c in y:
                for f, v in x.items():
                    corpus[c][f] += w * v / n

        values = []
        for c, d in corpus.items():
            for f, w in d.items():
                values.append({self.db.FIELD_LABEL: c, self.db.FIELD_FEATURE: f, self.db.FIELD_WEIGHT: w})

        with self.db.connect() as con:
            self.db.write_corpus(con, values=values)

        return self

    def predict(self, X):
        with self.db.connect() as con:
            sql = self.db.predict(self.table_items(con, X=X))
            labels = pd.read_sql(sql, con)

        labels = dict(zip(labels[self.db.FIELD_ITEM], labels[self.db.FIELD_LABEL]))
        return np.array([labels[i] if i in labels else None for i in range(len(X))])

    def predict_proba(self, X):
        with self.db.connect() as con:
            sql = self.db.predict_proba(self.table_items(con, X=X))
            proba = pd.read_sql(sql, con)

        proba = self.pivot(proba, index=self.db.FIELD_ITEM, columns=self.db.FIELD_LABEL, values=self.db.FIELD_WEIGHT)
        return proba.reindex(range(len(X))).sparse.to_dense()

    def explain(self, X=None):
        with self.db.connect() as con:
            sql = self.db.explain(None if X is None else self.table_item(con, X))
            weight = pd.read_sql(sql, con)

        return self.pivot(weight, index=self.db.FIELD_FEATURE, columns=self.db.FIELD_LABEL, values=self.db.FIELD_WEIGHT)

    def set_params(self, a, b, h):
        self.check_undeployed()

        with self.db.connect() as con:
            self.db.write_params(con, a=a, b=b, h=h)

    def get_params(self):
        with self.db.connect() as con:
            return self.db.read_params(con).to_dict(orient='records')[0]

    def table_items(self, con, X):
        values = []
        for i, bow in enumerate(X):
            for f, w in bow.items():
                values.append({self.db.FIELD_ITEM: i, self.db.FIELD_FEATURE: f, self.db.FIELD_WEIGHT: w})

        table = self.db.tmp_items()
        self.db.write(con, table=table, values=values)

        return table

    def table_item(self, con, X):
        if not isinstance(X, dict):
            raise ValueError(
                "X must be a single item {key: value, ...} and NOT a list of items [{key: value, ...}, ...]"
            )

        values = []
        for f, w in X.items():
            values.append({self.db.FIELD_FEATURE: f, self.db.FIELD_WEIGHT: w})

        table = self.db.tmp_item()
        self.db.write(con, table=table, values=values)

        return table

    @staticmethod
    def pivot(df, index, columns, values):
        df[values] = df[values].astype(pd.SparseDtype(float))
        df = df.pivot(index=index, columns=columns, values=values)
        df = df.astype(pd.SparseDtype(float, fill_value=0))
        df.rename_axis(None, axis=0, inplace=True)
        df.rename_axis(None, axis=1, inplace=True)
        return df

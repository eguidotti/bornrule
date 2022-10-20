import numpy as np
import pandas as pd
from .sqlite import SQLite
from .postgresql import PostgreSQL

try:
    from sqlalchemy import create_engine, String, Integer
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SQLAlchemy required but not installed. "
        "Please install SQLAlchemy with: pip install sqlalchemy")


class BornClassifierSQL:

    def __init__(self, engine='sqlite:///', prefix='bc', type_features=String, type_classes=Integer):

        if isinstance(engine, str):
            engine = create_engine(engine, echo=False)

        kwargs = {
            'engine': engine,
            'prefix': prefix,
            'type_features': type_features,
            'type_classes': type_classes
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

    def set_params(self, a, b, h):
        self.db.check_editable()

        if a <= 0:
            raise ValueError("The parameter 'a' must be strictly positive.")

        if b < 0:
            raise ValueError("The parameter 'b' must be non-negative.")

        if h < 0:
            raise ValueError("The parameter 'h' must be non-negative.")

        with self.db.connect() as con:
            self.db.write_params(con, a=a, b=b, h=h)

    def get_params(self):
        with self.db.connect() as con:
            return self.db.read_params(con).to_dict(orient='records')[0]

    def fit(self, X, y, sample_weight=None):
        self.db.check_editable()

        with self.db.connect() as con:
            self.db.table_corpus.drop(con, checkfirst=True)

        return self.partial_fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y, sample_weight=None):
        self.db.check_editable()

        if sample_weight is None:
            sample_weight = [1] * len(X)

        with self.db.connect() as con:
            self.db.write_corpus(con, X=X, y=y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        self.db.check_fitted()

        with self.db.connect() as con:
            classes = self.db.predict(con, X=X)

        classes = dict(zip(classes[self.db.FIELD_ITEM], classes[self.db.FIELD_CLASS]))
        classes = np.array([classes[i] if i in classes else None for i in range(len(X))])

        return classes

    def predict_proba(self, X):
        self.db.check_fitted()

        with self.db.connect() as con:
            proba = self.db.predict_proba(con, X=X)

        proba = self._pivot(proba, index=self.db.FIELD_ITEM, columns=self.db.FIELD_CLASS, values=self.db.FIELD_WEIGHT)
        proba = proba.reindex(range(len(X))).sparse.to_dense()

        return proba

    def explain(self, X=None, sample_weight=None):
        self.db.check_fitted()

        if X is not None:
            n = len(X)

            if sample_weight is None:
                if n > 1:
                    sample_weight = [1. / n] * n

            elif len(sample_weight) != n:
                raise ValueError(
                    "Dimension mismatch. X and sample_weight must have the same length."
                )

        with self.db.connect() as con:
            W = self.db.explain(con, X=X, sample_weight=sample_weight)

        return self._pivot(W, index=self.db.FIELD_FEATURE, columns=self.db.FIELD_CLASS, values=self.db.FIELD_WEIGHT)

    def deploy(self):
        with self.db.connect() as con:
            with con.begin():
                self.db.deploy(con)

    def undeploy(self):
        with self.db.connect() as con:
            with con.begin():
                self.db.undeploy(con)

    @staticmethod
    def _pivot(df, index, columns, values):
        df[values] = df[values].astype(pd.SparseDtype(float))
        df = df.pivot(index=index, columns=columns, values=values)
        df = df.astype(pd.SparseDtype(float, fill_value=0))
        df.rename_axis(None, axis=0, inplace=True)
        df.rename_axis(None, axis=1, inplace=True)
        return df

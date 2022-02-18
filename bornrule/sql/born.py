try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.exc import OperationalError
    from sqlalchemy import MetaData, Table, Column, String, Integer, Float
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SQLAlchemy required but not installed. "
        "Please install SQLAlchemy with: pip install sqlalchemy")

import json
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from scipy.sparse import coo_matrix, csr_matrix
from .database import Database
from .sqlite import SQLite
from .postgresql import PostgreSQL
from ..born import BornClassifier


class BornClassifierSQL(BornClassifier):

    # metadata fields
    param_field = "param"
    value_field = "value"

    # data fields
    features_field = "features"
    classes_field = "classes"
    weights_field = "weights"
    items_field = "items"

    def __init__(self, a=0.5, b=1., h=1.,
                 engine: Engine = None, prefix='born', features_type=String, classes_type=Integer):

        # init parent
        super().__init__(a=a, b=b, h=h)

        # engine
        if isinstance(engine, str):
            self.engine = create_engine(engine, echo=False)
        else:
            self.engine = engine

        # table names
        self.prefix = prefix
        self.params_table = f"{self.prefix}_params"
        self.corpus_table = f"{self.prefix}_corpus"
        self.weights_table = f"{self.prefix}_weights"

        # data types
        self.features_type = features_type
        self.classes_type = classes_type

        # vocabulary
        self.features_ = {}

        # deployed
        self.deployed_ = False

        # sync params
        if self.engine is not None:
            try:
                params = self.read_params_()
                for key, val in params.items():
                    setattr(self, key, val)
            except OperationalError:
                self.write_params_(if_exists="fail")

    """
    Public methods
    """

    def get_params(self, deep=True):
        # keep only the model hyper-parameters
        p = super().get_params(deep=deep)
        return {k: p[k] for k in ['a', 'b', 'h']}

    def set_params(self, **params):
        # do not overwrite if deployed
        if self.deployed_:
            raise ValueError('Cannot change parameters of a deployed instance. Undeploy it first.')
        # write to db
        if self.engine is not None:
            self.write_params_(**params)
        # set params
        super().set_params(**params)

    def fit(self, X, y, sample_weight=None):
        # do not overwrite if deployed
        if self.deployed_:
            raise ValueError('Cannot overwrite a deployed instance. Undeploy it first or use partial_fit().')
        # drop corpus from db
        if self.engine is not None:
            with self.engine.connect() as conn:
                self.corpus_table_().drop(conn, checkfirst=True)
        # reset vocabulary and fit
        self.features_ = {}
        return super().fit(X=X, y=y, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # update vocabulary
        [self.features_.setdefault(k, len(self.features_)) for k in {k for b in X for k in b.keys()}]
        # resize corpus
        if self.corpus_ is not None:
            self.corpus_.resize(len(self.features_), len(self.classes_))
        # fit in memory
        super().partial_fit(X=self.to_scipy_(X), y=y, classes=classes, sample_weight=sample_weight)
        # if engine
        if self.engine is not None:
            # write to db
            corpus = coo_matrix(self.corpus_)
            values = [{self.features_field: f, self.classes_field: c, self.weights_field: w}
                      for f, c, w in zip(self.i2f_(corpus.row), self.i2c_(corpus.col), corpus.data)]
            with self.engine.begin() as conn:
                db, table = self.db_(conn), self.corpus_table_()
                self.write_table_(db=db, table=table, values=values, if_exists='insert_or_sum')
            # reset corpus in memory
            self.corpus_ = None
        # return self
        return self

    def predict(self, X):
        # in-memory or sql
        return self.predict_py_(X) if self.engine is None else self.predict_sql_(X)

    def predict_proba(self, X):
        # in-memory or sql
        return self.predict_proba_py_(X) if self.engine is None else self.predict_proba_sql_(X)

    def explain(self, X=None):
        # in-memory or sql
        return self.explain_py_(X) if self.engine is None else self.explain_sql_(X)

    def deploy(self):
        # check db
        if self.engine is None:
            raise ValueError('No SQL engine found. Provide a valid engine upon initialization to (un)deploy a model.')
        # cache weights
        with self.engine.begin() as conn:
            db, corpus_table, weights_table = self.db_(conn), self.corpus_table_(), self.weights_table_()
            if not self.engine.dialect.has_table(db.conn, corpus_table.name):
                raise NotFittedError(
                    f"This {self.__class__.__name__} instance is not fitted yet. "
                    "Call 'fit' with appropriate arguments before deploying this model.")
            self.write_table_(db=db, table=weights_table, if_exists='replace')
            db.cache_weights(weights_table)
        # deploy
        self.write_params_(deployed_=True)

    def undeploy(self):
        # check db
        if self.engine is None:
            raise ValueError('No SQL engine found. Provide a valid engine upon initialization to (un)deploy a model.')
        # drop cached weights
        with self.engine.begin() as conn:
            self.weights_table_().drop(conn)
        # undeploy
        self.write_params_(deployed_=False)

    """
    In-memory and SQL methods
    """

    def predict_py_(self, X):
        return self.predict_proba(X).idxmax(axis=1).values

    def predict_sql_(self, X):
        with self.engine.connect() as conn:
            db = self.db_(conn)
            c = db.predict(items=self.tmp_items_(X=X, db=db), cache=self.deployed_)
        imap = dict(zip(c[self.items_field], c[self.classes_field]))
        return np.array([imap[i] if i in imap else None for i in range(len(X))])

    def predict_proba_py_(self, X):
        p = super().predict_proba(X=self.to_scipy_(X))
        return pd.DataFrame(p, columns=self.classes_)

    def predict_proba_sql_(self, X):
        with self.engine.connect() as conn:
            db = self.db_(conn)
            p = db.predict_proba(items=self.tmp_items_(X=X, db=db), cache=self.deployed_)
        p = self.pivot_(p, index=self.items_field, columns=self.classes_field, values=self.weights_field)
        return p.reindex(range(len(X)), fill_value=0).sparse.to_dense()

    def explain_py_(self, X=None):
        w = super().explain(self.to_scipy_(X) if X is not None else None)
        return pd.DataFrame.sparse.from_spmatrix(w, index=self.i2f_(range(len(self.features_))), columns=self.classes_)

    def explain_sql_(self, X=None):
        with self.engine.connect() as conn:
            db = self.db_(conn)
            if X is None:
                w = db.get_weights(cache=self.deployed_)
            else:
                if not isinstance(X, dict):
                    raise ValueError("X must be a single item {key: value, ...}")
                w = db.predict_weights(items=self.tmp_items_(X=[X], db=db), cache=self.deployed_)
        return self.pivot_(w, index=self.features_field, columns=self.classes_field, values=self.weights_field)

    """
    Read & Write params to the database
    """

    def read_params_(self):
        with self.engine.connect() as conn:
            params = self.db_(conn).read_table(self.params_table_())
            return {k: json.loads(v) for k, v in params}

    def write_params_(self, if_exists="insert_or_replace", **params):
        if not params:
            params = self.get_params()
        with self.engine.begin() as conn:
            vals = [{self.param_field: k, self.value_field: json.dumps(p)} for k, p, in params.items()]
            self.write_table_(db=self.db_(conn), table=self.params_table_(), values=vals, if_exists=if_exists)
        for key, val in params.items():
            setattr(self, key, val)

    """
    DataBase schema
    """

    def params_table_(self):
        return Table(
            self.params_table, MetaData(),
            Column(self.param_field, String, primary_key=True),
            Column(self.value_field, String),
        )

    def corpus_table_(self):
        return Table(
            self.corpus_table, MetaData(),
            Column(self.classes_field, self.classes_type, primary_key=True),
            Column(self.features_field, self.features_type, primary_key=True),
            Column(self.weights_field, Float),
        )

    def weights_table_(self):
        return Table(
            self.weights_table, MetaData(),
            Column(self.classes_field, self.classes_type, primary_key=True),
            Column(self.features_field, self.features_type, primary_key=True),
            Column(self.weights_field, Float),
        )

    """
    Database utilities
    """

    def db_(self, conn):
        kwargs = {
            'a': self.a,
            'b': self.b,
            'h': self.h,
            'conn': conn,
            'corpus_table': self.corpus_table_(),
            'weights_table': self.weights_table_(),
            'items_field': self.items_field,
            'weights_field': self.weights_field,
            'classes_field': self.classes_field,
            'features_field': self.features_field,
            'features_type': self.features_type
        }
        slug = self.engine.url.get_dialect().name
        if slug == 'sqlite':
            return SQLite(**kwargs)
        if slug == 'postgresql':
            return PostgreSQL(**kwargs)
        return Database(**kwargs)

    def tmp_items_(self, X, db):
        values = [{self.items_field: i, self.features_field: k, self.weights_field: w}
                  for i, b in enumerate(X) for k, w in b.items()]
        tmp_table = db.tmp_table()
        self.write_table_(db=db, table=tmp_table, values=values)
        return tmp_table

    def write_table_(self, db, table, values=None, if_exists='fail'):
        assert if_exists in ['fail', 'replace', 'insert', 'insert_or_ignore', 'insert_or_replace', 'insert_or_sum']
        if not self.engine.dialect.has_table(db.conn, table.name):
            table.create(db.conn)
        else:
            if if_exists == 'fail':
                raise ValueError(f"Table {table} already exists.")
            if if_exists == 'replace':
                table.drop(db.conn)
                table.create(db.conn)
        if values is not None:
            insert = getattr(db, "insert" if if_exists in ['fail', 'replace'] else if_exists)
            insert(table, values)

    """
    Transformers
    """

    @staticmethod
    def pivot_(df, index, columns, values):
        df[values] = df[values].astype(pd.SparseDtype(float))
        df = df.pivot(index=index, columns=columns, values=values)
        df = df.astype(pd.SparseDtype(float, fill_value=0))
        df.rename_axis(None, axis=0, inplace=True)
        df.rename_axis(None, axis=1, inplace=True)
        return df

    def to_scipy_(self, bow):
        val, row, col = [], [], []
        for i, b in enumerate(bow):
            for j, v in b.items():
                if j in self.features_:
                    val.append(v), row.append(i), col.append(self.features_[j])
        return csr_matrix((val, (row, col)), shape=(len(bow), len(self.features_)))

    def i2c_(self, idx):
        imap = dict(zip(range(len(self.classes_)), self.classes_))
        return [imap[i] for i in idx]

    def c2i_(self, classes):
        cmap = dict(zip(self.classes_, range(len(self.classes_))))
        return [cmap[c] for c in classes]

    def i2f_(self, idx):
        imap = dict(zip(self.features_.values(), self.features_.keys()))
        return [imap[i] for i in idx]

    def f2i_(self, features):
        fmap = self.features_
        return [fmap[f] for f in features]

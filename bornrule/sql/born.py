import pandas as pd
from collections import defaultdict

try:
    from sqlalchemy import create_engine, String, Integer

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SQLAlchemy required but not installed. "
        "Please install SQLAlchemy with e.g. -> pip install sqlalchemy")

from .schema import Schema, Or
from .database import Query
from .sqlite import SQLite
from .postgresql import PostgreSQL


class BornClassifierSQL:
    """SQL implementation of Born's Classifier

    This class is compatible with SQLite and PostgreSQL. Data items are to be passed as list of 
    dictionaries in the format `[{feature: value, ...}, ...]` or directly as SQL queries.

    Parameters
    ----------
    id : str
        The model id.
    engine : Engine or str
        [SQLAlchemy engine or connection string](https://docs.sqlalchemy.org/en/14/core/engines.html)
        to connect to the database.
    configs: dict
        Database configurations structured as follows.
        {
            'class': (table, item, field) or 'SELECT item, class, weight',
            'features': [
                (table, item, field) or 'SELECT item, feature, weight FROM ...',
                (table, item, field) or 'SELECT item, feature, weight FROM ...',
                ...
            ]
        }
    type_feature : TraversibleType
        [SQLAlchemy type](https://docs.sqlalchemy.org/en/14/core/type_basics.html#generic-camelcase-types)
        of features.
    type_class : TraversibleType
        [SQLAlchemy type](https://docs.sqlalchemy.org/en/14/core/type_basics.html#generic-camelcase-types)
        of classes.
    field_id : str
        Label to use for the model ids.
    field_item : str
        Label to use for data items.
    field_feature : str
        Label to use for features.
    field_class : str
        Label to use for classes.
    field_weight : str
        Label to use for weights.
    table_corpus : str
         Name of the table containing the corpus.
    table_params : str
        Name of the table containing the model's hyper-parameters.
    table_weights : str
        Name of the table containing the model's weigths.

    Attributes
    ----------
    db : Database
        [Database class](https://github.com/eguidotti/bornrule/blob/main/bornrule/sql/database.py) acting as
        interpreter between python and the database.

    """

    def __init__(self,
                 id='model',
                 engine='sqlite:///',
                 configs=None,
                 type_feature=String,
                 type_class=Integer,
                 field_id="id",
                 field_item="item",
                 field_feature="feature",
                 field_class="class",
                 field_weight="weight",
                 table_corpus="corpus",
                 table_params="params",
                 table_weights="weights"):

        self.configs = configs
        if configs is not None:
            Schema({'class': Or(tuple, str), 'features': [Or(tuple, str)]}).validate(configs)

        if isinstance(engine, str):
            engine = create_engine(engine)

        kwargs = {
            'id': id,
            'engine': engine,
            'type_feature': type_feature,
            'type_class': type_class,
            'field_id': field_id,
            'field_item': field_item,
            'field_feature': field_feature,
            'field_class': field_class,
            'field_weight': field_weight,
            'table_params': table_params,
            'table_corpus': table_corpus,
            'table_weights': table_weights,
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

        self.params = None

    def get_params(self):
        """Get parameters

        Returns
        -------
        params : dict
            Model's hyper-parameters `a`, `b`, `h`.

        """
        if self.params is None:
            with self.db.connect() as con:
                self.params = self.db.read_params(con)

        return self.params.copy()

    def set_params(self, **params):
        """Set parameters

        Parameters
        ----------
        **params
             Model's hyper-parameters: `a` (>0), `b` (>=0), and `h` (>=0).

        """
        p = self.get_params()
        p.update(params)

        if p['a'] <= 0:
            raise ValueError(
                "The parameter 'a' must be strictly positive."
            )

        if p['b'] < 0:
            raise ValueError(
                "The parameter 'b' must be non-negative."
            )

        if p['h'] < 0:
            raise ValueError(
                "The parameter 'h' must be non-negative."
            )

        with self.db.connect() as con:
            with con.begin():
                self.db.check_editable(con)
                self.db.write_params(con, **p)
                self.params = p

    def fit(self, X, y=None, sample_weight=None):
        """Fit the classifier according to the training data X, y

        Parameters
        ----------
        X : list of dict of length n_samples, or str
            Training data in the format `[{feature: value, ...}, ...]`, 
            or an SQL query in the format `SELECT item FROM ...` giving the ids of the items to use.
        y : list-like of length n_samples
            List giving the target class for each sample. If a list of dict in the format `[{class: value, ...}, ...]`,
            then each dict gives the distribution of the classes for each sample (e.g., multi-labeled samples).
            When `X` is an SQL query, `y` must be `None` and the classes are automatically retrieved from `configs`.
        sample_weight : list-like of length n_samples, or str
            List of weights that are assigned to individual samples, or an SQL query in the 
            format `SELECT item, weight FROM ...` giving the weight for each item.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self._validate(X=X, y=y, sample_weight=sample_weight)

        with self.db.connect() as con:
            with con.begin():
                self.db.check_editable(con)
                self.db.table_corpus.drop(con, checkfirst=True)

        return self.partial_fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """Incremental fit on a batch of samples

        This method is expected to be called several times consecutively on different chunks of a dataset so
        as to implement out-of-core or online learning.

        Parameters
        ----------
        X : list of dict of length n_samples, or str
            Training data in the format `[{feature: value, ...}, ...]`, 
            or an SQL query in the format `SELECT item FROM ...` giving the ids of the items to use.
        y : list-like of length n_samples
            List giving the target class for each sample. If a list of dict in the format `[{class: value, ...}, ...]`,
            then each dict gives the distribution of the classes for each sample (e.g., multi-labeled samples).
            When `X` is an SQL query, `y` must be `None` and the classes are automatically retrieved from `configs`.
        sample_weight : list-like of length n_samples, or str
            List of weights that are assigned to individual samples, or an SQL query in the 
            format `SELECT item, weight FROM ...` giving the weight for each item.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self._validate(X=X, y=y, sample_weight=sample_weight)
        X, sample_weight = self._transform(X=X, sample_weight=sample_weight)

        with self.db.connect() as con:
            with con.begin():
                self.db.partial_fit(con, X=X, y=y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Perform classification on the test data X

        Parameters
        ----------
        X : list of dict of length n_samples, or str
            Test data in the format `[{feature: value, ...}, ...]`, 
            or an SQL query in the format `SELECT item FROM ...` giving the ids of the items to use.

        Returns
        -------
        y : Series of shape (n_samples, )
            Predicted target classes for `X`.

        """
        self._validate(X=X)
        X = self._transform(X=X)

        with self.db.connect() as con:
            self.db.check_fitted(con)
            classes = self.db.predict(con, X=X)

        return self._pivot(classes, index=self.db.n, values=self.db.k, X=X)

    def predict_proba(self, X):
        """Return probability estimates for the test data X

        Parameters
        ----------
        X : list of dict of length n_samples
            Test data in the format `[{feature: value, ...}, ...]`,
            or an SQL query in the format `SELECT item FROM ...` giving the ids of the items to use.

        Returns
        -------
        y : DataFrame of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in the model.

        """
        self._validate(X=X)
        X = self._transform(X=X)

        with self.db.connect() as con:
            self.db.check_fitted(con)
            proba = self.db.predict_proba(con, X=X)

        return self._pivot(proba, index=self.db.n, columns=self.db.k, values=self.db.w, X=X)

    def explain(self, X=None, sample_weight=None):
        r"""Compute global and local explanation

        Parameters
        ----------
        X : list of dict of length n_samples
            Test data in the format `[{feature: value, ...}, ...]`,
            or an SQL query in the format `SELECT item FROM ...` giving the ids of the items to use.
            If not provided, then global weights are returned.
        sample_weight : list-like of length n_samples, or str
            List of weights that are assigned to individual samples, or an SQL query in the 
            format `SELECT item, weight FROM ...` giving the weight for each item.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        df : DataFrame of shape (n_features, n_classes)
            Returns the feature importance for each class in the model.

        """
        if X is not None:
            self._validate(X=X, sample_weight=sample_weight)
            X, sample_weight = self._transform(X=X, sample_weight=sample_weight)
            
            if isinstance(X, list):
                z = defaultdict(int)
                for x, w in zip(X, sample_weight):
                    n = sum(x.values())
                    if n != 0:
                        for f, v in x.items():
                            z[f] += w * v / n
                X = z
                sample_weight = 1
                
        with self.db.connect() as con:
            self.db.check_fitted(con)
            df = self.db.explain(con, X=X, sample_weight=sample_weight)

        return self._pivot(df, index=self.db.j, columns=self.db.k, values=self.db.w)

    def deploy(self, deep=False, overwrite=False):
        """Deploy the instance

        Generate and store the weights that are used for prediction to speed up inference time.

        Parameters
        ----------
        deep : bool
            Whether the corpus is dropped.
        overwrite : bool
            Whether to overwrite the weights if the instance is already deployed.

        """
        with self.db.connect() as con:
            with con.begin():
                if overwrite and self.is_deployed():
                    self.db.undeploy(con, deep=False)

                self.db.deploy(con, deep=deep)

    def undeploy(self, deep=False):
        """Undeploy the instance

        Drop the weights that are used for prediction. 
        
        Parameters
        ----------
        deep : bool
            Whether the corpus and parameters are also dropped.
            If `True`, the model is fully removed from the database.

        """
        with self.db.connect() as con:
            with con.begin():
                self.db.undeploy(con, deep=deep)

        if deep:
            self.params = None

    def is_fitted(self):
        """Is fitted?

        Checks whether the instance is fitted.

        Returns
        -------
        is : bool
            Returns `True` if the instance is fitted, `False` otherwise.

        """
        with self.db.connect() as con:
            return self.db.is_fitted(con)

    def is_deployed(self):
        """Is deployed?

        Checks whether the instance is deployed.

        Returns
        -------
        is : bool
            Returns `True` if the instance is deployed, `False` otherwise.

        """
        with self.db.connect() as con:
            return self.db.is_deployed(con)

    def _transform(self, X, sample_weight="no_transform"):
        """Transform input"""

        if sample_weight is None:
            sample_weight = 1
        if isinstance(X, str):
            X = Query(x=self.configs['features'], y=self.configs['class'], n=X)
        if isinstance(X, list) and isinstance(sample_weight, (int, float)):
            sample_weight = [sample_weight] * len(X)

        return X if sample_weight == "no_transform" else (X, sample_weight)

    @staticmethod
    def _validate(X, y="no_validation", sample_weight=None):
        """Validate input"""

        only_X = isinstance(y, str) and y == "no_validation"
        
        if isinstance(X, str):

            if not only_X and y is not None:
                raise ValueError(
                    "y must be None when X is a query string"
                )
            
            if sample_weight is not None and not isinstance(sample_weight, (str, int, float)):
                raise ValueError(
                    "sample_weight must be a query string or a number when X is a query string"
                )
                        
        else:

            if not isinstance(X, list):
                raise ValueError(
                    "X must be a list of dict or a query string"
                )
                            
            if not only_X and len(X) != len(y):
                raise ValueError(
                    "Dimension mismatch. X and y must have the same length"
                )

            if sample_weight is not None and not isinstance(sample_weight, (int, float)):
                if len(X) != len(sample_weight):
                    raise ValueError(
                        "Dimension mismatch. X and sample_weight must have the same length"
                    )

    @staticmethod
    def _pivot(df, index, values, columns=None, X=None):
        """Pivot table and clear axis"""

        if columns is not None:
            df[values] = df[values].astype(pd.SparseDtype(float))
            df = df.pivot(index=index, columns=columns, values=values)
            df = df.astype(pd.SparseDtype(float, fill_value=0))
            df.rename_axis(None, axis=0, inplace=True)
            df.rename_axis(None, axis=1, inplace=True)

        if columns is None:
            df = pd.Series(data=df[values].values, index=df[index].values)

        if X is not None:
            df = df.reindex(range(len(X)) if isinstance(X, list) else None)

        return df

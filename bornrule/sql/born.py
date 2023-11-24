import pandas as pd
from collections import defaultdict
from schema import Schema, Or

try:
    from sqlalchemy import create_engine, String, Integer

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SQLAlchemy required but not installed. "
        "Please install SQLAlchemy with e.g. -> pip install sqlalchemy")

from .sqlite import SQLite
from .postgresql import PostgreSQL


class BornClassifierSQL:
    """SQL implementation of Born's Classifier

    This class is compatible with SQLite and PostgreSQL.
    Data items are to be passed as list of dictionaries in the format `[{feature: value, ...}, ...]`.
    This classifier is suitable for classification with non-negative feature values.
    The values are treated as unnormalized probability distributions.
    If provided, configurations must be structured as follows:
        {
            'class': (table, item, field) OR 'SELECT item, class, weight',
            'features': [
                (table, item, field) OR 'SELECT item, feature, weight',
                (table, item, field) OR 'SELECT item, feature, weight',
                ...
            ]
        }

    Parameters
    ----------
    id : str
        The model id.
    engine : Engine or str
        [SQLAlchemy engine or connection string](https://docs.sqlalchemy.org/en/14/core/engines.html)
        to connect to the database.
    configs: dict
        Database configurations.
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

    def fit(self, X, y, sample_weight=None):
        """Fit the classifier according to the training data X, y

        Parameters
        ----------
        X : list of dict of length n_samples
            Training data in the format `[{feature: value, ...}, ...]`.
        y : list-like of length n_samples
            List giving the target class for each sample. If a list of dict in the format `[{class: value, ...}, ...]`,
            then each dict gives the distribution of the classes for each sample (e.g., multi-labeled samples)
        sample_weight : list-like of length n_samples
            List of weights that are assigned to individual samples.
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

    def partial_fit(self, X, y, sample_weight=None):
        """Incremental fit on a batch of samples

        This method is expected to be called several times consecutively on different chunks of a dataset so
        as to implement out-of-core or online learning.

        Parameters
        ----------
        X : list of dict of length n_samples
            Training data in the format `[{feature: value, ...}, ...]`.
        y : list-like of length n_samples
            List giving the target class for each sample. If a list of dict in the format `[{class: value, ...}, ...]`,
            then each dict gives the distribution of the classes for each sample (e.g., multi-labeled samples)
        sample_weight : list-like of length n_samples
            List of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        X, y = self._validate(X=X, y=y, sample_weight=sample_weight)

        with self.db.connect() as con:
            with con.begin():
                self.db.partial_fit(con, X=X, y=y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Perform classification on the test data X

        Parameters
        ----------
        X : list of dict of length n_samples
            Test data in the format `[{feature: value, ...}, ...]`.

        Returns
        -------
        y : Series of shape (n_samples, )
            Predicted target classes for `X`.

        """
        X = self._validate(X=X)

        with self.db.connect() as con:
            self.db.check_fitted(con)
            classes = self.db.predict(con, X=X)

        return self._pivot(classes, index=self.db.n, values=self.db.k, X=X)

    def predict_proba(self, X):
        """Return probability estimates for the test data X

        Parameters
        ----------
        X : list of dict of length n_samples
            Test data in the format `[{feature: value, ...}, ...]`.

        Returns
        -------
        y : DataFrame of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in the model.

        """
        X = self._validate(X=X)

        with self.db.connect() as con:
            self.db.check_fitted(con)
            proba = self.db.predict_proba(con, X=X)

        return self._pivot(proba, index=self.db.n, columns=self.db.k, values=self.db.w, X=X)

    def explain(self, X=None, sample_weight=None):
        r"""Global and local explanation

        For each test vector $`x`$, the $`a`$-th power of the unnormalized probability for the $`k`$-th class is
        given by the matrix product:

        ```math
        u_k^a = \sum_j W_{jk}x_j^a
        ```
        where $`W`$ is a matrix of non-negative weights that generally depends on the model's
        hyper-parameters ($`a`$, $`b`$, $`h`$). The classification probabilities are obtained by
        normalizing $`u`$ such that it sums up to $`1`$.

        This method returns global or local feature importance weights, depending on `X`:

        - When `X` is not provided, this method returns the global weights $`W`$.

        - When `X` is a single sample,
        this method returns a matrix of entries $`(j,k)`$ where each entry is given by $`W_{jk}x_j^a`$.

        - When `X` contains multiple samples,
        then the values above are computed for each sample and this method returns their weighted sum.
        By default, each sample is given unit weight.

        Parameters
        ----------
        X : list of dict of length n_samples
            Test data in the format `[{feature: value, ...}, ...]`.
            If not provided, then global weights are returned.
        sample_weight : list-like of length n_samples
            List of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        E : DataFrame of shape (n_features, n_classes)
            Returns the feature importance for each class in the model.

        """
        if X is not None:
            X = self._validate(X=X, sample_weight=sample_weight)
            if isinstance(X, list):
                Z = defaultdict(int)
     
                if sample_weight is None:
                    sample_weight = [1] * len(X)

                for x, w in zip(X, sample_weight):
                    n = sum(x.values())
                    if n != 0:
                        for f, v in x.items():
                            Z[f] += w * v / n
                
                X, sample_weight = [Z], 'norm'

        with self.db.connect() as con:
            self.db.check_fitted(con)
            W = self.db.explain(con, X=X, sample_weight=sample_weight)

        return self._pivot(W, index=self.db.j, columns=self.db.k, values=self.db.w)

    def deploy(self, deep=False):
        """Deploy the instance

        Generate and store the weights that are used for prediction to speed up inference time.
        A deployed instance cannot be modified. To update a deployed instance, undeploy it first.

        Parameters
        ----------
        deep : bool
            Whether the corpus is dropped.

        """
        with self.db.connect() as con:
            with con.begin():
                self.db.deploy(con, deep=deep)

    def undeploy(self, deep=False):
        """Undeploy the instance

        Drop the weights that are used for prediction. Weights will be recomputed each time on-the-fly.
        Useful for development, testing, and incremental fit.

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

    def redeploy(self, deep=False):
        """Redeploy the instance

        Undeploy and deploy the instance again. Useful to update the weights in a single transaction.

        Parameters
        ----------
        deep : bool
            Whether the corpus is dropped.

        """
        with self.db.connect() as con:
            with con.begin():
                self.db.undeploy(con, deep=False)
                self.db.deploy(con, deep=deep)

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

    def _validate(self, X, y="no_validation", sample_weight=None):
        """Input validation"""

        only_X = isinstance(y, str) and y == "no_validation"

        if (isinstance(X, str) or X is None) and (only_X or y is None):            
            X = dict(self.configs, where=X)
            return X if only_X else (X, y)

        if not isinstance(X, list):
            raise ValueError(
                "X must be a list of dict in the form [{feature: value, ...}, ...]"
            )

        for i, x in enumerate(X):
            if not isinstance(x, dict):
                raise ValueError(
                    f"Element {i} of X is not a dict"
                )

            for _, value in x.items():
                if value < 0:
                    raise ValueError(
                        f"Element {i} of X contains negative values"
                    )

        if sample_weight is not None:
            if len(X) != len(sample_weight):
                raise ValueError(
                    "Dimension mismatch. X and sample_weight must have the same length"
                )

        if not only_X:
            if len(X) != len(y):
                raise ValueError(
                    "Dimension mismatch. X and y must have the same length"
                )
            
        return X if only_X else (X, y)

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

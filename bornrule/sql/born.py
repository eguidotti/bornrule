import pandas as pd

try:
    from sqlalchemy import create_engine, String, Integer

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SQLAlchemy required but not installed. "
        "Please install SQLAlchemy with: pip install sqlalchemy")

from .sqlite import SQLite
from .postgresql import PostgreSQL


class BornClassifierSQL:
    """SQL implementation of Born's Classifier

    This class is compatible with SQLite and PostgreSQL.
    Data items are to be passed as list of dictionaries in the format `[{feature: value, ...}, ...]`.
    This classifier is suitable for classification with non-negative feature values.
    The values are treated as unnormalized probability distributions.

    Parameters
    ----------
    engine : Engine or str
        [SQLAlchemy engine or connection string](https://docs.sqlalchemy.org/en/14/core/engines.html)
        to connect to the database.
    prefix : str
        The prefix to use for the tables in the database.
        Instances created with different `prefix` are independent from each other.
    type_features : TraversibleType
        [SQLAlchemy type](https://docs.sqlalchemy.org/en/14/core/type_basics.html#generic-camelcase-types)
        of the features.
    type_classes : TraversibleType
        [SQLAlchemy type](https://docs.sqlalchemy.org/en/14/core/type_basics.html#generic-camelcase-types)
        of the classes.
    field_features : str
        Label to use for features.
    field_classes : str
       Label to use for classes.
    field_items : str
        Label to use for items.
    field_weights : str
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

    def __init__(self, engine='sqlite:///', prefix='bc', type_features=String, type_classes=Integer,
                 field_features="feature", field_classes="class", field_items="item", field_weights="weight",
                 table_corpus="corpus", table_params="params", table_weights="weights"):

        if isinstance(engine, str):
            engine = create_engine(engine, echo=False)

        kwargs = {
            'engine': engine,
            'prefix': prefix,
            'type_features': type_features,
            'type_classes': type_classes,
            'field_features': field_features,
            'field_classes': field_classes,
            'field_items': field_items,
            'field_weights': field_weights,
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

    def get_params(self):
        """Get parameters.

        Returns
        -------
        params : dict
            Model's hyper-parameters `a`, `b`, `h`.

        """
        with self.db.connect() as con:
            return self.db.read_params(con)

    def set_params(self, a, b, h):
        """Set parameters.

        Parameters
        ----------
        a : float
            Amplitude. Must be strictly positive.
        b : float
            Balance. Must be non-negative.
        h : float
            Entropy. Must be non-negative.

        """
        self.db.check_editable()

        if a <= 0:
            raise ValueError("The parameter 'a' must be strictly positive.")

        if b < 0:
            raise ValueError("The parameter 'b' must be non-negative.")

        if h < 0:
            raise ValueError("The parameter 'h' must be non-negative.")

        with self.db.begin() as con:
            self.db.write_params(con, a=a, b=b, h=h)

    def fit(self, X, y, sample_weight=None):
        """Fit the classifier according to the training data X, y.

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
        self.db.check_editable()
        self._validate(X=X, y=y, sample_weight=sample_weight)

        with self.db.begin() as con:
            self.db.table_corpus.drop(con, checkfirst=True)

        return self.partial_fit(X, y, sample_weight=sample_weight)

    def partial_fit(self, X, y, sample_weight=None):
        """Incremental fit on a batch of samples.

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
        self.db.check_editable()
        self._validate(X=X, y=y, sample_weight=sample_weight)

        if sample_weight is None:
            sample_weight = [1] * len(X)

        with self.db.begin() as con:
            self.db.write_corpus(con, X=X, y=y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Perform classification on the test data X.

        Parameters
        ----------
        X : list of dict of length n_samples
            Test data in the format `[{feature: value, ...}, ...]`.

        Returns
        -------
        y : list of length n_samples
            Predicted target classes for `X`.

        """
        self.db.check_fitted()
        self._validate(X=X)

        with self.db.connect() as con:
            classes = self.db.predict(con, X=X)

        classes = dict(zip(classes[self.db.n], classes[self.db.k]))
        classes = [classes[i] if i in classes else None for i in range(len(X))]

        return classes

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : list of dict of length n_samples
            Test data in the format `[{feature: value, ...}, ...]`.

        Returns
        -------
        y : DataFrame of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in the model.

        """
        self.db.check_fitted()
        self._validate(X=X)

        with self.db.connect() as con:
            proba = self.db.predict_proba(con, X=X)

        proba = self._pivot(proba, index=self.db.n, columns=self.db.k, values=self.db.w)
        proba = proba.reindex(range(len(X))).sparse.to_dense()

        return proba

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
        self.db.check_fitted()

        if X is not None:
            self._validate(X=X, sample_weight=sample_weight)

        with self.db.connect() as con:
            W = self.db.explain(con, X=X, sample_weight=sample_weight)

        return self._pivot(W, index=self.db.j, columns=self.db.k, values=self.db.w)

    def deploy(self):
        """Deploy the instance

        Generate and store the weights that are used for prediction to speed up inference time.
        A deployed instance cannot be modified. To update a deployed instance, undeploy it first.

        """
        with self.db.begin() as con:
            self.db.deploy(con)

    def undeploy(self):
        """Undeploy the instance

        Drop the weights that are used for prediction. Weights will be recomputed each time on-the-fly.
        Useful for development, testing, and incremental fit.

        """
        with self.db.begin() as con:
            self.db.undeploy(con)

    def is_fitted(self):
        """Is fitted?

        Checks whether the instance is fitted.

        Returns
        -------
        is : bool
            Returns `True` if the instance is fitted, `False` otherwise.

        """
        return self.db.is_fitted()

    def is_deployed(self):
        """Is deployed?

        Checks whether the instance is deployed.

        Returns
        -------
        is : bool
            Returns `True` if the instance is deployed, `False` otherwise.

        """
        return self.db.is_deployed()

    @staticmethod
    def _validate(X, y="no_validation", sample_weight=None):
        only_X = isinstance(y, str) and y == "no_validation"

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

            for i, value in enumerate(sample_weight):
                if value < 0:
                    raise ValueError(
                        f"Element {i} of sample_weight contains negative values"
                    )

        if not only_X:
            if len(X) != len(y):
                raise ValueError(
                    "Dimension mismatch. X and y must have the same length"
                )

    @staticmethod
    def _pivot(df, index, columns, values):
        df[values] = df[values].astype(pd.SparseDtype(float))
        df = df.pivot(index=index, columns=columns, values=values)
        df = df.astype(pd.SparseDtype(float, fill_value=0))
        df.rename_axis(None, axis=0, inplace=True)
        df.rename_axis(None, axis=1, inplace=True)
        return df

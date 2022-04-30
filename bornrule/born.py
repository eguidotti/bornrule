import numpy
import scipy.sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, _check_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError


class BornClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, a=0.5, b=1, h=1):
        self.a, self.b, self.h = a, b, h
        self.corpus_, self.classes_, self.weights_ = None, None, None
        self.gpu_, self.dense_, self.sparse_ = False, numpy, scipy.sparse

    def set_params(self, **params):
        self.weights_ = None
        return super().set_params(**params)

    def fit(self, X, y, sample_weight=None):
        self.reset_(X, y)
        return self.partial_fit(X, y, classes=y, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.weights_ = None
        X, y = self.sanitize_(X, y)
        self.check_partial_fit_first_call_(classes)

        if not self.sparse_.issparse(y) and y.ndim == 1:
            y = self.one_hot_(y)

        if sample_weight is not None:
            sample_weight = self.check_sample_weight_(sample_weight, X)
            y = self.multiply_(y, sample_weight.reshape(-1, 1))

        corpus = X.T @ self.multiply_(y, self.power_(self.sum_(X, axis=1), -1))
        self.corpus_ = corpus if self.corpus_ is None else self.corpus_ + corpus

        return self

    def predict(self, X):
        return self.classes_[self.dense_.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        if self.corpus_ is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.")

        u = self.power_(self.power_(self.sanitize_(X), self.a) @ self.get_weights_(), 1. / self.a)
        y = self.normalize_(u, axis=1)

        if self.sparse_.issparse(y):
            y = self.dense_.asarray(y.todense())

        return y

    def explain(self, X=None):
        if X is None:
            return self.get_weights_()

        return self.multiply_(self.get_weights_(), self.power_(self.sanitize_(X).T, self.a))

    def get_weights_(self):
        if self.weights_ is None:
            C_ji = self.corpus_
            if self.b != 0:
                # normalize over the classes
                C_ji = self.normalize_(C_ji, axis=0, p=-self.b)
            if self.b != 1:
                # normalize over the features
                C_ji = self.normalize_(C_ji, axis=1, p=self.b - 1)

            W_ji = self.power_(C_ji, self.a)
            if self.h != 0:
                # probability of class i given feature j
                P_ji = self.normalize_(C_ji, axis=1)
                # compute entropy per feature
                H_j = 1 + self.sum_(self.multiply_(P_ji, self.log_(P_ji)), axis=1) / self.dense_.log(P_ji.shape[1])
                # regularize the weights
                W_ji = self.multiply_(W_ji, self.power_(H_j, self.h))

            self.weights_ = W_ji

        return self.weights_

    def sum_(self, x, axis):
        if self.sparse_.issparse(x):
            return x.sum(axis=axis)

        return x.sum(axis=axis, keepdims=True)

    def multiply_(self, x, y):
        if self.sparse_.issparse(x):
            return x.multiply(y).tocsr()

        return self.dense_.multiply(x, y)

    def power_(self, x, p):
        x = x.copy()

        if self.sparse_.issparse(x):
            x.data = self.dense_.power(x.data, p)

        else:
            nz = self.dense_.nonzero(x)
            x[nz] = self.dense_.power(x[nz], p)

        return x

    def log_(self, x):
        x = x.copy()

        if self.sparse_.issparse(x):
            x.data = self.dense_.log(x.data)

        else:
            nz = self.dense_.nonzero(x)
            x[nz] = self.dense_.log(x[nz])

        return x

    def normalize_(self, x, axis, p=-1):
        return self.multiply_(x, self.power_(self.sum_(x, axis=axis), p))

    def sanitize_(self, X, y=None, dtype=(numpy.float32, numpy.float64)):
        if self.gpu_:
            return X if y is None else (X, y)

        elif y is None:
            return check_array(X, accept_sparse='csr', dtype=dtype)

        else:
            return check_X_y(X, y, accept_sparse='csr', multi_output=True, dtype=dtype)

    def check_sample_weight_(self, sample_weight, X):
        if self.gpu_:
            return sample_weight

        return _check_sample_weight(sample_weight=sample_weight, X=X)

    def check_partial_fit_first_call_(self, classes):
        if self.classes_ is None and classes is None:
            raise ValueError("classes must be passed on the first call to partial_fit.")

        elif classes is not None:
            classes = self.unique_labels_(classes)

            if self.classes_ is not None:
                if not self.dense_.array_equal(self.classes_, classes):
                    raise ValueError(
                        "`classes=%r` is not the same as on last call "
                        "to partial_fit, was: %r" % (classes, self.classes_))

            else:
                self.classes_ = classes

    def one_hot_(self, y):
        n, m = len(y), len(self.classes_)
        idx = {c: i for i, c in enumerate(self.classes_ if not self.gpu_ else self.classes_.get())}

        col = self.dense_.array([idx[c] for c in (y if not self.gpu_ else y.get())])
        row = self.dense_.array(range(0, n))
        val = self.dense_.ones(n)

        return self.sparse_.csr_matrix((val, (row, col)), shape=(n, m))

    def unique_labels_(self, y):
        if self.sparse_.issparse(y) or y.ndim == 2:
            return self.dense_.arange(0, y.shape[1])

        elif self.gpu_:
            return self.dense_.unique(y)

        else:
            return unique_labels(y)

    def reset_(self, X, y):
        self.corpus_, self.classes_, self.weights_ = None, None, None
        self.gpu_, self.dense_, self.sparse_ = False, numpy, scipy.sparse

        try:
            import cupy
            cp_X = cupy.get_array_module(X).__name__ == "cupy"
            cp_y = cupy.get_array_module(y).__name__ == "cupy"

            if cp_X and cp_y:
                self.gpu_, self.dense_, self.sparse_ = True, cupy, cupy.sparse

            elif cp_X and not cp_y:
                raise ValueError("X is on GPU, but y is not.")

            elif not cp_X and cp_y:
                raise ValueError("y is on GPU, but X is not.")

        except ModuleNotFoundError:
            pass

import numpy
import scipy.sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError


class BornClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, a=0.5, b=1., h=1.):
        self.a = a
        self.b = b
        self.h = h

    def fit(self, X, y, sample_weight=None):
        attrs = [
            "gpu_",
            "corpus_",
            "classes_",
            "n_features_in_"
        ]

        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)

        return self.partial_fit(X, y, classes=y, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        X, y = self._sanitize(X, y)

        first_call = self._check_partial_fit_first_call(classes)
        if first_call:
            self.n_features_in_ = X.shape[1]

        if not self._check_encoded(y):
            y = self._one_hot_encoding(y)

        if sample_weight is not None:
            sample_weight = self._check_sample_weight(sample_weight, X)
            y = self._multiply(y, sample_weight.reshape(-1, 1))

        corpus = X.T @ self._multiply(y, self._power(self._sum(X, axis=1), -1))
        self.corpus_ = corpus if first_call else self.corpus_ + corpus

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = self._dense().argmax(proba, axis=1)
        
        return self.classes_[idx]

    def predict_proba(self, X):
        self._check_fitted()

        X = self._sanitize(X)
        u = self._power(self._power(X, self.a) @ self._weights(), 1. / self.a)
        y = self._normalize(u, axis=1)

        if self._sparse().issparse(y):
            y = self._dense().asarray(y.todense())

        return y

    def explain(self, X=None, sample_weight=None):
        self._check_fitted()

        if X is None:
            return self._weights()

        X = self._sanitize(X)
        X = self._power(X, self.a)
        if sample_weight is not None:
            sample_weight = self._check_sample_weight(sample_weight, X)

        E_ji = 0
        W_ji = self._weights()
        X_ki = X @ W_ji
        for k in range(X.shape[0]):

            X_j, X_i = X[k:k+1].T, X_ki[k:k+1]
            X_ji = self._multiply(W_ji, X_j)

            U_i = self._power(X_i, 1. / self.a)
            Y_i = self._normalize(U_i, axis=1)

            if self._sparse().issparse(X_j):
                Z_j = X_j != 0
                X_i = Z_j @ X_i
                Y_i = Z_j @ Y_i

            U_ji = self._power(X_i - X_ji, 1. / self.a)
            Y_ji = self._normalize(U_ji, axis=1)

            D_ji = Y_i - Y_ji
            if sample_weight is not None:
                D_ji = sample_weight[k] * D_ji

            E_ji += D_ji

        return E_ji if sample_weight is not None else E_ji / X.shape[0]

    def _dense(self):
        return cupy if self.gpu_ else numpy

    def _sparse(self):
        return cupy.sparse if self.gpu_ else scipy.sparse

    def _weights(self):
        C_ji = self.corpus_
        if self.b != 0:
            C_ji = self._normalize(C_ji, axis=0, p=self.b)
        if self.b != 1:
            C_ji = self._normalize(C_ji, axis=1, p=1-self.b)

        W_ji = self._power(C_ji, self.a)
        if self.h != 0 and len(self.classes_) > 1:
            P_ji = self._normalize(C_ji, axis=1)
            H_j = 1 + self._sum(self._multiply(P_ji, self._log(P_ji)), axis=1) / self._dense().log(P_ji.shape[1])
            W_ji = self._multiply(W_ji, self._power(H_j, self.h))

        return W_ji

    def _sum(self, x, axis):
        if self._sparse().issparse(x):
            return x.sum(axis=axis)

        if isinstance(x, self._dense().matrix):
            return x.sum(axis=axis)

        return x.sum(axis=axis, keepdims=True)

    def _multiply(self, x, y):
        if self._sparse().issparse(x):
            return x.multiply(y).tocsr()

        if self._sparse().issparse(y):
            return y.multiply(x).tocsr()

        return self._dense().multiply(x, y)

    def _power(self, x, p):
        x = x.copy()

        if self._sparse().issparse(x):
            x.data = self._dense().power(x.data, p)

        else:
            nz = self._dense().nonzero(x)
            x[nz] = self._dense().power(x[nz], p)

        return x

    def _log(self, x):
        x = x.copy()

        if self._sparse().issparse(x):
            x.data = self._dense().log(x.data)

        else:
            nz = self._dense().nonzero(x)
            x[nz] = self._dense().log(x[nz])

        return x

    def _normalize(self, x, axis, p=1.):
        s = self._sum(x, axis)
        n = self._power(s, -p)

        return self._multiply(x, n)

    def _sanitize(self, X, y="no_validation"):
        only_X = isinstance(y, str) and y == "no_validation"

        gpu = self._check_gpu(X=X, y=y if not only_X else None)
        if getattr(self, "gpu_", None) is None:
            self.gpu_ = gpu

        elif self.gpu_ != gpu:
            raise ValueError(
                "X is not on the same device (CPU/GPU) as on last call "
                "to partial_fit, was: %r" % (self.gpu_, ))

        if not self.gpu_:
            kwargs = {
                "accept_sparse": "csr",
                "reset": False,
                "dtype": (numpy.float32, numpy.float64)
            }

            if only_X:
                X = super()._validate_data(X=X, **kwargs)

            else:
                X, y = super()._validate_data(X=X, y=y, multi_output=self._check_encoded(y), **kwargs)

            if not self._check_non_negative(X):
                raise ValueError("X must contain non-negative values")

        return X if only_X else (X, y)

    def _unique_labels(self, y):
        if self._check_encoded(y):
            return self._dense().arange(0, y.shape[1])

        elif self.gpu_:
            return self._dense().unique(y)

        else:
            return unique_labels(y)

    def _one_hot_encoding(self, y):
        classes = self.classes_
        n, m = len(y), len(classes)

        if self.gpu_:
            y = y.get()
            classes = classes.get()

        unseen = set(y) - set(classes)
        if unseen:
            raise ValueError(
                "`classes=%r` were not allowed on first call "
                "to partial_fit" % (unseen, ))

        idx = {c: i for i, c in enumerate(classes)}
        col = self._dense().array([idx[c] for c in y])
        row = self._dense().array(range(0, n))
        val = self._dense().ones(n)

        return self._sparse().csr_matrix((val, (row, col)), shape=(n, m))

    def _check_encoded(self, y):
        return self._sparse().issparse(y) or (getattr(y, "ndim", 0) == 2 and y.shape[1] > 1)

    def _check_non_negative(self, X):
        if self._sparse().issparse(X):
            if self._dense().any(X.data < 0):
                return False

        elif self._dense().any(X < 0):
            return False

        return True

    def _check_sample_weight(self, sample_weight, X):
        if self.gpu_:
            return sample_weight

        return _check_sample_weight(sample_weight=sample_weight, X=X)

    def _check_partial_fit_first_call(self, classes):
        if getattr(self, "classes_", None) is None and classes is None:
            raise ValueError("classes must be passed on the first call to partial_fit.")

        elif classes is not None:
            classes = self._unique_labels(classes)

            if getattr(self, "classes_", None) is not None:
                if not self._dense().array_equal(self.classes_, classes):
                    raise ValueError(
                        "`classes=%r` is not the same as on last call "
                        "to partial_fit, was: %r" % (classes, self.classes_))

            else:
                self.classes_ = classes
                return True

        return False

    def _check_gpu(self, X, y=None):
        try:
            import cupy

            cp_X = cupy.get_array_module(X).__name__ == "cupy"
            if y is None:
                return cp_X

            cp_y = cupy.get_array_module(y).__name__ == "cupy"
            if cp_X and cp_y:
                return True

            elif cp_X and not cp_y:
                raise ValueError("X is on GPU, but y is not.")

            elif not cp_X and cp_y:
                raise ValueError("y is on GPU, but X is not.")

        except ModuleNotFoundError:
            pass

        return False

    def _check_fitted(self):
        if getattr(self, "corpus_", None) is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.")

    def _more_tags(self):
        return {
            'requires_y': True,
            'requires_positive_X': True,
            'X_types': ['2darray', 'sparse'],
            '_xfail_checks': {
                'check_classifiers_classes':
                    'This is a pathological data set for BornClassifier. '
                    'For some specific cases, it predicts less classes than expected',
                'check_classifiers_train':
                    'Test fails because of negative values in X'
            }
        }

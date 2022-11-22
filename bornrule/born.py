import numpy
import scipy.sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError

try:
    import cupy
    gpu_support = True

except ModuleNotFoundError:
    gpu_support = False


class BornClassifier(ClassifierMixin, BaseEstimator):
    """Scikit-learn implementation of Born's Classifier

    This class is compatible with the [scikit-learn](https://scikit-learn.org) ecosystem.
    It supports both dense and sparse input and GPU-accelerated computing via [CuPy](https://cupy.dev).
    This classifier is suitable for classification with non-negative feature vectors.
    The data `X` are treated as unnormalized probability distributions.

    Parameters
    ----------
    a : float
        Amplitude. Must be strictly positive.
    b : float
        Balance. Must be non-negative.
    h : float
        Entropy. Must be non-negative.

    Attributes
    ----------
    gpu_ : bool
        Whether the model was fitted on GPU.
    corpus_ : array-like of shape (n_features_in_, n_classes)
        Fitted corpus.
    classes_ : ndarray of shape (n_classes,)
        Unique classes labels.
    n_features_in_ : int
        Number of features seen during `fit`.

    """

    def __init__(self, a=0.5, b=1., h=1.):
        self.a = a
        self.b = b
        self.h = h

    def fit(self, X, y, sample_weight=None):
        """Fit the classifier according to the training data X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values. If 2d array, this is the probability
            distribution over the `n_classes` for each of the `n_samples`.
        sample_weight : array-like of shape (n_samples,)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
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
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively on different chunks of a dataset so
        as to implement out-of-core or online learning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values. If 2d array, this is the probability
            distribution over the `n_classes` for each of the `n_samples`.
        classes : array-like of shape (n_classes,)
            List of all the classes that can possibly appear in the `y` vector.
            Must be provided at the first call to `partial_fit`, can be omitted in subsequent calls.
        sample_weight : array-like of shape (n_samples,)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        X, y = self._sanitize(X, y)

        first_call = self._check_partial_fit_first_call(classes)
        if first_call:
            self.corpus_ = 0
            self.n_features_in_ = X.shape[1]

        if not self._check_encoded(y):
            y = self._one_hot_encoding(y)

        if sample_weight is not None:
            sample_weight = self._check_sample_weight(sample_weight, X)
            y = self._multiply(y, sample_weight.reshape(-1, 1))

        self.corpus_ += X.T @ self._multiply(y, self._power(self._sum(X, axis=1), -1))

        return self

    def predict(self, X):
        """Perform classification on the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted target classes for `X`.

        """
        proba = self.predict_proba(X)
        idx = self._dense().argmax(proba, axis=1)
        
        return self.classes_[idx]

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        y : ndarray of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in the model.
            The columns correspond to the classes in sorted order, as they appear in the attribute `classes_`.

        """
        self._check_fitted()

        X = self._sanitize(X)
        u = self._power(self._power(X, self.a) @ self._weights(), 1. / self.a)
        y = self._normalize(u, axis=1)

        if self._sparse().issparse(y):
            y = y.todense()

        return self._dense().asarray(y)

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
        X : array-like of shape (n_samples, n_features)
            Test data, where `n_samples` is the number of samples
            and `n_features` is the number of features. If not provided,
            then global weights are returned.
        sample_weight : array-like of shape (n_samples,)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        E : array-like of shape (n_features, n_classes)
            Returns the feature importance for each class in the model.
            The columns correspond to the classes in sorted order, as they appear in the attribute `classes_`.

        """
        self._check_fitted()

        if X is None:
            return self._weights()

        X = self._sanitize(X)
        X = self._normalize(X, axis=1)
        X = self._power(X, self.a)

        if sample_weight is not None:
            sample_weight = self._check_sample_weight(sample_weight, X)
            X = self._multiply(X, sample_weight.reshape(-1, 1))

        return self._multiply(self._weights(), self._sum(X, axis=0).T)

        # X = self._sanitize(X)
        # if sample_weight is not None:
        #     sample_weight = self._check_sample_weight(sample_weight, X)
        #
        # W_jk = self._weights()
        # X_nj = self._power(X, self.a)
        # X_nk = X_nj @ W_jk
        #
        # if self.gpu_ and self._sparse().issparse(X_nj):
        #     X_nj = X_nj.tocsc()
        #
        # E_jk = 0
        # for n in range(X.shape[0]):
        #
        #     X_j, X_k = X_nj[n:n+1].T, X_nk[n:n+1]
        #     X_jk = self._multiply(W_jk, X_j)
        #
        #     U_k = self._power(X_k, 1. / self.a)
        #     Y_k = self._normalize(U_k, axis=1)
        #
        #     if self._sparse().issparse(X_j):
        #         Z_j = X_j != 0
        #         X_k = Z_j @ X_k
        #         Y_k = Z_j @ Y_k
        #
        #     U_jk = self._power(X_k - X_jk, 1. / self.a)
        #     Y_jk = self._normalize(U_jk, axis=1)
        #
        #     D_jk = Y_k - Y_jk
        #     if sample_weight is not None:
        #         D_jk = sample_weight[n] * D_jk
        #
        #     E_jk += D_jk
        #
        # return E_jk if sample_weight is not None else E_jk / X.shape[0]

    def _dense(self):
        return cupy if self.gpu_ else numpy

    def _sparse(self):
        return cupy.sparse if self.gpu_ else scipy.sparse

    def _weights(self):
        P_jk = self.corpus_
        if self.b != 0:
            P_jk = self._multiply(P_jk, self._power(self._sum(self.corpus_, axis=0), -self.b))
        if self.b != 1:
            P_jk = self._multiply(P_jk, self._power(self._sum(self.corpus_, axis=1), self.b-1))

        W_jk = self._power(P_jk, self.a)
        if self.h != 0 and len(self.classes_) > 1:
            P_jk = self._normalize(P_jk, axis=1)
            H_j = 1 + self._sum(self._multiply(P_jk, self._log(P_jk)), axis=1) / self._dense().log(P_jk.shape[1])
            W_jk = self._multiply(W_jk, self._power(H_j, self.h))

        return W_jk

    def _sum(self, x, axis):
        if self._sparse().issparse(x):
            return x.sum(axis=axis)

        return self._dense().asarray(x).sum(axis=axis, keepdims=True)

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
            raise ValueError("classes must be passed on the first call to partial_fit")

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
        if not gpu_support:
            return False

        cp_X = cupy.get_array_module(X).__name__ == "cupy"
        if y is None:
            return cp_X

        cp_y = cupy.get_array_module(y).__name__ == "cupy"
        if cp_X == cp_y:
            return cp_X

        elif cp_X and not cp_y:
            raise ValueError("X is on GPU, but y is not")

        elif not cp_X and cp_y:
            raise ValueError("y is on GPU, but X is not")

    def _check_fitted(self):
        if getattr(self, "corpus_", None) is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator")

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

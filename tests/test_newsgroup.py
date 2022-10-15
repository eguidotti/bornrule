import nltk
import pytest
import warnings
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import SparseEfficiencyWarning
from bornrule import BornClassifier


# Download dataset
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

# Vectorizer
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=False)

# Transform train set
X_train = vectorizer.fit_transform(train.data)
y_train = np.array([train.target_names[t] for t in train.target])

# Transform test set
X_test = vectorizer.transform(test.data)
y_test = np.array([test.target_names[t] for t in test.target])


# Sparse to numpy array
def dense(x):
    return np.asarray(x.todense())


@pytest.mark.parametrize(
    "classifier, accuracy", [
        (BornClassifier(), 0.87267657992565)
    ])
def test_accuracy(classifier, accuracy):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)

    # Check accuracy score
    assert np.allclose(accuracy, score), f"Accuracy score = {score} instead of {accuracy}"


@pytest.mark.parametrize(
    "classifier, item", [
        (BornClassifier(), 0),
        (BornClassifier(a=1, b=0, h=0), 100),
        (BornClassifier(a=0.3, b=0.3, h=0.3), 101)
    ])
def test_explain_single(classifier, item):
    classifier.fit(X_train, y_train)

    # Explain one item
    X = X_test[item]
    E = classifier.explain(X)

    # For each feature
    for i in X.tocoo().col:

        # X with and without feature i
        X1, X2 = X.copy(), X.copy()
        X2[0, i] = 0

        # Compute probabilities with and without feature i
        P1 = classifier.predict_proba(X1)
        P2 = classifier.predict_proba(X2)

        # Check that explanation weights match the difference in classification probabilities
        assert np.allclose(P1 - P2, E[i].todense()), f"Explanation does not match for feature {i}"


@pytest.mark.parametrize(
    "classifier, nitems", [
        (BornClassifier(), 100),
        (BornClassifier(a=1, b=0, h=0), 100),
        (BornClassifier(a=0.3, b=0.3, h=0.3), 100)
    ])
def test_explain_multiple(classifier, nitems):
    classifier.fit(X_train, y_train)

    # Explain multiple items
    X_test_1 = X_test[0:nitems]
    E = classifier.explain(X_test_1)

    # Drop one feature
    X_test_2 = X_test_1.copy()
    feature = Counter(X_test_2.tocoo().col).most_common(1)[0][0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        X_test_2[:, feature] = 0

    # Compute probabilities with and without the feature
    P1 = classifier.predict_proba(X_test_1)
    P2 = classifier.predict_proba(X_test_2)

    # Check that explanation weights match the average difference in classification probabilities
    Ef = (P1 - P2).sum(axis=0) / X_test_1.shape[0]
    assert np.allclose(Ef, E[feature].todense()), f"Explanation does not match"


@pytest.mark.parametrize(
    "classifier, nitems", [
        (BornClassifier(), 100),
        (BornClassifier(a=1, b=0, h=0), 100),
        (BornClassifier(a=0.3, b=0.3, h=0.3), 100)
    ])
def test_explain_weights(classifier, nitems):
    classifier.fit(X_train, y_train)

    # Explain multiple items using weights
    X_test_1 = X_test[0:nitems]
    weights = np.array(range(nitems))
    E = classifier.explain(X_test_1, sample_weight=weights)

    # Drop one feature
    X_test_2 = X_test_1.copy()
    feature = Counter(X_test_2.tocoo().col).most_common(1)[0][0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        X_test_2[:, feature] = 0

    # Compute probabilities with and without the feature
    P1 = classifier.predict_proba(X_test_1)
    P2 = classifier.predict_proba(X_test_2)

    # Check that explanation weights match the weighted sum of differences in classification probabilities
    Ef = ((P1 - P2) * weights.reshape((-1, 1))).sum(axis=0)
    assert np.allclose(Ef, E[feature].todense()), f"Explanation does not match"


@pytest.mark.parametrize(
    "classifier", [
        (BornClassifier())
    ])
def test_sparse(classifier):

    # Subset to speed up testing
    Xn_train, yn_train, Xn_test = X_train[0:100], y_train[0:100], X_test[0:10]

    # Fit sparse
    classifier.fit(Xn_train, yn_train)

    assert all(classifier.predict(Xn_train) == classifier.predict(dense(Xn_train))), \
        "predict does not match"

    assert np.allclose(classifier.predict_proba(Xn_train), classifier.predict_proba(dense(Xn_train))), \
        "predict_proba does not match"

    assert np.allclose(dense(classifier.explain(Xn_test)), classifier.explain(dense(Xn_test))), \
        "explain does not match"

    # Fit dense
    classifier.partial_fit(dense(Xn_train), yn_train)

    assert all(classifier.predict(Xn_train) == classifier.predict(dense(Xn_train))), \
        "predict does not match"

    assert np.allclose(classifier.predict_proba(Xn_train), classifier.predict_proba(dense(Xn_train))), \
        "predict_proba does not match"

    assert np.allclose(classifier.explain(Xn_test), classifier.explain(dense(Xn_test))), \
        "explain does not match"

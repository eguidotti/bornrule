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
vectorizer = TfidfVectorizer()

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
        (BornClassifier(), 0.863515666489)
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

    # Compute probabilities from explanation
    a = classifier.get_params()['a']
    u = np.power(E.sum(axis=0), 1/a)
    y = u / u.sum()

    # Check agreement with predict_proba
    assert np.allclose(y, classifier.predict_proba(X)), "Explanation does not match"


@pytest.mark.parametrize(
    "classifier, nitems", [
        (BornClassifier(), 100),
        (BornClassifier(a=1, b=0, h=0), 100),
        (BornClassifier(a=0.3, b=0.3, h=0.3), 100)
    ])
def test_explain_multiple(classifier, nitems):
    classifier.fit(X_train, y_train)

    # Explain multiple items
    E1 = classifier.explain(X_test[0:nitems])

    # Explain average item
    E2 = classifier.explain(
        sum([X_test[i] / X_test[i].sum() for i in range(0, nitems)]), 
        sample_weight=[nitems]
    )

    # Check explanations
    assert np.allclose(E1.todense(), E2.todense()), f"Explanation does not match"


@pytest.mark.parametrize(
    "classifier, nitems", [
        (BornClassifier(), 100),
        (BornClassifier(a=1, b=0, h=0), 100),
        (BornClassifier(a=0.3, b=0.3, h=0.3), 100)
    ])
def test_explain_weights(classifier, nitems):
    classifier.fit(X_train, y_train)
    sample_weight = range(0, nitems)

    # Explain multiple items with sample_weight
    E1 = classifier.explain(X_test[0:nitems], sample_weight=sample_weight)

    # Explain average item
    E2 = classifier.explain(
        sum([sample_weight[i] * X_test[i] / X_test[i].sum() for i in range(0, nitems)]), 
        sample_weight=[sum(sample_weight)]
    )

    # Check explanations
    assert np.allclose(E1.todense(), E2.todense()), f"Explanation does not match"

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

    assert np.allclose(dense(classifier.explain(Xn_test)), dense(classifier.explain(dense(Xn_test)))), \
        "explain does not match"

    # Fit dense
    classifier.partial_fit(dense(Xn_train), yn_train)

    assert all(classifier.predict(Xn_train) == classifier.predict(dense(Xn_train))), \
        "predict does not match"

    assert np.allclose(classifier.predict_proba(Xn_train), classifier.predict_proba(dense(Xn_train))), \
        "predict_proba does not match"

    assert np.allclose(classifier.explain(Xn_test), classifier.explain(dense(Xn_test))), \
        "explain does not match"

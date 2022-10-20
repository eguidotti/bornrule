import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from bornrule import BornClassifier
from bornrule.sql import BornClassifierSQL


# Convert scipy to list of dict
def bow(x, names):
    bow = []
    for i in range(x.shape[0]):
        item = x[i].tocoo()
        data = item.data
        keys = [names[c] for c in item.col]
        bow.append(dict(zip(keys, data)))
    return bow


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

# Transform to bow
B_test = bow(X_test, names=vectorizer.get_feature_names())
B_train = bow(X_train, names=vectorizer.get_feature_names())


@pytest.mark.parametrize(
    "params, engine", [
        ({"a": 0.5, "b": 1.0, "h": 1.0}, "sqlite:///"),
        ({"a": 0.7, "b": 0.3, "h": 0.4}, "sqlite:///"),
        ({"a": 0.7, "b": 0.3, "h": 0.4}, "sqlite:///test.db")
    ])
def test_sqlite(params, engine):
    skl = BornClassifier()
    sql = BornClassifierSQL(engine=engine)

    # Parameters
    skl.set_params(**params)
    sql.set_params(**params)

    # Fit
    skl.fit(X_train, y_train)
    sql.fit(B_train, y_train)

    # Subset to speed up testing
    Xn_test, Bn_test = X_test[0:100], B_test[0:100]

    # Check predictions
    assert all(skl.predict(Xn_test) == sql.predict(Bn_test)), \
        f"Predictions do not match"

    # Check probabilities
    assert np.allclose(skl.predict_proba(Xn_test), sql.predict_proba(Bn_test)), \
        f"Probabilities do not match"

    # Incremental fit with a subset of training data
    skl.partial_fit(X_train[0:100], y_train[0:100])
    sql.partial_fit(B_train[0:100], y_train[0:100])

    # Check predictions
    assert all(skl.predict(Xn_test) == sql.predict(Bn_test)), \
        f"Predictions do not match"

    # Check probabilities
    assert np.allclose(skl.predict_proba(Xn_test), sql.predict_proba(Bn_test)), \
        f"Probabilities do not match"

    # Deploy to speed up
    sql.deploy()

    # Classes and vocabulary
    classes, features = skl.classes_, vectorizer.get_feature_names()

    # Check global explanation
    ex1 = sql.explain()
    ex2 = skl.explain()
    ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=features, columns=classes)
    assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
        f"Global explanation does not match"

    # Check local explanation
    for i in range(100):
        ex1 = sql.explain(B_test[i:i+1])
        ex2 = skl.explain(X_test[i:i+1])
        ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=features, columns=classes)
        assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
            f"Local explanation does not match for item {i}"

    # Check multiple explanations
    ex1 = sql.explain(B_test[0:100])
    ex2 = skl.explain(X_test[0:100])
    ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=features, columns=classes)
    assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
        f"Multiple-items explanation does not match"

    # Check explanations with sample_weight
    sample_weight = list(range(100))
    ex1 = sql.explain(B_test[0:100], sample_weight=sample_weight)
    ex2 = skl.explain(X_test[0:100], sample_weight=sample_weight)
    ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=features, columns=classes)
    assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
        f"Explanation with sample_weight does not match"

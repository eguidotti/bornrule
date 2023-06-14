import pytest
import sqlite3
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

# Feature names
if hasattr(vectorizer, 'get_feature_names_out'):
    feature_names = vectorizer.get_feature_names_out()
else:
    feature_names = vectorizer.get_feature_names()

# Transform to bow
B_test = bow(X_test, names=feature_names)
B_train = bow(X_train, names=feature_names)

# Test engine
testengine = "sqlite:///test.db"


def test_version():
    ver = sqlite3.sqlite_version.split(".")
    assert int(ver[0]) > 3 or (int(ver[0]) == 3 and int(ver[1]) >= 24), \
        f"Required SQLite v3.24.0+ but version {'.'.join(ver)} is provided."


def test_get_params():
    sql = BornClassifierSQL()
    assert sql.get_params() == dict(a=0.5, b=1, h=1), "Params default have changed"


def test_set_params():
    sql = BornClassifierSQL()
    sql.set_params(a=99)
    assert sql.get_params()['a'] == 99, "Params are not set"

    sql1 = BornClassifierSQL(id="test_set_params_1", engine=testengine)
    sql1.set_params(a=99)
    assert sql1.get_params() == dict(a=99, b=1, h=1), "Params are not set"

    sql2 = BornClassifierSQL(id="test_set_params_2", engine=testengine)
    sql2.set_params(b=99)
    assert sql2.get_params() == dict(a=0.5, b=99, h=1), "Params are not set"


def test_err_params():
    sql = BornClassifierSQL()
    with pytest.raises(Exception):
        sql.set_params(wrong=True)


def test_fit_predict():
    sql = BornClassifierSQL()
    sql.fit(B_train[0:10], y_train[0:10])
    pred = sql.predict(B_test[0:10])

    sql1 = BornClassifierSQL(id="test_fit_predict_1", engine=testengine)
    sql1.fit(B_train[0:10], y_train[0:10])
    pred1 = sql1.predict(B_test[0:10])
    assert pred1 == pred, "Predictions do not match"

    sql2 = BornClassifierSQL(id="test_fit_predict_2", engine=testengine)
    sql2.fit(B_train[0:10], y_train[0:10])
    pred2 = sql2.predict(B_test[0:10])
    assert pred2 == pred, "Predictions do not match"


def test_predict():
    sql = BornClassifierSQL()
    sql.fit(B_train[0:10], [1] * 10)
    pred = sql.predict(B_test[0:10])
    assert all([p == 1 for p in pred]), "Broken predictions when the model is fitted with only one class"


@pytest.mark.parametrize(
    "params, id, engine", [
        ({"a": 0.5, "b": 1.0, "h": 1.0}, "test_sqlite_1", "sqlite:///"),
        ({"a": 0.7, "b": 0.3, "h": 0.4}, "test_sqlite_2", "sqlite:///"),
        ({"a": 0.5, "b": 1.0, "h": 1.0}, "test_sqlite_3", testengine),
        ({"a": 0.7, "b": 0.3, "h": 0.4}, "test_sqlite_4", testengine)
    ])
def test_sqlite(params, id, engine):
    skl = BornClassifier()
    sql = BornClassifierSQL(id=id, engine=engine)

    # Fit
    sql.fit(B_train[0:100], y_train[0:100])

    # Parameters
    skl.set_params(**params)
    sql.set_params(**params)

    # Overwrite fit
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

    # Check global explanation
    ex1 = sql.explain()
    ex2 = skl.explain()
    ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=feature_names, columns=skl.classes_)
    assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
        f"Global explanation does not match"

    # Check local explanation
    for i in range(100):
        ex1 = sql.explain(B_test[i:i+1])
        ex2 = skl.explain(X_test[i:i+1])
        ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=feature_names, columns=skl.classes_)
        assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
            f"Local explanation does not match for item {i}"

    # Check multiple explanations
    ex1 = sql.explain(B_test[0:100])
    ex2 = skl.explain(X_test[0:100])
    ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=feature_names, columns=skl.classes_)
    assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
        f"Multiple-items explanation does not match"

    # Check explanations with sample_weight
    sample_weight = list(range(100))
    ex1 = sql.explain(B_test[0:100], sample_weight=sample_weight)
    ex2 = skl.explain(X_test[0:100], sample_weight=sample_weight)
    ex2 = pd.DataFrame.sparse.from_spmatrix(ex2, index=feature_names, columns=skl.classes_)
    assert np.allclose(ex1, ex2.loc[ex1.index][ex1.columns]), \
        f"Explanation with sample_weight does not match"

    # Undeploy
    sql.undeploy()

    # Check deployed
    assert not sql.is_deployed(), "Undeployed instance is still deployed"

    # Check fitted
    assert sql.is_fitted(), "Undeployed instance is not fitted"

    # Check params and corpus
    with sql.db.connect() as con:
        assert sql.db.is_params(con), "Undeployed instance has no params"
        assert sql.db.is_corpus(con), "Undeployed instance has no corpus"

    # Deep deploy
    sql.deploy(deep=True)

    # Check deployed
    assert sql.is_deployed(), "Deep deployed instance is not deployed"

    # Check fitted
    assert sql.is_fitted(), "Deep deployed instance is not fitted"

    # Check corpus
    with sql.db.connect() as con:
        assert sql.db.is_params(con), "Deep deployed instance does not have params"
        assert not sql.db.is_corpus(con), "Deep deployed instance does have corpus"

    # Check error on undeploy of deep deployed instance
    with pytest.raises(ValueError):
        sql.undeploy()

    # Deep undeploy
    sql.undeploy(deep=True)

    # Check empty parameters
    with sql.db.connect() as con:
        assert not sql.db.is_params(con), "Parameters are not empty after deep undeploy"

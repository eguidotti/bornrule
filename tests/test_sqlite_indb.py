import numpy as np
import pandas as pd
from bornrule.sql import BornClassifierSQL


# in-memory
bc1 = BornClassifierSQL()

# db config 1
bc2 = BornClassifierSQL('c1', engine="sqlite:///zoo.db", configs={
  'class': ('zoo', 'animal_id', 'class_type'),
    'features': [
        ('zoo', 'animal_id', 'hair'),
        ('zoo', 'animal_id', 'feathers'),
    ]
})

# db config 2
bc3 = BornClassifierSQL('c2', engine="sqlite:///zoo.db", configs={
  'class': "SELECT animal_id AS item, class_type AS class, 1 AS weight FROM zoo",
    'features': [
        "SELECT animal_id AS item, 'zoo:hair:'||hair AS feature, 1 as weight FROM zoo",
        "SELECT animal_id AS item, 'zoo:feathers:'||feathers AS feature, 1 as weight FROM zoo"
    ]
})

# write zoo to db
zoo = pd.read_csv('./tests/zoo.csv')
with bc2.db.connect() as con:
  zoo['animal_id'] = zoo.index
  zoo.to_sql("zoo", con)

# X, y
X = [{f'zoo:hair:{h}': 1, f'zoo:feathers:{f}': 1} for h, f in zip(zoo.hair, zoo.feathers)]
y = list(zoo.class_type)
sample_weights = list(zoo['legs'])


def test_fit():
    bc1.fit(X, y)
    bc2.fit("SELECT animal_id AS item FROM zoo")
    bc3.fit("SELECT animal_id AS item FROM zoo")

    ex1 = bc1.explain()
    ex2 = bc2.explain()
    ex3 = bc3.explain()

    assert ex1.equals(ex2) and ex1.equals(ex3), "fit does not match"


def test_fit_sample_weights():
    bc1.fit(X, y, sample_weights)
    bc2.fit("SELECT animal_id AS item FROM zoo", sample_weight="SELECT animal_id AS item, legs AS weight FROM zoo")
    bc3.fit("SELECT animal_id AS item FROM zoo", sample_weight="SELECT animal_id AS item, legs AS weight FROM zoo")

    ex1 = bc1.explain()
    ex2 = bc2.explain()
    ex3 = bc3.explain()

    assert ex1.equals(ex2) and ex1.equals(ex3), "fit with sample_weights does not match"


def test_unlearn():
    bc1.fit(X[0:50], y[0:50])

    bc2.fit("SELECT animal_id AS item FROM zoo")
    bc2.partial_fit("SELECT animal_id AS item FROM zoo WHERE animal_id >= 50", sample_weight=-1)
    
    bc3.fit("SELECT animal_id AS item FROM zoo")
    bc3.partial_fit("SELECT animal_id AS item FROM zoo WHERE animal_id >= 50", sample_weight=-1)

    ex1 = bc1.explain()
    ex2 = bc2.explain()
    ex3 = bc3.explain()

    assert ex1.equals(ex2) and ex1.equals(ex3), "unlearn does not match"


def test_unlearn_sample_weights():
    bc1.fit(X[0:50], y[0:50], sample_weights[0:50])

    bc2.fit("SELECT animal_id AS item FROM zoo", sample_weight="SELECT animal_id AS item, legs AS weight FROM zoo")
    bc2.partial_fit("SELECT animal_id AS item FROM zoo WHERE animal_id >= 50", sample_weight="SELECT animal_id AS item, -legs AS weight FROM zoo")

    bc3.fit("SELECT animal_id AS item FROM zoo", sample_weight="SELECT animal_id AS item, legs AS weight FROM zoo")
    bc3.partial_fit("SELECT animal_id AS item FROM zoo WHERE animal_id >= 50", sample_weight="SELECT animal_id AS item, -legs AS weight FROM zoo")

    ex1 = bc1.explain()
    ex2 = bc2.explain()
    ex3 = bc3.explain()

    assert ex1.equals(ex2) and ex1.equals(ex3), "unlearn with sample_weights does not match"


def test_predict():
    bc1.fit(X, y)
    bc2.fit("SELECT animal_id AS item FROM zoo")
    bc3.fit("SELECT animal_id AS item FROM zoo")

    pred1 = bc1.predict(X)
    pred2 = bc2.predict(X)
    pred3 = bc3.predict(X)
    pred4 = bc2.predict("SELECT animal_id AS item FROM zoo")
    pred5 = bc3.predict("SELECT animal_id AS item FROM zoo")
    
    assert pred1.equals(pred2), "predict does not match (1, 2)"
    assert pred1.equals(pred3), "predict does not match (1, 3)"
    assert pred1.equals(pred4), "predict does not match (1, 4)"
    assert pred1.equals(pred5), "predict does not match (1, 5)"


def test_predict_proba():
    bc1.fit(X, y)
    bc2.fit("SELECT animal_id AS item FROM zoo")
    bc3.fit("SELECT animal_id AS item FROM zoo")

    pred1 = bc1.predict_proba(X)
    pred2 = bc2.predict_proba(X)
    pred3 = bc3.predict_proba(X)
    pred4 = bc2.predict_proba("SELECT animal_id AS item FROM zoo")
    pred5 = bc3.predict_proba("SELECT animal_id AS item FROM zoo")
    
    assert pred1.equals(pred2), "predict proba does not match (1, 2)"
    assert pred1.equals(pred3), "predict proba does not match (1, 3)"
    assert pred1.equals(pred4), "predict proba does not match (1, 4)"
    assert pred1.equals(pred5), "predict proba does not match (1, 5)"


def test_explain():
    bc1.fit(X, y)
    bc2.fit("SELECT animal_id AS item FROM zoo")
    bc3.fit("SELECT animal_id AS item FROM zoo")

    ex1 = bc1.explain(X)
    ex2 = bc2.explain(X)
    ex3 = bc3.explain(X)
    ex4 = bc2.explain("SELECT animal_id AS item FROM zoo")
    ex5 = bc3.explain("SELECT animal_id AS item FROM zoo")
    
    assert ex1.equals(ex2), "explain does not match (1, 2)"
    assert ex1.equals(ex3), "explain does not match (1, 3)"
    assert ex1.equals(ex4), "explain does not match (1, 4)"
    assert ex1.equals(ex5), "explain does not match (1, 5)"


def test_explain_sample_weights():
    bc1.fit(X, y)
    bc2.fit("SELECT animal_id AS item FROM zoo")
    bc3.fit("SELECT animal_id AS item FROM zoo")

    ex1 = bc1.explain(X[13:97], sample_weight=sample_weights[13:97])
    ex2 = bc2.explain(X[13:97], sample_weight=sample_weights[13:97])
    ex3 = bc3.explain(X[13:97], sample_weight=sample_weights[13:97])
    ex4 = bc2.explain("SELECT animal_id AS item FROM zoo WHERE animal_id BETWEEN 13 AND 96", sample_weight="SELECT animal_id AS item, legs AS weight FROM zoo")
    ex5 = bc3.explain("SELECT animal_id AS item FROM zoo WHERE animal_id BETWEEN 13 AND 96", sample_weight="SELECT animal_id AS item, legs AS weight FROM zoo")
    
    assert ex1.equals(ex2), "explain with sample_weights does not match (1, 2)"
    assert ex1.equals(ex3), "explain with sample_weights does not match (1, 3)"
    assert ex1.equals(ex4), "explain with sample_weights does not match (1, 4)"
    assert ex1.equals(ex5), "explain with sample_weights does not match (1, 5)"

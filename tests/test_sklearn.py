from bornrule import BornClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks([BornClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

import numpy as np
import pytest
from sklearn import metrics as skm

from src.metrics import accuracy, confusion_matrix, f1_score, precision, recall


@pytest.mark.parametrize("seed", range(5))
def test_agrees_with_sklearn_on_random_labels(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred))
    assert accuracy(y_true, y_pred) == pytest.approx(skm.accuracy_score(y_true, y_pred))
    assert precision(y_true, y_pred) == pytest.approx(skm.precision_score(y_true, y_pred))
    assert recall(y_true, y_pred) == pytest.approx(skm.recall_score(y_true, y_pred))
    assert f1_score(y_true, y_pred) == pytest.approx(skm.f1_score(y_true, y_pred))


def test_hand_computed_example():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0])          # tp=2 fn=1 fp=1 tn=3
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), [[3, 1], [1, 2]])
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert accuracy(y_true, y_pred) == pytest.approx(5 / 7)


def test_zero_division_returns_zero():
    y_true = np.array([1, 0, 1])
    never_positive = np.zeros(3, dtype=int)
    assert precision(y_true, never_positive) == 0.0
    assert f1_score(y_true, never_positive) == 0.0
    assert recall(np.zeros(3, dtype=int), np.zeros(3, dtype=int)) == 0.0


def test_accepts_lists_and_float_labels():
    assert f1_score([1, 0, 1], [1.0, 0.0, 1.0]) == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        f1_score([1, 0], [1])

"""Binary classification metrics, written by hand. Positive class = 1 (Spam).

Undefined ratios (0/0) return 0.0, matching sklearn's zero_division=0 behaviour.
"""
from __future__ import annotations

import numpy as np


def _as_int_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """[[tn, fp], [fn, tp]] - rows are the true class, columns the predicted class."""
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    return float((y_true == y_pred).mean())


def precision(y_true, y_pred) -> float:
    """Of the messages we flagged as spam, how many really were spam."""
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true, y_pred) -> float:
    """Of the real spam, how much we caught."""
    (_, _), (fn, tp) = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true, y_pred) -> float:
    """Harmonic mean of precision and recall on the Spam class - the assignment's metric."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0

"""Validation metrics with Churn = Yes (class 1) as the positive class."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

POSITIVE_CLASS = 1


def positive_scores(model, X) -> np.ndarray:
    """Probability of class 1, looked up in classes_ rather than assumed to be column 1."""
    column = list(model.classes_).index(POSITIVE_CLASS)
    return model.predict_proba(X)[:, column]


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    scores = positive_scores(model, X)
    predicted = (scores >= threshold).astype(int)
    # Rows = actual [0, 1], columns = predicted [0, 1].
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

"""Milestone 1: two numeric features, one simple classifier and a majority baseline (validation only)."""

import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from churnlab.data import load_raw, parse_numeric, prepare_target, validate_dataset
from churnlab.split import split_data

SEED = 42
FIRST_FEATURES = ["tenure", "MonthlyCharges"]

df = load_raw()
validate_dataset(df)
train_df, valid_df, _test_df = split_data(df, SEED)  # test stays reserved


def numeric_inputs(part):
    return part[FIRST_FEATURES].apply(parse_numeric)


X_train, y_train = numeric_inputs(train_df), prepare_target(train_df)
X_valid, y_valid = numeric_inputs(valid_df), prepare_target(valid_df)

report = {
    "positive_class": "Churn = Yes (1)",
    "features": FIRST_FEATURES,
    "seed": SEED,
    "confusion_matrix_layout": "rows = actual [0 No, 1 Yes], columns = predicted [0 No, 1 Yes]",
}
models = {
    "majority_baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000),
}
for name, model in models.items():
    model.fit(X_train, y_train)  # baseline is fitted on training data too
    predicted = model.predict(X_valid)
    report[name] = {
        "accuracy": float(accuracy_score(y_valid, predicted)),
        "confusion_matrix": confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist(),
        "predicted_churners": int(predicted.sum()),
    }

out = Path("reports/tp02/first_model.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))

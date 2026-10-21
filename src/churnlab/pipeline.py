"""Construction of unfitted preprocessing + classifier pipelines.

Nothing in this module reads data or calls fit.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from churnlab.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Documented preprocessing settings; also written to the artifact metadata.
PREPROCESSING = {
    "numeric": {
        "columns": NUMERIC_FEATURES,
        "imputer": "SimpleImputer(strategy='median')  # training medians",
        "scaler": "StandardScaler()  # training mean / std",
    },
    "categorical": {
        "columns": CATEGORICAL_FEATURES,
        "imputer": "SimpleImputer(strategy='most_frequent')  # training mode",
        "encoder": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
        "unknown_category_policy": "encoded as all zeros for that column (no error)",
    },
}

CLASSIFIERS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # remainder="drop": any extra column (customerID, Churn, ...) never reaches the model.
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_classifier(kind: str, params: dict):
    return CLASSIFIERS[kind](**params)


def build_pipeline(classifier) -> Pipeline:
    """A fresh preprocessor every call, so candidates never share fitted state."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])

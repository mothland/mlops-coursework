"""Focused reliability checks for the TP02 pipeline (needs data/raw/telco.csv)."""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from churnlab.data import (
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RAW_PATH,
    load_raw,
    parse_numeric,
    prepare_features,
    prepare_target,
)
from churnlab.pipeline import build_classifier, build_pipeline
from churnlab.split import assert_disjoint, split_data

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="run scripts/download_data.py first")

SEED = 42


@pytest.fixture(scope="module")
def splits():
    train_df, valid_df, test_df = split_data(load_raw(), SEED)
    return {"train": train_df, "valid": valid_df, "test": test_df}


@pytest.fixture(scope="module")
def fitted(splits):
    pipeline = build_pipeline(build_classifier("logistic_regression", {"max_iter": 1000}))
    train = splits["train"]
    return pipeline.fit(prepare_features(train), prepare_target(train))


def numeric_imputer(pipeline):
    return pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]


def test_customer_ids_are_disjoint_and_complete(splits):
    assert_disjoint(splits)
    ids = [set(part[ID_COLUMN]) for part in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(s) for s in ids) == 7043


def test_split_is_reproducible(splits):
    again = split_data(load_raw(), SEED)
    for part, original in zip(again, splits.values()):
        assert part[ID_COLUMN].tolist() == original[ID_COLUMN].tolist()


def test_imputer_medians_come_from_training_only(splits, fitted):
    train_medians = prepare_features(splits["train"])[NUMERIC_FEATURES].median()
    full_medians = prepare_features(load_raw())[NUMERIC_FEATURES].median()
    learned = numeric_imputer(fitted).statistics_
    np.testing.assert_allclose(learned, train_medians.to_numpy())
    # TotalCharges median differs between train and the full data, so this is a real test.
    assert train_medians["TotalCharges"] != full_medians["TotalCharges"]


def test_scaler_mean_comes_from_training_only(splits, fitted):
    train_X = prepare_features(splits["train"])[NUMERIC_FEATURES]
    scaler = fitted.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    imputed = train_X.fillna(train_X.median())
    np.testing.assert_allclose(scaler.mean_, imputed.mean().to_numpy())


def test_predict_does_not_refit(splits, fitted):
    before = numeric_imputer(fitted).statistics_.copy()
    fitted.predict(prepare_features(splits["valid"]))
    fitted.predict(prepare_features(splits["test"].head(5)))
    np.testing.assert_array_equal(numeric_imputer(fitted).statistics_, before)


def test_blank_numeric_is_missing_but_malformed_text_fails():
    parsed = parse_numeric(pd.Series(["29.85", " ", "", None], name="TotalCharges"))
    assert parsed.iloc[0] == 29.85 and parsed.iloc[1:].isna().all()
    with pytest.raises(ValueError, match="Malformed"):
        parse_numeric(pd.Series(["29.85", "abc"], name="TotalCharges"))


def test_missing_numeric_uses_training_median(splits, fitted):
    row = splits["valid"].head(1).copy()
    row["TotalCharges"] = " "
    filled = row.copy()
    filled["TotalCharges"] = str(numeric_imputer(fitted).statistics_[NUMERIC_FEATURES.index("TotalCharges")])
    np.testing.assert_allclose(
        fitted.predict_proba(prepare_features(row)),
        fitted.predict_proba(prepare_features(filled)),
    )


def test_unseen_category_is_encoded_as_zeros(splits, fitted):
    row = splits["valid"].head(1).copy()
    row["Contract"] = "Five year"  # never seen in training
    encoded = fitted.named_steps["preprocessor"].transform(prepare_features(row))
    names = fitted.named_steps["preprocessor"].get_feature_names_out()
    contract_columns = [i for i, name in enumerate(names) if name.startswith("cat__Contract_")]
    assert np.all(encoded[0, contract_columns] == 0)
    assert fitted.predict_proba(prepare_features(row)).shape == (1, 2)  # no error, but no reliability claim


def test_missing_required_feature_fails_visibly(splits, fitted):
    batch = splits["valid"].head(3).drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Missing required feature"):
        prepare_features(batch)
    # Even bypassing the boundary check, the fitted pipeline itself refuses the input.
    X = prepare_features(splits["valid"].head(3)).drop(columns=["Contract"])
    with pytest.raises(ValueError):
        fitted.predict(X)


def test_identifier_and_target_never_reach_the_model(splits, fitted):
    names = fitted.named_steps["preprocessor"].get_feature_names_out()
    assert not any(ID_COLUMN in n or "Churn" in n for n in names)


def test_reload_in_fresh_process_matches(tmp_path, splits, fitted):
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(fitted, artifact)
    batch = splits["valid"][[ID_COLUMN, *FEATURES]].head(10)  # no Churn column
    batch_path = tmp_path / "batch.csv"
    batch.to_csv(batch_path, index=False)
    expected = fitted.predict_proba(prepare_features(batch))[:, 1]

    code = (
        "import sys, joblib, pandas as pd\n"
        "from churnlab.data import prepare_features\n"
        "p = joblib.load(sys.argv[1])\n"
        "X = prepare_features(pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False))\n"
        "for v in p.predict_proba(X)[:, 1]: print(repr(v))\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code, str(artifact), str(batch_path)], text=True)
    reloaded = np.array([float(v) for v in out.split()])
    np.testing.assert_allclose(reloaded, expected, rtol=0, atol=1e-12)

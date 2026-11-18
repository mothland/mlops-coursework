"""TP03 tracking check on a throwaway local store (no server; needs data/raw/telco.csv).

Runs the tracked workflow with two models, then checks what was recorded and that the
registered alias loads a pipeline that reproduces the training process's predictions.
"""

import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from churnlab.data import RAW_PATH, prepare_features
from churnlab.tracking import run

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="run scripts/download_data.py first")


@pytest.fixture(scope="module")
def tracked(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlflow")
    tracking_uri = f"sqlite:///{(tmp / 'mlflow.db').as_posix()}"
    config = json.loads(open("configs/tp03.json", encoding="utf-8").read())
    config["candidates"] = {"logistic_regression": config["candidates"]["logistic_regression"]}
    config["tracking"].update(tracking_uri=tracking_uri, experiment="test-tp03")
    config["output_dir"] = (tmp / "out").as_posix()
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment("test-tp03", artifact_location=(tmp / "artifacts").as_uri())
    handover = run(config_path)
    yield handover, tmp / "out"
    mlflow.set_tracking_uri(None)


def test_run_records_params_metrics_tags_and_datasets(tracked):
    handover, _ = tracked
    recorded = mlflow.get_run(handover["source_run_id"])
    assert recorded.data.params["model_type"] == "logistic_regression"
    assert recorded.data.params["threshold"] == "0.5"
    for key in ["val_average_precision", "val_precision", "val_recall", "val_f1"]:
        assert 0.0 <= recorded.data.metrics[key] <= 1.0
    assert recorded.data.tags["data_sha256"] == handover["data_sha256"]
    assert recorded.data.tags["git_commit"] == handover["git"]["commit"]
    contexts = {
        d.dataset.name: [t.value for t in d.tags if t.key == "mlflow.data.context"]
        for d in recorded.inputs.dataset_inputs
    }
    assert contexts == {"telco_train": ["training"], "telco_validation": ["validation"]}


def test_alias_loads_pipeline_with_same_predictions(tracked):
    handover, out_dir = tracked
    assert handover["selected_candidate"] == "logistic_regression"
    loaded = mlflow.sklearn.load_model(handover["alias_uri"])
    sample = pd.read_csv(out_dir / "inference_sample.csv", dtype=str, keep_default_na=False)
    reference = pd.read_csv(out_dir / "runs" / "logistic_regression" / "reference_predictions.csv")
    proba = loaded.predict_proba(prepare_features(sample))[:, list(loaded.classes_).index(1)]
    np.testing.assert_allclose(proba, reference["churn_probability"], rtol=0, atol=1e-12)

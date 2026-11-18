"""Inference only (TP03): load the registered pipeline through its alias and score a batch.

No training data, no labels, no fit call, and nothing from churnlab.train/churnlab.tracking is
imported. CSV parsing uses the same stateless rules as training (churnlab.data.prepare_features).

    uv run python scripts/predict_mlflow.py --input artifacts/tp03/inference_sample.csv \
        --output reports/tp03/alias_predictions.csv --check-reference --report reports/tp03/reload_check.json

--check-reference downloads the predictions that the *training process* made for the same
customers (logged in the source run) and compares them with the reloaded model's predictions.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from churnlab.data import ID_COLUMN, TARGET, prepare_features

TOLERANCE = 1e-12

# MLflow may print emoji; a redirected Windows console (cp1252) cannot encode them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
parser.add_argument("--model-name", default="telco-churn")
parser.add_argument("--alias", default="candidate")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--check-reference", action="store_true",
                    help="compare with the training process's predictions logged in the source run")
parser.add_argument("--report", type=Path, help="write the comparison as JSON")
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
client = MlflowClient()

# Resolve the alias once, then load that exact version: the printout states what was loaded.
version = client.get_model_version_by_alias(args.model_name, args.alias)
alias_uri = f"models:/{args.model_name}@{args.alias}"
print(f"alias {alias_uri} -> version {version.version} (source run {version.run_id})")
print(f"logged model source: {version.source}")

batch = pd.read_csv(args.input, dtype=str, keep_default_na=False)
if TARGET in batch.columns:
    raise ValueError(f"Inference input must not contain {TARGET}")
X = prepare_features(batch)  # raises on a missing column or a malformed numeric value

pipeline = mlflow.sklearn.load_model(alias_uri)  # our own trusted artifact only (pickle)
positive_column = list(pipeline.classes_).index(1)
probability = pipeline.predict_proba(X)[:, positive_column]

predictions = pd.DataFrame({
    ID_COLUMN: batch[ID_COLUMN] if ID_COLUMN in batch.columns else np.arange(len(batch)),
    "churn_probability": probability,
    "churn_predicted": (probability >= args.threshold).astype(int),
})
args.output.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(args.output, index=False, float_format="%.17g")
print(predictions.to_string(index=False))

if args.check_reference:
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="reference/reference_predictions.csv", dst_path=tmp
        )
        reference = pd.read_csv(local)
    if not reference[ID_COLUMN].astype(str).equals(predictions[ID_COLUMN].astype(str)):
        raise ValueError("Reference and batch customers differ: use the logged inference sample")
    max_diff = float(np.max(np.abs(reference["churn_probability"] - predictions["churn_probability"])))
    classes_equal = bool((reference["churn_predicted"] == predictions["churn_predicted"]).all())
    passed = max_diff <= TOLERANCE and classes_equal
    print(f"\nmax |probability difference| = {max_diff:.3g} (tolerance {TOLERANCE:g}); "
          f"class predictions identical: {classes_equal}")
    if args.report:
        report = {
            "tracking_uri": args.tracking_uri,
            "alias_uri": alias_uri,
            "registered_version": str(version.version),
            "source_run_id": version.run_id,
            "logged_model_source": version.source,
            "input": args.input.as_posix(),
            "rows": len(predictions),
            "threshold": args.threshold,
            "max_abs_probability_difference": max_diff,
            "tolerance": TOLERANCE,
            "class_predictions_identical": classes_equal,
            "passed": passed,
            "fit_called": False,
            "labels_in_input": False,
            "process_id": os.getpid(),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Reloaded predictions do not match the training process")

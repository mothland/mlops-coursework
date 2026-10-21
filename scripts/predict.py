"""Inference only: load a trusted fitted pipeline and score a batch of customers.

No training data, labels or fit calls are involved. CSV parsing and input typing happen here,
at the boundary, with the same stateless rules used during training.

    uv run python scripts/predict.py --input artifacts/tp02/inference_sample.csv \
        --output reports/tp02/predictions.csv \
        --reference artifacts/tp02/reference_predictions.csv
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlab.data import ID_COLUMN, TARGET, prepare_features

TOLERANCE = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument("--artifact", type=Path, default=Path("artifacts/tp02/pipeline.joblib"))
parser.add_argument("--metadata", type=Path, default=Path("artifacts/tp02/metadata.json"))
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path, help="predictions from the training process to compare against")
parser.add_argument("--report", type=Path, help="write the reference comparison as JSON")
args = parser.parse_args()

metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
threshold = metadata["threshold"]

batch = pd.read_csv(args.input, dtype=str, keep_default_na=False)
if TARGET in batch.columns:
    raise ValueError(f"Inference input must not contain {TARGET}")
X = prepare_features(batch)  # raises if a required feature column is missing

pipeline = joblib.load(args.artifact)  # trusted artifact only: joblib can execute code on load
positive_column = list(pipeline.classes_).index(1)
probability = pipeline.predict_proba(X)[:, positive_column]

predictions = pd.DataFrame({
    ID_COLUMN: batch[ID_COLUMN] if ID_COLUMN in batch.columns else np.arange(len(batch)),
    "churn_probability": probability,
    "churn_predicted": (probability >= threshold).astype(int),
})
args.output.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(args.output, index=False, float_format="%.17g")
print(predictions.to_string(index=False))

if args.reference:
    reference = pd.read_csv(args.reference)
    if not reference[ID_COLUMN].astype(str).equals(predictions[ID_COLUMN].astype(str)):
        raise ValueError("Reference and batch customers differ")
    max_diff = float(np.max(np.abs(reference["churn_probability"] - predictions["churn_probability"])))
    classes_equal = bool((reference["churn_predicted"] == predictions["churn_predicted"]).all())
    print(f"\nmax |probability difference| = {max_diff:.3g} (tolerance {TOLERANCE:g}); "
          f"class predictions identical: {classes_equal}")
    if args.report:
        report = {
            "artifact": args.artifact.as_posix(),
            "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
            "input": args.input.as_posix(),
            "reference": args.reference.as_posix(),
            "rows": len(predictions),
            "threshold": threshold,
            "max_abs_probability_difference": max_diff,
            "tolerance": TOLERANCE,
            "class_predictions_identical": classes_equal,
            "passed": max_diff <= TOLERANCE and classes_equal,
            "fit_called": False,
            "labels_in_input": False,
            "process_id": os.getpid(),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if max_diff > TOLERANCE or not classes_equal:
        raise SystemExit("Reloaded predictions do not match the training process")

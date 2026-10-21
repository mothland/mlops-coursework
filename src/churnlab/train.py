"""Training workflow: load -> split -> compare candidates on validation -> save selected pipeline.

The test split is allocated and recorded but never evaluated here.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from churnlab.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    RAW_PATH,
    TARGET,
    load_raw,
    prepare_features,
    prepare_target,
    sha256_of,
    validate_dataset,
)
from churnlab.evaluate import evaluate_model, positive_scores
from churnlab.pipeline import PREPROCESSING, build_classifier, build_pipeline
from churnlab.split import allocation_hash, allocation_table, assert_disjoint, split_data

SAMPLE_SIZE = 20


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_candidate(name: str, spec: dict, X_train, y_train, X_valid, y_valid, threshold):
    pipeline = build_pipeline(build_classifier(spec["kind"], spec["params"]))
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    row = {
        "model": name,
        "kind": spec["kind"],
        "params": json.dumps(spec["params"], sort_keys=True),
        "fit_seconds": round(fit_seconds, 4),
        "threshold": threshold,
        **evaluate_model(pipeline, X_valid, y_valid, threshold),
    }
    return pipeline, row


def select_model(rows: list[dict], selection: dict) -> str:
    metric = selection["metric"]
    order = selection["simplicity_order"]
    best = max(row[metric] for row in rows)
    tied = [row["model"] for row in rows if row[metric] >= best - selection["tie_tolerance"]]
    return min(tied, key=order.index)


def save_with_reference(pipeline, path: Path, sample: pd.DataFrame, threshold: float) -> None:
    """Save a fitted pipeline and, next to it, its predictions on the reference sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    scores = positive_scores(pipeline, prepare_features(sample))
    pd.DataFrame({
        ID_COLUMN: sample[ID_COLUMN].to_numpy(),
        "churn_probability": scores,
        "churn_predicted": (scores >= threshold).astype(int),
    }).to_csv(path.parent / "reference_predictions.csv", index=False, float_format="%.17g")


def git_identity() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ["churnlab", "scikit-learn", "pandas", "numpy", "joblib"]
        },
    }


def run(config_path: Path) -> dict:
    config = load_config(config_path)
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = config["threshold"]

    raw = load_raw(RAW_PATH)
    validate_dataset(raw)
    train_df, valid_df, test_df = split_data(
        raw, config["seed"], config["valid_size"], config["test_size"]
    )
    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    assert_disjoint(splits)
    allocation = allocation_table(splits)
    allocation.to_csv(out_dir / "split_allocation.csv", index=False)

    X_train, y_train = prepare_features(train_df), prepare_target(train_df)
    X_valid, y_valid = prepare_features(valid_df), prepare_target(valid_df)

    specs = {"baseline": config["baseline"], **config["candidates"]}
    fitted, rows = {}, []
    for name, spec in specs.items():
        fitted[name], row = train_candidate(
            name, spec, X_train, y_train, X_valid, y_valid, threshold
        )
        rows.append(row)
        print(f"{name:20s} AP={row['average_precision']:.4f} F1={row['f1']:.4f} "
              f"acc={row['accuracy']:.4f} fit={row['fit_seconds']:.2f}s")

    learned = [row for row in rows if row["model"] != "baseline"]
    selected_name = select_model(learned, config["selection"])
    selected = fitted[selected_name]
    selected_row = next(row for row in rows if row["model"] == selected_name)
    baseline_row = next(row for row in rows if row["model"] == "baseline")
    print(f"selected: {selected_name}")

    comparison = pd.DataFrame(rows)
    comparison["selected"] = comparison["model"].eq(selected_name)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    # Reference batch for the fresh-process reload check: features only, no Churn.
    sample = valid_df[[ID_COLUMN, *FEATURES]].head(SAMPLE_SIZE)
    sample.to_csv(out_dir / "inference_sample.csv", index=False)

    artifact_path = out_dir / "pipeline.joblib"
    save_with_reference(selected, artifact_path, sample, threshold)
    # Every compared candidate is kept too, so each one can be reload-checked.
    candidate_artifacts = {}
    for name, pipeline in fitted.items():
        path = out_dir / "candidates" / name / "pipeline.joblib"
        save_with_reference(pipeline, path, sample, threshold)
        candidate_artifacts[name] = {"path": path.as_posix(), "sha256": sha256_of(path)}

    num_branch = selected.named_steps["preprocessor"].named_transformers_["num"]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": {"path": artifact_path.as_posix(), "sha256": sha256_of(artifact_path)},
        "candidate_artifacts": candidate_artifacts,
        "source": {"path": RAW_PATH.as_posix(), "sha256": sha256_of(RAW_PATH)},
        "config": {"path": Path(config_path).as_posix(), "sha256": sha256_of(config_path)},
        "split": {
            "seed": config["seed"],
            "fractions": {"train": round(1 - config["valid_size"] - config["test_size"], 10),
                          "valid": config["valid_size"], "test": config["test_size"]},
            "stratified_on": TARGET,
            "counts": {name: len(part) for name, part in splits.items()},
            "churn_rate": {name: float(prepare_target(part).mean()) for name, part in splits.items()},
            "allocation_file": (out_dir / "split_allocation.csv").as_posix(),
            "allocation_sha256": allocation_hash(allocation),
            "test_evaluated": False,
        },
        "features": {
            "identifier_not_input": ID_COLUMN,
            "target": TARGET,
            "positive_class": {"label": POSITIVE_LABEL, "encoded_as": 1},
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "input_types": {
                "numeric": "float; NaN, blank or whitespace-only = missing (imputed); other text rejected",
                "categorical": "string; blank = missing (imputed); unseen values encoded as all zeros",
            },
        },
        "preprocessing": PREPROCESSING,
        "fitted_statistics": {
            "numeric_imputer_medians": dict(zip(NUMERIC_FEATURES, num_branch.named_steps["imputer"].statistics_.tolist())),
            "scaler_means": dict(zip(NUMERIC_FEATURES, num_branch.named_steps["scaler"].mean_.tolist())),
        },
        "model": {
            "name": selected_name,
            "kind": config["candidates"][selected_name]["kind"],
            "class": type(selected.named_steps["classifier"]).__name__,
            "params": {k: repr(v) for k, v in selected.named_steps["classifier"].get_params().items()},
        },
        "threshold": threshold,
        "selection": {**config["selection"], "selected": selected_name},
        "validation_metrics": selected_row,
        "baseline_validation_metrics": baseline_row,
        "code": git_identity(),
        "environment": environment(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved {artifact_path} and {out_dir / 'metadata.json'}")
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare churn pipelines.")
    parser.add_argument("--config", type=Path, default=Path("configs/tp02.json"))
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()

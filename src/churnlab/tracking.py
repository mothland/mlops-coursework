"""TP03 tracked training: the TP02 workflow, with one MLflow run per candidate.

Same data, split, features, preprocessing, metrics and selection rule as TP02 (churnlab.train);
this module only adds explicit MLflow logging, model logging and registration.

    uv run python scripts/train_mlflow.py   # needs the tracking server (see README)
"""

import argparse
import json
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.data  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from churnlab.data import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    LABEL_MAP,
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
from churnlab.evaluate import positive_scores  # noqa: E402
from churnlab.pipeline import PREPROCESSING  # noqa: E402
from churnlab.split import (  # noqa: E402
    allocation_hash,
    allocation_table,
    assert_disjoint,
    split_data,
)
from churnlab.train import (  # noqa: E402
    SAMPLE_SIZE,
    environment,
    git_identity,
    load_config,
    select_model,
    train_candidate,
)

# Metric names logged to MLflow (validation only; the test split is never evaluated).
METRIC_KEYS = ["average_precision", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp"]


def dataset_frame(part: pd.DataFrame) -> pd.DataFrame:
    """What a dataset record describes: customerID, the six typed features and the encoded target.

    customerID is kept so that the dataset digest changes when the customer allocation changes.
    It is never passed to the estimator.
    """
    return pd.concat(
        [part[[ID_COLUMN]], prepare_features(part), prepare_target(part)], axis=1
    ).reset_index(drop=True)


def confusion_matrix_figure(row: dict, title: str, path: Path) -> Path:
    matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    display = ConfusionMatrixDisplay(
        confusion_matrix=pd.DataFrame(matrix).to_numpy(), display_labels=["No (0)", "Yes (1)"]
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    display.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_xlabel("Predicted Churn")
    ax.set_ylabel("Actual Churn")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def provenance_record(config, config_path, split_info, code, name, spec, threshold) -> dict:
    return {
        "candidate": {"name": name, "kind": spec["kind"], "params": spec["params"]},
        "data": {
            "source_file": RAW_PATH.as_posix(),
            "source_sha256": sha256_of(RAW_PATH),
            "source_metadata": "data/raw/telco.source.json",
            "note": "The hash identifies the CSV bytes; the CSV itself is not stored in MLflow.",
        },
        "split": split_info,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded": [ID_COLUMN, TARGET],
            "input_types": {
                "numeric": "float; NaN, blank or whitespace-only = missing (imputed); other text rejected",
                "categorical": "string; blank = missing (imputed); unseen values encoded as all zeros",
            },
            "parsing": "churnlab.data.prepare_features (stateless, same at training and inference)",
        },
        "class_mapping": {"target": TARGET, "mapping": LABEL_MAP, "positive_label": POSITIVE_LABEL,
                          "positive_class": 1},
        "preprocessing": PREPROCESSING,
        "evaluation": {
            "dataset": "validation only; test split reserved and not evaluated",
            "primary_metric": "val_average_precision (from continuous P(class 1), column found via classes_)",
            "threshold": threshold,
            "thresholded_metrics": "val_precision, val_recall, val_f1, val_accuracy at score >= threshold",
            "confusion_matrix_order": "rows = actual [0 No, 1 Yes], columns = predicted [0 No, 1 Yes]",
            "selection": config["selection"],
        },
        "config": {"path": Path(config_path).as_posix(), "sha256": sha256_of(config_path)},
        "code": code,
        "environment": environment(),
    }


def run(config_path: Path) -> dict:
    config = load_config(config_path)
    tracking = config["tracking"]
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = config["threshold"]

    mlflow.set_tracking_uri(tracking["tracking_uri"])
    experiment = mlflow.set_experiment(tracking["experiment"])

    # --- Same data and split as TP02 -------------------------------------------------------
    raw = load_raw(RAW_PATH)
    validate_dataset(raw)
    train_df, valid_df, test_df = split_data(
        raw, config["seed"], config["valid_size"], config["test_size"]
    )
    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    assert_disjoint(splits)
    allocation = allocation_table(splits)
    allocation_path = out_dir / "split_allocation.csv"
    allocation.to_csv(allocation_path, index=False)

    X_train, y_train = prepare_features(train_df), prepare_target(train_df)
    X_valid, y_valid = prepare_features(valid_df), prepare_target(valid_df)

    # Fixed reference batch (validation customers, no Churn) for the reload agreement check.
    sample = valid_df[[ID_COLUMN, *FEATURES]].head(SAMPLE_SIZE)
    sample_path = out_dir / "inference_sample.csv"
    sample.to_csv(sample_path, index=False)

    data_sha256 = sha256_of(RAW_PATH)
    code = git_identity()
    split_info = {
        "seed": config["seed"],
        "fractions": {"train": round(1 - config["valid_size"] - config["test_size"], 10),
                      "valid": config["valid_size"], "test": config["test_size"]},
        "stratified_on": TARGET,
        "counts": {name: len(part) for name, part in splits.items()},
        "allocation_file": "split_allocation.csv (logged artifact, customerID -> split)",
        "allocation_sha256": allocation_hash(allocation),
        "test_evaluated": False,
    }

    train_dataset = mlflow.data.from_pandas(
        dataset_frame(train_df), source=RAW_PATH.as_posix(), name="telco_train", targets=TARGET
    )
    valid_dataset = mlflow.data.from_pandas(
        dataset_frame(valid_df), source=RAW_PATH.as_posix(), name="telco_validation", targets=TARGET
    )

    # One id shared by the runs of this comparison, to find them again with search_runs.
    session = uuid.uuid4().hex[:12]
    specs = {"baseline": config["baseline"], **config["candidates"]}
    results = {}
    for name, spec in specs.items():
        run_name = spec.get("run_name", name)
        with mlflow.start_run(run_name=run_name) as active:
            classifier_params = dict(spec["params"])
            mlflow.log_params({
                "model_type": spec["kind"],
                "candidate": name,
                "split_seed": config["seed"],
                "valid_size": config["valid_size"],
                "test_size": config["test_size"],
                "threshold": threshold,
                "features": ",".join(FEATURES),
                **{f"clf__{key}": value for key, value in classifier_params.items()},
            })
            mlflow.set_tags({
                "git_commit": code["commit"],
                "git_dirty": str(code["dirty"]).lower(),
                "data_sha256": data_sha256,
                "allocation_sha256": split_info["allocation_sha256"],
                "config_sha256": sha256_of(config_path),
                "role": "baseline" if name == "baseline" else "candidate",
                "comparison_session": session,
                "primary_metric": "val_average_precision",
            })
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(valid_dataset, context="validation")

            pipeline, row = train_candidate(
                name, spec, X_train, y_train, X_valid, y_valid, threshold
            )
            # Every classifier setting actually used, including sklearn defaults.
            mlflow.log_params({
                f"clf_effective__{key}": repr(value)
                for key, value in pipeline.named_steps["classifier"].get_params().items()
            })
            mlflow.log_metrics({f"val_{key}": float(row[key]) for key in METRIC_KEYS})
            mlflow.log_metric("fit_seconds", row["fit_seconds"])

            run_dir = out_dir / "runs" / name
            cm_path = confusion_matrix_figure(
                row, f"{run_name} - validation (threshold {threshold})",
                run_dir / "val_confusion_matrix.png",
            )
            mlflow.log_artifact(str(cm_path))
            mlflow.log_dict(
                {"labels": [0, 1], "rows": "actual", "columns": "predicted",
                 "matrix": [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]},
                "val_confusion_matrix.json",
            )
            mlflow.log_dict(
                provenance_record(config, config_path, split_info, code, name, spec, threshold),
                "provenance.json",
            )
            mlflow.log_artifact(str(allocation_path))

            # Predictions made by the training process, to compare with a later reload.
            scores = positive_scores(pipeline, prepare_features(sample))
            reference = pd.DataFrame({
                ID_COLUMN: sample[ID_COLUMN].to_numpy(),
                "churn_probability": scores,
                "churn_predicted": (scores >= threshold).astype(int),
            })
            reference_path = run_dir / "reference_predictions.csv"
            reference.to_csv(reference_path, index=False, float_format="%.17g")
            mlflow.log_artifact(str(reference_path), artifact_path="reference")
            mlflow.log_artifact(str(sample_path), artifact_path="reference")

            # Complete fitted preprocessing + classifier, logged with MLflow's model API.
            # Signature inferred from the whole training input, so TotalCharges is recorded as
            # optional (it has blanks); the 3-row example alone would mark every column required.
            signature = infer_signature(X_train, pipeline.predict(X_train))
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=tracking["model_name"],
                signature=signature,
                input_example=X_train.head(3),
                serialization_format="cloudpickle",
            )
            mlflow.set_tag("logged_model_uri", model_info.model_uri)
            results[name] = {"run_id": active.info.run_id, "run_name": run_name,
                             "model_uri": model_info.model_uri, "model_id": model_info.model_id,
                             "row": row}
            print(f"{run_name:24s} run={active.info.run_id} AP={row['average_precision']:.4f} "
                  f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")

    # --- Validation-based selection with the rule fixed in the config ----------------------
    learned = [results[name]["row"] for name in config["candidates"]]
    selected_name = select_model(learned, config["selection"])
    selected = results[selected_name]
    client = MlflowClient()
    for name, result in results.items():
        client.set_tag(result["run_id"], "selected", str(name == selected_name).lower())
    print(f"selected: {selected_name} ({selected['run_name']})")

    # --- Register the selected logged pipeline and point the alias at that version --------
    version = mlflow.register_model(selected["model_uri"], tracking["registered_model"])
    client.set_registered_model_alias(tracking["registered_model"], tracking["alias"], version.version)
    client.set_model_version_tag(tracking["registered_model"], version.version, "source_run_id", selected["run_id"])
    client.set_model_version_tag(tracking["registered_model"], version.version, "git_commit", code["commit"])
    client.set_model_version_tag(tracking["registered_model"], version.version, "data_sha256", data_sha256)
    client.set_model_version_tag(tracking["registered_model"], version.version, "val_average_precision",
                                 f"{selected['row']['average_precision']:.6f}")
    print(f"registered {tracking['registered_model']} version {version.version} "
          f"-> alias @{tracking['alias']}")

    # --- Comparison exported from the tracking server, not transcribed ---------------------
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.comparison_session = '{session}'",
        order_by=["metrics.val_average_precision DESC"],
    )
    columns = ["run_id", "tags.mlflow.runName", "params.model_type", "tags.role", "tags.selected",
               "metrics.val_average_precision", "metrics.val_precision", "metrics.val_recall",
               "metrics.val_f1", "metrics.val_accuracy", "metrics.fit_seconds", "tags.git_commit",
               "tags.git_dirty"]
    runs[columns].to_csv(out_dir / "comparison.csv", index=False)

    handover = {
        "tracking_uri": tracking["tracking_uri"],
        "experiment": {"name": tracking["experiment"], "id": experiment.experiment_id},
        "comparison_session": session,
        "selected_candidate": selected_name,
        "source_run_id": selected["run_id"],
        "source_run_name": selected["run_name"],
        "logged_model_uri": selected["model_uri"],
        "logged_model_id": selected["model_id"],
        "registered_model": tracking["registered_model"],
        "registered_version": str(version.version),
        "alias": tracking["alias"],
        "alias_uri": f"models:/{tracking['registered_model']}@{tracking['alias']}",
        "version_uri": f"models:/{tracking['registered_model']}/{version.version}",
        "git": code,
        "data_sha256": data_sha256,
        "allocation_sha256": split_info["allocation_sha256"],
        "threshold": threshold,
        "validation_metrics": {k: selected["row"][k] for k in METRIC_KEYS},
        "candidates": {name: {"run_id": r["run_id"], "run_name": r["run_name"],
                              "model_uri": r["model_uri"],
                              "val_average_precision": r["row"]["average_precision"]}
                       for name, r in results.items()},
    }
    (out_dir / "handover.json").write_text(json.dumps(handover, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out_dir / 'handover.json'} and {out_dir / 'comparison.csv'}")
    return handover


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Tracked training of churn pipelines with MLflow.")
    parser.add_argument("--config", type=Path, default=Path("configs/tp03.json"))
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()

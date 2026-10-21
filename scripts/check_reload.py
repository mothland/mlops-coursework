"""Reload check for every compared candidate and the selected artifact.

Each artifact is loaded by scripts/predict.py in its own fresh Python process (no fit, no labels)
and compared with the predictions made by the training process.

    uv run python scripts/check_reload.py
"""

import json
import subprocess
import sys
from pathlib import Path

ARTIFACTS = Path("artifacts/tp02")
REPORTS = Path("reports/tp02/reload")

targets = {"selected": ARTIFACTS} | {
    path.name: path for path in sorted((ARTIFACTS / "candidates").iterdir()) if path.is_dir()
}

results = {}
for name, folder in targets.items():
    report = REPORTS / f"{name}.json"
    subprocess.run(
        [
            sys.executable, "scripts/predict.py",
            "--artifact", str(folder / "pipeline.joblib"),
            "--metadata", str(ARTIFACTS / "metadata.json"),
            "--input", str(ARTIFACTS / "inference_sample.csv"),
            "--output", str(REPORTS / f"{name}_predictions.csv"),
            "--reference", str(folder / "reference_predictions.csv"),
            "--report", str(report),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    results[name] = json.loads(report.read_text(encoding="utf-8"))
    r = results[name]
    print(f"{name:20s} max diff={r['max_abs_probability_difference']:.3g} "
          f"classes identical={r['class_predictions_identical']} pid={r['process_id']} "
          f"passed={r['passed']}")

summary = {"all_passed": all(r["passed"] for r in results.values()), "models": results}
(REPORTS.parent / "reload_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
if not summary["all_passed"]:
    raise SystemExit("At least one reloaded artifact does not match")

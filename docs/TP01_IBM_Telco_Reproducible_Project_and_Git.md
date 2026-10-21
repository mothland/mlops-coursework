# TP01 - A Reproducible Churn Project with Git

**ESILV - Machine Learning Operations**  
**Dataset:** IBM Telco Customer Churn  
**Suggested duration:** 90 minutes  
**Format:** work in pairs; submit individual explanations  
**Prerequisites:** basic Python, Git, a GitHub account, and uv installed

> **Driving question:** Can another engineer obtain the same input data and run your project without guessing?

## 1. The project you will build throughout the course

A telecommunications company wants to identify customers who may leave its service. Your team will develop a churn prediction system, then track its training, package it, deploy it and monitor it.

Today, you prepare the foundation: a declared Python environment, an identifiable data source, a small executable inspection script, and a reviewed Git change. You will train models in a later TP.

### Learning outcomes

By the end of this TP, you should be able to:

- Distinguish a package manager, an environment and a dependency lock file.
- Recreate a project environment from `pyproject.toml` and `uv.lock`.
- Identify the dataset used by a project through its source and SHA-256 hash.
- Make and review a focused change using a branch and pull request.
- Explain which parts of ML reproducibility are still missing after this TP.

### Time plan

| Part | Task | Time |
|---|---|---:|
| A | Understand the case and project layout | 10 min |
| B | Create and inspect the environment | 15 min |
| C | Inspect and fingerprint the Telco dataset | 20 min |
| D | Recreate the environment | 15 min |
| E | Collaborate through a pull request | 20 min |
| F | Recovery exercise and evidence | 10 min |

The time plan assumes the tools and course starter project are already available. Installation troubleshooting is preparation time.

## 2. Before you start

Use the instructor-provided **`churnlab-2026` reference project**, included in the complete TP package. This handout extends TP01 of that project; it does not ask you to create an unrelated project from scratch.

Extract the package and open a terminal in `churnlab-2026/`, the directory containing `pyproject.toml`. Keep your own copy of the project and your own GitHub repository.

Use **Python 3.11** for the common classroom route. The reference project also accepts Python 3.12. The supplied lock file was prepared with uv 0.12.23. Use that version or a compatible version confirmed by the instructor.

Check your tools:

```sh
python --version
git --version
uv --version
```

On some systems, the Python command is `python3`. After setup, the commands in this handout use `uv run python`, which selects the project interpreter.

> **Shell convention:** Git and uv commands work in common terminals. Edit the Python files below in your editor, then run them as files. Later Docker/Kubernetes TPs use a Linux or WSL2 shell.

## Part A - Understand the project and data

### A1. Inspect the project structure

Locate these paths:

| Path | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, Python constraints, dependencies and tool configuration |
| `uv.lock` | Resolved package versions and distribution information |
| `src/churnlab/` | Importable data, training and serving code |
| `scripts/` | Executable commands for the workflow |
| `tests/` | Small behavior checks |
| `data/raw/` | Original input snapshot and source metadata |
| `data/processed/` | Generated splits in later TPs |
| `artifacts/` | Generated model files in later TPs |
| `reports/` | Generated reports |
| `k8s/` and `pipelines/` | Deployment and orchestration work in later TPs |

**Question A1:** Why should the original CSV, a fitted model and the training source code have different roles in the repository?

### A2. Understand IBM Telco Customer Churn

The course uses the CSV published in IBM's Telco churn example repository. The bundled snapshot contains **7,043 rows and 21 columns**.

| Field | Meaning in this exercise | Role |
|---|---|---|
| `customerID` | Customer identifier | Identification, not a model feature |
| `tenure` | Months the customer has stayed with the company | Numeric input |
| `MonthlyCharges` | Monthly charge | Numeric input |
| `TotalCharges` | Total charges | Numeric input requiring inspection |
| `Contract` | Contract category | Categorical input |
| `InternetService` | Internet service category | Categorical input |
| `PaymentMethod` | Payment method category | Categorical input |
| `Churn` | `Yes` or `No` churn label | Target |

The later course baseline uses these six input fields; the original dataset contains additional columns.

> This is a customer snapshot. It is not a dated customer event log or a randomized retention experiment. Do not invent an event history or treatment assignment and present it as observed IBM data.

**Question A2:** Why would including `Churn` among the model inputs invalidate the prediction task?

## Part B - Create and inspect the Python environment

### B1. Create the project environment

Run from the project root:

```sh
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"
```

Expected package version:

```text
0.1.0
```

`uv sync --locked` installs the selected project environment while requiring the existing lock file to remain consistent with the project definition. If the lock needs an update, investigate the mismatch instead of silently resolving a new environment.

Do not install the TensorFlow or Kubeflow extras yet. The starter project declares packages used throughout the course; installing them now does not mean you must understand or use all of them in TP01.

### B2. Record the environment

Create `scripts/record_environment.py`:

```python
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

record = {
    "python": sys.version,
    "platform": platform.platform(),
    "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
    "packages": {
        name: version(name)
        for name in ["churnlab", "pandas", "numpy", "scikit-learn"]
    },
}

out = Path("evidence/tp01/environment.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
```

Run:

```sh
uv run python scripts/record_environment.py
uv tree
```

**Tasks B2:**

1. Locate the Python constraint and one direct dependency in `pyproject.toml`.
2. Find the same dependency in `uv.lock`.
3. Use `uv tree` to identify one transitive dependency.
4. Explain the difference between `pyproject.toml`, `uv.lock` and `.venv/`.

### B3. Interpret a version constraint

This is a version-specifier example, not a command to run:

```text
numpy ~= 1.26.0
```

Its release range is:

```text
>=1.26.0, <1.27.0
```

Compare it with `numpy==1.26.4`. A range permits several candidate versions; a lock file records resolved selections. Version numbering alone does not universally guarantee compatibility.

## Part C - Inspect and fingerprint the Telco dataset

### C1. Obtain the source snapshot

The package includes `data/raw/telco.csv`. The download script fetches it if it is absent and records its source and hash:

```sh
uv run python scripts/download_data.py
```

Inspect:

- `data/raw/telco.csv`
- `data/raw/telco.source.json`

The course source is pinned to an IBM repository commit:

[Download the commit-pinned CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv).

Expected SHA-256 for the bundled bytes:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

> A filename is not a data version. Two files named `telco.csv` can contain different bytes. A hash identifies the bytes you used; it does not prove that the source or labels are trustworthy.

### C2. Create an executable data inspection script

Create `scripts/inspect_telco.py`:

```python
import hashlib
import json
from pathlib import Path

import pandas as pd

source = Path("data/raw/telco.csv")
expected_sha256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
if actual_sha256 != expected_sha256:
    raise ValueError("Source hash mismatch: investigate before continuing.")

df = pd.read_csv(source)
required = {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}
missing_columns = sorted(required - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

target_counts = df["Churn"].value_counts(dropna=False)
blank_total_charges = (
    df["TotalCharges"].astype("string").str.strip().eq("").sum()
)

report = {
    "source_file": source.as_posix(),
    "sha256": actual_sha256,
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "target_counts": {str(k): int(v) for k, v in target_counts.items()},
    "target_rates": {
        str(k): float(v / len(df)) for k, v in target_counts.items()
    },
    "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
    "blank_total_charges": int(blank_total_charges),
    "dtypes": {name: str(dtype) for name, dtype in df.dtypes.items()},
}

out = Path("reports/tp01_data_overview.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
```

Run:

```sh
uv run python scripts/inspect_telco.py
```

Expected observations for the unmodified snapshot:

| Observation | Expected value |
|---|---:|
| Rows | 7,043 |
| Columns | 21 |
| `Churn = No` | 5,174 |
| `Churn = Yes` | 1,869 |
| Churn proportion | Approximately 26.54% |
| Duplicate customer IDs | 0 |
| Whitespace-only `TotalCharges` values | 11 |

### C3. Interpret the result

Answer in your own words:

1. Why is churn prediction a classification task?
2. What does the class balance suggest about using accuracy alone later?
3. Why can `TotalCharges` be read as an object/string column even though it represents a numeric quantity?
4. Why may `isna()` alone fail to detect a whitespace-only value?
5. Which evidence would tell a teammate that they inspected the same source file?

Keep the original CSV unchanged. Full contracts, cleaning, splitting and missing-value handling belong to TP02. This initial check establishes what your project received.

## Part D - Prove environment recreation

### D1. Recreate rather than assume

1. Keep `pyproject.toml`, `uv.lock` and the raw CSV unchanged.
2. Save `evidence/tp01/environment.json` as `environment-before.json` in the same directory.
3. Close Python processes and notebooks using the project environment.
4. Rename `.venv/` to `.venv-backup/` using your file manager. If a backup already exists, choose another name.
5. Run:

```sh
uv sync --locked
uv run python scripts/record_environment.py
uv run python scripts/inspect_telco.py
```

6. Compare the before/after Python and package versions, dataset hash and inspection outputs.
7. After successful verification, remove the backup with your file manager.

Do not commit either environment directory.

### D2. Explain what you have reproduced

Complete this table in `evidence/tp01/answers.md`:

| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit | Uncommitted edits are not identified by that commit |
| Package selections | `pyproject.toml`, `uv.lock` | Platform/interpreter selection can affect installations |
| Installed environment | Before/after environment records | This does not capture every OS or hardware detail |
| Source data | Source URL and SHA-256 | This does not establish label quality or absence of leakage |
| Trained model | Not produced in TP01 | Requires later training/configuration/artifact evidence |

> Recreating package selections and identifying the input data are necessary steps. They do not guarantee identical results for every future ML training run.

## Part E - Collaborate through Git and a pull request

### E1. Check what belongs in Git

Review `.gitignore` before staging files. The starter project already excludes generated models, reports, environments and common local state.

Ensure these additional entries exist:

```gitignore
.venv-backup/
.venv-backup-*/
```

For this exercise:

| Commit | Keep outside Git |
|---|---|
| Python source and tests | `.venv/` and environment backups |
| `pyproject.toml` and `uv.lock` | Credentials and `.env` |
| README and non-sensitive evidence | Generated models and tracking stores |
| Source URL/hash metadata | Raw CSV and generated reports, following the starter policy |

`reports/` is ignored. Copy your small, non-sensitive TP01 report to `evidence/tp01/data-overview.json` using your file manager so it can be included with your submission. Do not override the ignore rule for the whole reports directory.

### E2. Initialize your personal repository

If you extracted the reference project from the archive and it has no Git repository yet:

```sh
git init
git status --short
git add .
git diff --cached --stat
git commit -m "Initialize Telco churn project"
git branch -M main
```

Inspect the staged files before committing. Create an **empty** GitHub repository without an automatically generated README, then replace the placeholder URL:

```sh
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

Do not type the angle-bracket placeholder literally.

If you cloned an existing course repository instead, inspect `git status` and `git remote -v`. Use the instructor's fork/collaboration route; do not initialize it again or push to a repository you do not own.

### E3. Propose a README improvement

Create a branch:

```sh
git switch -c docs/telco-onboarding
```

Update `README.md` with:

1. The churn business question and target column.
2. The project Python version and environment setup command.
3. The dataset source, hash and inspection command.
4. The purpose of at least two directories.
5. The distinction between this snapshot and an event-time or intervention dataset.

Verify the instructions, then commit:

```sh
git status
git diff
git add README.md
git commit -m "Document reproducible Telco onboarding"
git push -u origin docs/telco-onboarding
```

Open a pull request into `main`. Its description must explain the change and the verification you performed.

### E4. Review and merge

Your partner must review the instructions and ask one substantive question, for example:

- Can I reproduce the dataset check from a fresh environment?
- Which dataset bytes were used, and how can I verify them?
- Does the README explain what the lock file cannot reproduce?

Respond, improve the change if needed, and merge after review. Then run:

```sh
git switch main
git pull
git log --oneline --graph --decorate -8
```

Keep the PR URL and relevant commit hash in `evidence/tp01/answers.md`.

## Part F - Recover from a local mistake

Use a disposable branch and a clean working tree. Do not run a restore command over work you need to preserve.

```sh
git switch -c exercise/local-recovery
```

### F1. Discard one deliberate unstaged edit

Append a temporary sentence to `README.md`, save it, then run:

```sh
git diff -- README.md
git restore README.md
git diff -- README.md
```

The temporary edit disappears. `git restore README.md` discards unstaged changes in that file; it is not merely an unstage command.

### F2. Unstage without deleting the working edit

Append a different temporary sentence, save it, then run:

```sh
git add README.md
git diff --cached -- README.md
git restore --staged README.md
git status --short
git diff -- README.md
```

The edit remains in your working file but is no longer staged. Explain the difference between F1 and F2.

Discard only the deliberate temporary sentence before returning to `main`:

```sh
git restore README.md
git switch main
```

## 3. Submission

Submit your repository or an exported project with the following evidence:

| Deliverable | Expected content |
|---|---|
| Updated `README.md` | Setup, project purpose, source provenance and inspection instructions |
| `scripts/record_environment.py` | Executable environment record |
| `scripts/inspect_telco.py` | Executable source hash and data overview check |
| `evidence/tp01/environment-before.json` | Environment before recreation |
| `evidence/tp01/environment.json` | Environment after recreation |
| `evidence/tp01/data-overview.json` | Copy of the generated report |
| `evidence/tp01/answers.md` | Answers, PR URL, commit hash and remaining reproducibility limits |

Use a final focused commit for any evidence not already committed. Submit the reviewed PR even if you also submit an archive. Do not submit `.venv/`, passwords, tokens or generated model files.

### Acceptance checklist

- [ ] `uv sync --locked` succeeds without changing `uv.lock`.
- [ ] `import churnlab` succeeds in the recreated environment.
- [ ] The source hash matches the expected snapshot.
- [ ] The inspection report matches the expected counts.
- [ ] Original raw data were not modified.
- [ ] The README lets a teammate repeat setup and inspection.
- [ ] A reviewed PR includes verification evidence.
- [ ] You can distinguish restoring a file from unstaging it.
- [ ] Your submission explains what TP01 does not yet reproduce.

### Suggested assessment

| Criterion | Points |
|---|---:|
| Environment setup and recreation evidence | 3 |
| Dataset identification and executable inspection | 3 |
| Git workflow, review and recovery explanation | 2 |
| Clear README and explanation of limitations | 2 |
| **Total** | **10** |

## 4. Optional extensions

1. **Deliberate source mismatch:** copy the CSV to a temporary location, change one value, and adapt the inspection script to inspect that copy. Confirm that the hash check rejects it. Keep the original snapshot unchanged.
2. **Cross-machine comparison:** ask your partner to run the same commit and lock file. Compare package versions and source hashes. Explain platform differences rather than assuming every environment detail must be identical.
3. **Merge conflict:** create a deliberate README conflict with your partner and resolve it while preserving both intended improvements.

## 5. Troubleshooting

| Symptom | Check |
|---|---|
| `uv` is not found | Installation and terminal PATH; reopen the terminal if needed |
| `pyproject.toml` is not found | Run from the `churnlab-2026/` root |
| `--locked` fails | Project/lock consistency and supported Python version; keep the supplied files together |
| `ModuleNotFoundError: churnlab` | Run `uv sync --locked`, then use `uv run python` |
| CSV is missing | Run `scripts/download_data.py`; use the supplied snapshot if the network is blocked |
| SHA-256 mismatch | Check the source version and local edits; obtain the original snapshot rather than disabling the check |
| Inspection report is absent from Git | `reports/` is ignored; copy the small report to `evidence/tp01/` |
| No changes appear in a README PR | Make and save the README edit on the branch before staging it |
| `git remote add origin` reports an existing remote | Inspect `git remote -v`; follow the clone/fork route |
| Git requests an author identity | Configure your name/email using the instructor's repository policy |

## 6. References

- [IBM Telco churn source repository](https://github.com/IBM/telco-customer-churn-on-icp4d)
- [Commit-pinned dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv)
- [uv: Working on projects](https://docs.astral.sh/uv/guides/projects/)
- [Python packaging: Version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/)
- [Pro Git](https://git-scm.com/book/en/v2)
- [Git: restore](https://git-scm.com/docs/git-restore)
- [GitHub: Pull requests](https://docs.github.com/en/pull-requests)


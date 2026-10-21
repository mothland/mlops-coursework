# churnlab

Telco customer churn project for the ESILV MLOps course.

## Business question

A telecommunications company wants to know **which customers are likely to leave its service**,
so it can act before they do. This is a binary classification problem:

- **Target column:** `Churn` (`Yes` / `No`)
- **Identifier (not a feature):** `customerID`
- **Baseline inputs (later TPs):** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`,
  `InternetService`, `PaymentMethod`

## Environment setup

Requirements: Python **3.11** (3.12 also accepted, see `requires-python` in `pyproject.toml`),
Git and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"   # -> 0.1.0
uv run python scripts/record_environment.py                         # -> evidence/tp01/environment.json
```

`--locked` fails instead of silently re-resolving if `uv.lock` no longer matches `pyproject.toml`.
If Python 3.11 is missing, `uv python install 3.11` provides it.

## Data

The dataset is the IBM Telco Customer Churn CSV (7,043 rows, 21 columns), pinned to a commit of
IBM's repository:

- Source: <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv>
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

The raw CSV is **not committed**. Fetch and verify it, then run the inspection check:

```sh
uv run python scripts/download_data.py   # downloads data/raw/telco.csv if absent, verifies hash, writes telco.source.json
uv run python scripts/inspect_telco.py   # hash check + overview -> reports/tp01_data_overview.json
```

Expected: 7,043 rows, 21 columns, `Churn` = 5,174 No / 1,869 Yes (~26.5% churn),
0 duplicate `customerID`, 11 whitespace-only `TotalCharges`. Do not edit `data/raw/telco.csv`;
any change breaks the hash check by design.

### What this dataset is not

This is a **single customer snapshot**: one row per customer with their state at one point in time.
It is **not** a dated event log (no timestamps of when customers churned or changed plans) and
**not** an intervention/experiment dataset (no randomized retention offers). Models trained on it
predict association with churn, not the effect of an action, and we must not invent event times or
treatment assignments and present them as IBM data.

## Project layout

| Path | Purpose |
|---|---|
| `pyproject.toml` / `uv.lock` | Declared dependencies / exact resolved versions |
| `src/churnlab/` | Importable project code (data, training, serving) |
| `scripts/` | Executable workflow commands (download, inspect, record environment) |
| `tests/` | Small behavior checks (`uv run pytest`) |
| `data/raw/` | Original input snapshot (ignored) and its source metadata (committed) |
| `data/processed/` | Generated splits (later TPs, ignored) |
| `artifacts/` | Generated model files (later TPs, ignored) |
| `reports/` | Generated reports (ignored; copy small evidence to `evidence/`) |
| `evidence/` | Non-sensitive TP evidence committed for submission |
| `k8s/`, `pipelines/` | Deployment and orchestration (later TPs) |

## What is not reproduced (yet)

`uv.lock` pins the exact package versions and their file hashes. It does **not** reproduce:

- **The interpreter, OS and hardware:** only the Python version range is constrained; the exact
  Python build, OS, CPU and numerical libraries (BLAS) can differ between machines.
- **The data:** uv does not track `data/raw/telco.csv`; the SHA-256 check in
  `scripts/inspect_telco.py` covers that.
- **Uncommitted code:** a commit only identifies what was committed, not local edits.
- **Training results:** random seeds, data splits, hyper-parameters and configuration are not
  recorded yet, so future training runs are not guaranteed to give identical models or metrics.
- **Data quality:** a matching hash proves we used the same bytes, not that the labels are correct
  or that the features are free of leakage.

# churnlab

Telco customer churn project for the ESILV MLOps course.

## Business question

A telecommunications company wants to know **which customers are likely to leave its service**,
so it can act before they do. This is a binary classification problem:

- **Target column:** `Churn` (`Yes` / `No`)
- **Identifier (not a feature):** `customerID`
- **Baseline inputs (later TPs):** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`,
  `InternetService`, `PaymentMethod`

## Quick start (first time? start here)

This takes about 5 minutes. You type every command below in a **terminal**: on Windows open
"PowerShell" from the Start menu, on macOS open "Terminal". You don't need to install Python yourself,
uv does it for you.

### 1. Install the two tools you need

- **Git** (downloads the code): <https://git-scm.com/downloads>, default options are fine.
- **uv** (installs Python and the project's libraries):

  ```sh
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

**Close and reopen the terminal**, then check that both work (each one should print a version number):

```sh
git --version
uv --version
```

### 2. Get the project and install it

```sh
git clone https://github.com/mothland/mlops-coursework.git
cd mlops-coursework
uv sync --locked
```

`uv sync` creates a private Python environment in a `.venv/` folder inside the project, with the exact
library versions the project was tested with. You never need to "activate" it: just start every
command with `uv run`, like below.

> All the following commands must be run **from the project folder** (the one containing `README.md`).
> If you closed the terminal, `cd` back into it first.

### 3. Download the data

```sh
uv run python scripts/download_data.py
```

This downloads the IBM Telco dataset to `data/raw/telco.csv` and checks it's the exact expected file.

### 4. Train the models

```sh
uv run python scripts/train.py
```

It trains three models plus a "dumb" baseline, compares them, and saves the best one. It takes a few
seconds and should end with (slashes may be `/` on macOS/Linux):

```text
selected: random_forest
saved artifacts\tp02\pipeline.joblib and artifacts\tp02\metadata.json
```

### 5. Predict churn for your own customers

Create a CSV file (e.g. `my_customers.csv` in the project folder, with Excel or any text editor) with
these columns, one customer per line:

```csv
customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,PaymentMethod
NEW-0001,1,85.5,85.5,Month-to-month,Fiber optic,Electronic check
NEW-0002,60,20.0,1200.0,Two year,No,Bank transfer (automatic)
```

| Column | What to put |
|---|---|
| `customerID` | Any name/ID, only used to label the output |
| `tenure` | Number of months the customer has been with the company |
| `MonthlyCharges`, `TotalCharges` | Amounts billed per month / in total (a dot for decimals); can be left blank |
| `Contract` | `Month-to-month`, `One year` or `Two year` |
| `InternetService` | `DSL`, `Fiber optic` or `No` |
| `PaymentMethod` | `Electronic check`, `Mailed check`, `Bank transfer (automatic)` or `Credit card (automatic)` |

Then run:

```sh
uv run python scripts/predict.py --input my_customers.csv --output my_predictions.csv
```

```text
customerID  churn_probability  churn_predicted
  NEW-0001           0.845814                1
  NEW-0002           0.007439                0
```

`churn_probability` is the model's churn score (higher = more likely to leave; best used to **rank**
customers, it is not a calibrated percentage). `churn_predicted` is `1` when the score is at least 0.5.
The same table is saved in `my_predictions.csv`.

### 6. (Optional) Check that everything works

```sh
uv run python scripts/check_reload.py   # every saved model gives the same predictions after reloading
uv run python scripts/run_tests.py      # should end with "12 passed"
```

### If something goes wrong

| Message | Fix |
|---|---|
| `uv` / `git` is not recognized, or `command not found` | Close and reopen the terminal after installing. Still failing? Reinstall the tool. |
| `No such file or directory: 'data/raw/telco.csv'` | Run step 3 first, from the project folder. |
| `Source hash mismatch` | The data file is damaged or modified: delete `data/raw/telco.csv` and run step 3 again. |
| `No such file or directory: 'artifacts/tp02/...'` | Run step 4 first (the model isn't committed, it's created by training). |
| `Missing required feature columns: [...]` | Your CSV is missing a column, or a column name is misspelled (they are case-sensitive). |
| `Malformed numeric values in ...` | A number column contains text (e.g. `85.5€` or `none` instead of `85.5` or a blank). |
| `ModuleNotFoundError` | You typed `python ...` instead of `uv run python ...`. |
| `can't open file ... scripts/...` | You're not in the project folder: `cd` into it. |

The rest of this README is the detailed technical documentation.

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

## Training pipeline (TP02)

Prerequisite: `uv sync --locked` and `data/raw/telco.csv` downloaded (see above).

```sh
uv run python scripts/first_model.py   # milestone 1: tenure + MonthlyCharges vs majority baseline -> reports/tp02/first_model.json
uv run python scripts/train.py         # compare candidates, save selected pipeline -> artifacts/tp02/
uv run python scripts/check_reload.py  # reload every saved model in its own fresh process, compare predictions -> reports/tp02/reload_check.json
uv run python scripts/run_tests.py     # focused reliability checks (pytest) -> reports/tp02/pytest.log with project-relative paths
```

Everything `train.py` does is driven by `configs/tp02.json` (seed, split sizes, threshold, selection
rule, candidates and their parameters). Outputs in `artifacts/tp02/`:

| File | Content |
|---|---|
| `pipeline.joblib` | Selected **fitted** preprocessing + classifier (one sklearn `Pipeline`) |
| `candidates/<name>/` | Every compared model's fitted pipeline + its reference predictions |
| `metadata.json` | Source/config/artifact hashes, seed + allocation hash, feature roles, preprocessing, fitted statistics, model params, metrics, threshold, git commit, environment |
| `comparison.csv` | Validation metrics, configuration and fit time for every candidate + baseline |
| `split_allocation.csv` | `customerID` -> `train` / `valid` / `test` |
| `inference_sample.csv`, `reference_predictions.csv` | 20 validation customers (features only, no `Churn`) and the selected model's predictions for them, made by the training process |

`artifacts/` is ignored by Git; small copies of the evidence (including the raw logs and the per-model reload reports) are in `evidence/tp02/`.

To score your own batch with one artifact: `uv run python scripts/predict.py --input customers.csv --output predictions.csv` (add `--artifact artifacts/tp02/candidates/<name>/pipeline.joblib` for another model).

The artifact can be regenerated exactly by re-running `train.py` with the same data, config and environment.

### Task and split

- Positive class: `Churn = Yes` -> `1`. `customerID` is only used to identify customers and check splits.
- Stratified 60/20/20 train/validation/test split with seed 42 (4,225 / 1,409 / 1,409 customers).
- Candidates are compared on **validation only**. The **test set is reserved** and never evaluated in TP02.

### Input contract (for inference)

The pipeline consumes a DataFrame with these six columns (extra columns are dropped by the
`ColumnTransformer`; a missing column is an error):

| Column | Type | Missing value policy |
|---|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | float | NaN / blank / whitespace-only -> imputed with the **training median**, then standardized |
| `Contract`, `InternetService`, `PaymentMethod` | string | blank -> imputed with the **training mode**, then one-hot encoded |

Parsing is done **before** the pipeline by `churnlab.data.prepare_features`, with the same stateless
rules at training and inference: blank/whitespace-only numerics become NaN, but any other non-numeric
text (e.g. `"abc"`) raises an error instead of quietly becoming missing. A category never seen in
training is encoded as all zeros (`handle_unknown="ignore"`): the call succeeds, but that prediction
is not validated.

### Model selection

The selection rule was fixed in `configs/tp02.json` before looking at results: highest validation
**average precision** (computed from the positive-class probability); candidates within 0.01 of the
best count as tied, and ties go to the simpler model (logistic regression < decision tree < random
forest). Precision, recall, F1 (threshold 0.5), accuracy and the confusion matrix are reported too.
See `evidence/tp02/answers.md` for the table and the decision.

### Loading the artifact

```python
import joblib
from churnlab.data import prepare_features
pipeline = joblib.load("artifacts/tp02/pipeline.joblib")   # trusted files only: loading can run code
proba = pipeline.predict_proba(prepare_features(customers_df))[:, 1]   # P(Churn = Yes)
```

Load with the versions recorded in `metadata.json` (`uv sync --locked` gives scikit-learn 1.5.2);
pickled sklearn objects are not guaranteed to load in other versions.

### Limitations

- Average precision measures ranking quality; predicted probabilities are not calibrated, so 0.8 is not
  an 80% churn risk.
- The 0.5 threshold is a default, not a business decision.
- The 11 blank `TotalCharges` belong to `tenure = 0` customers; the median fill (~1,374) is a simple
  documented choice, not their real value (likely 0).
- One split, one seed: metric differences of about 0.01 are within noise.
- `metadata.json` records the git commit **and** whether the working tree was dirty; a dirty run is
  not exactly identified by the commit.

## Project layout

| Path | Purpose |
|---|---|
| `pyproject.toml` / `uv.lock` | Declared dependencies / exact resolved versions |
| `src/churnlab/` | Importable code: `data` (load/parse), `split`, `pipeline` (unfitted builders), `evaluate`, `train` |
| `scripts/` | Executable workflow commands (download, inspect, record environment, first model, train, predict, reload check, tests) |
| `configs/` | Training configuration (`tp02.json`) |
| `tests/` | Small behavior checks (`uv run pytest`, or `scripts/run_tests.py` to also save the log) |
| `data/raw/` | Original input snapshot (ignored) and its source metadata (committed) |
| `data/processed/` | Generated splits (later TPs, ignored) |
| `artifacts/` | Generated model files and run records (ignored) |
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
- **Training results:** since TP02, the seed, split allocation, configuration and environment are
  recorded in `artifacts/tp02/metadata.json`, but results can still differ across platforms/BLAS builds.
- **Data quality:** a matching hash proves we used the same bytes, not that the labels are correct
  or that the features are free of leakage.

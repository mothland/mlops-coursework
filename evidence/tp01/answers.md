# TP01 - Answers

## Part A

**A1.** Why should the original CSV, a fitted model and the training source code have different roles in the repository?

Because they have different roles in production, are updated at different rates, and have different "targets". The original CSV is a (potentially) massive immutable dataset in prod. Using the source URL plus the SHA-256 hash as metadata remains much more convenient than storing Gigabytes of data on GitHub for no real gain. Said dataset could also contain sensitive information in certain cases. All good reasons to just, not push it.

The training source code is something we review and edit frequently, it is what devs work on. This means that tight versioning is crucial, so we can review code, work on different branches, etc.

And the fitted model itself is the *output* of the source code run on the data (with a given environment and configuration), so it can be regenerated from those, and it can be a pretty large binary file. It shouldn't live on a Git repo: but CD should be able to send said binary to an artifact repository.


**A2.** Why would including `Churn` among the model inputs invalidate the prediction task?

Because that would be pretty terrible data leakage. `Churn` is the target. Predicting the output using the output is a pointless task that cannot work in real life: when you make a prediction, the customer hasn't left yet, so Churn doesn't exist as an input. The model would also look perfect during testing, since it just copies the answer, and so that broken metric becomes dangerous.

## Part B

**B2.1.** Locate the Python constraint and one direct dependency in `pyproject.toml`.

- Python constraint: `requires-python = ">=3.11,<3.13"`
- Direct dependency: `"pandas>=2.2,<2.3"` (in `[project] dependencies`)

**B2.2.** Find the same dependency in `uv.lock`.

```toml
[[package]]
name = "pandas"
version = "2.2.3"
source = { registry = "https://pypi.org/simple" }
```

The range `>=2.2,<2.3` from `pyproject.toml` is resolved to exactly `2.2.3`, along with the source registry and the hashes of each downloadable file (sdist/wheels).

**B2.3.** Use `uv tree` to identify one transitive dependency.

`scipy v1.17.1`: it is not declared in `pyproject.toml`, it is pulled in by `scikit-learn`:

```text
├── scikit-learn v1.5.2
│   ├── joblib v1.6.0
│   ├── numpy v1.26.4
│   ├── scipy v1.17.1
│   └── threadpoolctl v3.7.0
```


**B2.4.** Explain the difference between `pyproject.toml`, `uv.lock` and `.venv/`.

`pyproject.toml` is a list of needed direct dependencies, as version ranges, plus the required Python version. `uv.lock` records the exact versions uv resolved from the toml file's constraints, including transitive dependencies. `.venv` is your virtual environment, containing the project's Python interpreter and installed packages, isolated from your global machine's Python instance. The toml and lock files are committed, while `.venv` stays out of Git and is rebuilt with `uv sync --locked`, for clean reproducibility.

## Part C

**C3.1.** Why is churn prediction a classification task?

Because we're predicting categorical discrete values (here, a binary Yes/No), not a continuous number like `MonthlyCharges` (which would be regression).

**C3.2.** What does the class balance suggest about using accuracy alone later?

As a reminder:

| `Churn` | Count | Share |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |
| **Total** | **7,043** | 100% |

The classes are pretty imbalanced (~3:1, with 73% No). A model that always predicts "No" would already reach ~73.5% accuracy while catching 0 churners, which is exactly the customers we want to find.

We only have 1.8K-ish rows with Churn = Yes, which means that simply going for undersampling could potentially leave us with not enough rows for decent training (depending on how complex the prediction would be). Class weighting might be a good first try.

Either way, it means that by default, accuracy alone won't mean much and recall or F1 score (as well as a good look at the confusion matrix) will be more critical than in a balanced dataset.

**C3.3.** Why can `TotalCharges` be read as an object/string column even though it represents a numeric quantity?

CSV files don't have types, and whitespace is string. If even one value isn't a number, then pandas will assume the whole column is string (`object`), even when all you have beyond numbers is whitespace. And we have 11 of these: rows where `TotalCharges` is `" "` (customers with `tenure = 0`, not billed yet).

**C3.4.** Why may `isna()` alone fail to detect a whitespace-only value?

Because `isna()` only detects real missing values (`NaN`/`None`). pandas turns *empty* CSV fields into `NaN`, but `" "` is not empty: it's a present, non-null string, so `isna()` returns `False`. You need to strip the values in code first (e.g. `.str.strip().eq("")`, as `inspect_telco.py` does, or `pd.to_numeric(..., errors="coerce")`), without modifying the raw CSV file itself.

**C3.5.** Which evidence would tell a teammate that they inspected the same source file?

The SHA-256 hash (`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`): same hash means byte-for-byte the same file. It is checked by `inspect_telco.py` and recorded with the commit-pinned source URL in `data/raw/telco.source.json`. The URL tells where the file came from, but only the hash proves the bytes are the same (the filename or matching row counts don't).

## Part D

**D1.** Compare the before/after Python and package versions, dataset hash and inspection outputs.

`.venv/` was renamed to `.venv-backup/`, then `uv sync --locked`, `record_environment.py` and `inspect_telco.py` were re-run.

| | Before | After |
|---|---|---|
| Python | 3.11.15 | 3.11.15 |
| churnlab / pandas / numpy / scikit-learn | 0.1.0 / 2.2.3 / 1.26.4 / 1.5.2 | 0.1.0 / 2.2.3 / 1.26.4 / 1.5.2 |
| Dataset SHA-256 | `16320c9c...5e91` | `16320c9c...5e91` |
| Inspection report | 7,043 rows, 21 cols, 5,174 No / 1,869 Yes, 0 dup IDs, 11 blank `TotalCharges` | identical |

`environment-before.json` and `environment.json` are identical, the data report is identical, and `uv.lock` was not modified by the sync. The backup was then deleted.

**D2.** What has been reproduced

| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit | Uncommitted edits are not identified by that commit |
| Package selections | `pyproject.toml`, `uv.lock` | Platform/interpreter selection can affect installations |
| Installed environment | Before/after environment records | This does not capture every OS or hardware detail |
| Source data | Source URL and SHA-256 | This does not establish label quality or absence of leakage |
| Trained model | Not produced in TP01 | Requires later training/configuration/artifact evidence |

## Part E

- PR URL: https://github.com/mothland/mlops-coursework/pull/2
- Reviewer question (@latomate07): "Does the README explain what the lock file cannot reproduce?"
- Response: it did not. Added a "What is not reproduced (yet)" section to the README (interpreter/OS/hardware, data, uncommitted code, training results, data quality), then merged.
- Relevant commit hashes:
  - `d362860` Document reproducible Telco onboarding
  - `5614a63` Document what the lock file cannot reproduce (review fix)
  - `9014a28` Merge pull request #2 (merge into `main`)

## Part F

**F.** Explain the difference between F1 (`git restore README.md`) and F2 (`git restore --staged README.md`).

- **F1** `git restore README.md` overwrites the working file with the last staged/committed version: the unstaged edit is **lost** (`git diff` shows nothing afterwards).
- **F2** `git restore --staged README.md` only removes the change from the staging area (index): the edit is **kept** in the working file and shows up again in `git diff` as an unstaged change.

So `git restore` discards working changes (destructive), while `git restore --staged` just unstages (safe).

# TP02 - Answers

## 1. Get a first model working

**1.0.** Your checkpoint: a first learned model runs from data to validation predictions. Record at least accuracy and a confusion matrix, with the positive class and matrix labels stated.

- Split: stratified on `Churn`, seed 42. Train 4,225 / validation 1,409 / test 1,409.
- Churn rate: train 26.53%, validation 26.54%, test 26.54%.
- Positive class: `Churn = Yes` -> 1. Features: `tenure`, `MonthlyCharges`.
- Confusion matrices: rows = actual [No, Yes], columns = predicted [No, Yes], on validation (`first_model.json`).

| Model | Accuracy | Confusion matrix | Predicted churners |
|---|---:|---|---:|
| Majority baseline (`DummyClassifier(most_frequent)`) | 0.7346 | `[[1035, 0], [374, 0]]` | 0 |
| Logistic regression (`max_iter=1000`) | 0.7828 | `[[950, 85], [221, 153]]` | 238 |

So the first logistic regression does a bit better than the baseline in accuracy (78.3% vs 73.5%), and more importantly it actually finds churners: 153 of them. It still misses 221 out of 374 though, but with only two features that's not shocking.


**1.1.** What would a system that always predicts “No churn” achieve?

It would achieve that:

| Metric (validation) | Majority baseline |
|---|---:|
| Accuracy | 0.7346 |
| Churners caught | 0 / 374 |
| Average precision | 0.2654 |
| Precision / recall / F1 | 0 / 0 / 0 |

Which matches the amount of no churn rows: so the high accuracy is really misleading here.


**1.2.** Can high accuracy coexist with failing to identify any churners?

Yes, as seen in 1.1

**1.3.** Which data are permitted to influence model selection?

Only train and validation. Train is for fitting, validation is for comparing candidates and picking one. The test set stays locked away until the very end: if it influences any choice, it stops being an honest estimate of how the model does on unseen customers, because we basically tuned on it.


**1.4.** Which dataset should the baseline be fitted on?

The training set, like every other model. The `DummyClassifier` does learn something (which class is the majority), so fitting it on validation would be leaking, and it wouldn't be a fair comparison anymore. Here it doesn't change much (the churn rate is 26.5% everywhere thanks to the stratified split), but it's the principle.


## 2. Connect preprocessing and the model

**2.0.** Your checkpoint: one call to fit the complete pipeline trains the preprocessing and classifier; prediction accepts original, unencoded feature columns in the documented input types. You do not separately refit an encoder or scaler for validation.

The whole thing is one sklearn `Pipeline` (`src/churnlab/pipeline.py`): a `ColumnTransformer` with two branches, then the classifier.

- Numeric (`tenure`, `MonthlyCharges`, `TotalCharges`): `SimpleImputer(strategy="median")` then `StandardScaler`.
- Categorical (`Contract`, `InternetService`, `PaymentMethod`): `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Anything else (`customerID`, `Churn`...) is dropped by the `ColumnTransformer` (`remainder="drop"`), so it can't sneak into the model.

One `pipe.fit(X_train, y_train)` fits the imputers, scaler, encoder and classifier, all on train. Validation and new customers just go through `predict`, nothing gets refitted.

Parsing happens before the pipeline (`src/churnlab/data.py`), with the same rules at training and inference: the CSV is read as text, blank/whitespace-only numerics become NaN (that's the 11 `" "` in `TotalCharges`), and anything else that isn't a number raises an error. `to_numeric(errors="raise")` would crash on the `" "` values too, and `errors="coerce"` would silently turn a real typo like `"abc"` into NaN, which is exactly what we don't want. So: strip blanks first, coerce, and raise if a non-blank value still failed.

Unknown category at inference: the encoder puts all zeros for that column instead of crashing. The prediction still comes out, but nobody checked it's any good for that customer.


**2.1.** What do `fit`, `transform` and `predict` each do?

- `fit` learns from the data: medians for the imputer, mean/std for the scaler, list of categories for the encoder, weights/trees for the model.
- `transform` applies what was learned, without changing it.
- `predict` runs the input through the fitted transforms, then the fitted model, and gives the output.


**2.2.** What goes wrong if you fit a scaler on validation data?

That's leakage: validation statistics end up inside the preprocessing, so the validation score is no longer an honest "never seen this data" score. And if you refit it separately for validation, the same customer gets scaled differently than during training, so the model receives inputs that don't mean the same thing anymore.


**2.3.** Does using a Pipeline automatically prevent leakage if you give it the wrong training data?

Nope. The Pipeline only guarantees that everything is fitted on whatever you give to `fit`. If you give it the full dataset, or validation, it'll happily leak everything just the same. It's a tool, not a guard: the split still has to be done right before.


**2.4.** Which operations are ordered by `Pipeline`, and which columns are routed by `ColumnTransformer`?

`Pipeline` orders the steps: preprocess first, then classify (and inside each branch, impute first, then scale or encode). `ColumnTransformer` routes the columns: the three numeric ones go to the numeric branch, the three categorical ones to the categorical branch, and everything else is dropped.


## 3. Try different models fairly

**3.0.** Your checkpoint: a machine-readable comparison table and a short selection decision. Do not claim that a change from the initial two-feature model proves an algorithm improvement: the feature set also changed.

As a reminder:

Validation set, threshold 0.5, same split and six features (`comparison.csv`). Rule in `configs/tp02.json`:
best average precision, within 0.01 = tie, ties go to the simpler model (logistic < tree < forest).

| Model | Config | AP | Precision | Recall | F1 | Accuracy | TN / FP / FN / TP | Fit (s) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Baseline | `strategy=most_frequent` | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 | 1035 / 0 / 374 / 0 | 0.01 |
| Logistic regression | `max_iter=1000` | 0.6475 | 0.6644 | 0.5294 | 0.5893 | 0.8041 | 935 / 100 / 176 / 198 | 0.02 |
| Decision tree | `max_depth=5, min_samples_leaf=20` | 0.6173 | 0.6356 | 0.6016 | 0.6181 | 0.8027 | 906 / 129 / 149 / 225 | 0.01 |
| Random forest | `n_estimators=300, max_depth=8, min_samples_leaf=10` | 0.6642 | 0.6534 | 0.4840 | 0.5561 | 0.7949 | 939 / 96 / 193 / 181 | 0.27 |

- Selected by the rule: `random_forest` (AP gap to logistic regression: 0.0167).
- Test set: not evaluated (`metadata.json` -> `"test_evaluated": false`).

I picked **average precision** as the main metric before running anything, because the classes are imbalanced (accuracy just rewards saying "No") and the team needs to rank customers by churn risk to know who to contact first. AP is computed from the positive class probabilities, so it doesn't depend on the 0.5 threshold. The tie rule is in the config, so I couldn't cheat after seeing the numbers.

**Decision: random forest.** It has the best AP (0.664), 0.017 above logistic regression, so outside the 0.01 tie margin. All three learned models are way above the baseline (0.265, which is just the churn rate).

A few honest caveats though: the gap is small and it's from a single split. At the 0.5 threshold the forest has the *lowest* recall of the three, so if the team wants to catch more churners, the move is lowering the threshold (that's a business decision), not swapping models. And comparing with the first two-feature model doesn't prove anything about algorithms, since the features changed too.


**3.1.** Would changing the split for every classifier make the comparison easier to interpret?

No, harder. If every model gets different validation customers, you can't tell if a better score comes from the model or from getting an easier set of customers. Same split, same features, same metrics: then the only thing that changes is the model.


**3.2.** Does the model with the highest accuracy necessarily meet the team's needs?

As a reminder:

| Model | Accuracy | AP | Recall | Churners caught (of 374) |
|---|---:|---:|---:|---:|
| Logistic regression | **0.8041** | 0.6475 | 0.5294 | 198 |
| Decision tree | 0.8027 | 0.6173 | **0.6016** | **225** |
| Random forest | 0.7949 | **0.6642** | 0.4840 | 181 |

Not necessarily, and our table shows it pretty well: logistic regression has the best accuracy, but the forest ranks churners better and the tree catches the most churners at 0.5. Even the baseline gets 73.5% accuracy while being useless. What the team needs is finding the people about to leave, so recall/precision and AP matter more than accuracy here.


**3.3.** If two candidates are close, what would justify choosing the simpler one?

Simpler model = easier to explain (to the telecom team too), faster to train and to run, fewer hyperparameters to babysit, and less risk of overfitting. And on a single split, a tiny metric gap is probably just noise anyway. That's why my selection rule gives ties to the simpler model.


**3.4.** Is a predicted probability of 0.8 automatically a well-calibrated 80% risk?

No. It's a score between 0 and 1, not a promise. For a random forest it's basically the share of trees voting "churn", which isn't calibrated by default. To claim 80% really means 80%, you'd need to check it (calibration curve: do customers scored ~0.8 actually churn ~80% of the time?) and maybe add a calibration step. We didn't, so it's only good for ranking.


## 4. Refactor for reuse, then save the whole fitted pipeline

**4.0.** Your checkpoint: loaded and original predictions agree within a stated tolerance on the same examples. The artifact consumes the documented feature schema; CSV parsing and input-type validation remain explicit boundary responsibilities.

As a reminder:

- Saved: `artifacts/tp02/pipeline.joblib` (selected, fitted preprocessing + random forest), plus every
  candidate in `artifacts/tp02/candidates/<name>/`.
- `metadata.json` contains: artifact/source/config SHA-256, seed, split counts + allocation hash, feature
  roles and input types, preprocessing settings, fitted medians/means, model params, validation metrics,
  positive class, threshold 0.5, git commit + dirty flag, Python/package versions.
- Reload check (`reload_check.json`): 20 validation customers, each artifact loaded in its own fresh
  process, no fit, no labels. Tolerance 1e-12.

| Artifact | Max abs. probability difference | Class predictions identical | Passed |
|---|---:|:---:|:---:|
| Selected (random forest) | 1.11e-16 | yes | yes |
| Baseline | 0 | yes | yes |
| Logistic regression | 1.11e-16 | yes | yes |
| Decision tree | 1.11e-16 | yes | yes |
| Random forest | 1.11e-16 | yes | yes |

The saved file is the whole fitted pipeline (preprocessing + random forest), not just the model. Reloaded in a separate Python process with no `fit` and no labels, it gives the same predictions as before saving: max difference 1.11e-16, which is just floating point rounding, way below the 1e-12 tolerance. Did the same for all the other candidates, all identical.

The artifact expects the six documented columns. Reading the CSV and checking the types is done before it, in `prepare_features` (same function at training and inference), so a broken input gets caught at the door instead of inside the model.


**4.1.** Why is saving only the classifier insufficient?

Because the classifier only understands already scaled and one-hot encoded inputs. Without the fitted preprocessing, whoever loads it would have to rebuild it by hand, with the exact same medians, means and category order. Get one detail wrong and the predictions are silently garbage, no error message.


**4.2.** Which fitted information lives inside an imputer or encoder?

- The imputer stores `statistics_`: the training medians (numeric) or most frequent values (categorical) used to fill the gaps.
- The encoder stores `categories_`: the categories seen in training for each column, in order, which decides the output columns.
- (And the scaler stores `mean_` and `scale_`.)


**4.3.** Could two files called `model.joblib` contain different models?

Yes, a filename means nothing. Different data, seed, config or code would all give a different `model.joblib`. That's why `metadata.json` records the SHA-256 of the artifact next to the data hash, config hash, seed and git commit: same hash = same bytes, like for the CSV in TP01.


**4.4.** How does a sklearn Pipeline differ from a workflow DAG such as Kubeflow Pipelines?

A sklearn Pipeline is a single Python object: a chain of transforms + a model, fitted and used in one process. Kubeflow Pipelines is a workflow of separate steps (load data, train, evaluate, deploy...), each running in its own container, with inputs/outputs passed between them, scheduling, retries, etc. Our sklearn Pipeline could literally be what one of those steps trains and saves.


## 5. Demonstrate reliability with a few focused checks

Provide evidence that:

**5.1.** Customer IDs do not overlap across train, validation and test.

As a reminder:

- `split_allocation.csv`: 7,043 rows, every `customerID` unique, one split each.
- 4,225 + 1,409 + 1,409 = 7,043.
- Tests: `test_customer_ids_are_disjoint_and_complete`, `test_split_is_reproducible` (passed).

Every customer ends up in exactly one split, and running the split again with seed 42 gives the exact same allocation.


**5.2.** A learned imputation/scaling statistic matches training data, rather than the full dataset.

As a reminder:

| Column | Imputer median (fitted) | Training median | Full-dataset median | Scaler mean (fitted) |
|---|---:|---:|---:|---:|
| `tenure` | 29.0 | 29.0 | 29.0 | 32.443 |
| `MonthlyCharges` | 70.55 | 70.55 | 70.35 | 64.895 |
| `TotalCharges` | 1373.625 | 1373.625 | 1397.475 | 2297.765 |

- Blank `TotalCharges`: 7 in train, 11 in the full dataset.
- Tests: `test_imputer_medians_come_from_training_only`, `test_scaler_mean_comes_from_training_only` (passed).

The fitted medians match the training medians, not the full dataset ones (`MonthlyCharges` 70.55 vs 70.35, `TotalCharges` 1373.625 vs 1397.475), so the test can actually tell the difference.


**5.3.** A prediction call does not refit or change that statistic.

As a reminder:

- Test: `test_predict_does_not_refit` (passed): imputer `statistics_` identical before and after predicting validation + test rows.

Predicting just reads the learned values, it doesn't touch them.


**5.4.** A permitted missing numeric input and a previously unseen category follow your documented policy. Handling unknown categories does not prove predictions are reliable for them.

As a reminder:

- `test_blank_numeric_is_missing_but_malformed_text_fails` (passed): `" "` / `""` / `None` -> NaN; `"abc"` -> `ValueError`.
- `test_missing_numeric_uses_training_median` (passed): blank `TotalCharges` gives the same prediction as filling 1373.625.
- `test_unseen_category_is_encoded_as_zeros` (passed): `Contract = "Five year"` -> all `Contract` one-hot columns = 0, prediction returned.

A blank number gets the training median, as documented, and a real typo crashes instead of becoming a fake "missing". The unknown category doesn't crash either, but like the sheet says: getting a prediction doesn't mean it's a reliable one, the model has never seen a five-year contract.


**5.5.** A missing required feature produces a visible failure, rather than a silent incorrect prediction.

As a reminder:

- `test_missing_required_feature_fails_visibly` (passed): dropping `Contract` -> `ValueError: Missing required feature columns: ['Contract']`; the fitted pipeline alone also raises.

So a missing column is a loud error, not a quiet wrong prediction.


**5.6.** Saving/reloading preserves predictions; a separate inference process does not need training labels or training execution.

As a reminder:

- See the reload table in 4.0 (5 artifacts, 5 separate processes, all passed).
- `scripts/predict.py` rejects input containing `Churn` and never calls `fit`.
- Test: `test_reload_in_fresh_process_matches` (passed).

The inference process only needs the artifact and the six features: no labels, no training data, no `fit`.


## Peer review

Exchange projects with a peer. Ask them to reproduce training or load a trusted artifact using only your README, then record one issue and its resolution.



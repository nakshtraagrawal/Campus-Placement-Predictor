# 01 — Project Spec: Data, Preprocessing, Features, Models, Evaluation, Feature Influence

## 1. Problem & objectives
- **Task:** binary classification. Target `Placed` (1 = Placed, 0 = Not Placed).
- **Objectives**
  1. Build a clean, reproducible pipeline from raw data to a deployable model.
  2. Train and compare **≥ 4 models** (guidelines require ≥ 3).
  3. **Identify the most influential features** using several independent methods and check they agree.
  4. Provide an interactive web app for single-student prediction, explanation and "what-if" advice.

### Core features (the 8 in the problem statement)
| Final column name | Type | Range | Source |
|---|---|---|---|
| `CGPA` | float | 0–10 | Kaggle `CGPA` |
| `ProgrammingSkills` | int (rating) | 1–10 | **Simulated** (see §2.2) |
| `Internships` | int | 0–3 | Kaggle `Internships` |
| `Certifications` | int | 0–5 | Kaggle `Workshops/Certifications` |
| `CommunicationSkills` | float (rating) | 1–5 | Kaggle `SoftSkillsRating` |
| `AptitudeScore` | int | 0–100 | Kaggle `AptitudeTestScore` |
| `Projects` | int | 0–5 | Kaggle `Projects` |
| `Attendance` | float (%) | 0–100 | **Simulated** (see §2.2) |

Extra Kaggle columns (`ExtracurricularActivities`, `PlacementTraining`, `SSC_Marks`, `HSC_Marks`) are kept in the
cleaned data and used **only in an ablation experiment** (§7.4). The app and the main model use the 8 core features.
Drop `StudentID`.

## 2. Dataset

### 2.1 Base dataset — download (`src/get_data.py`)
"College Placement / Placement Prediction Dataset" (`placementdata.csv`, ~10,000 rows, columns: StudentID, CGPA,
Internships, Projects, Workshops/Certifications, AptitudeTestScore, SoftSkillsRating, ExtracurricularActivities,
PlacementTraining, SSC_Marks, HSC_Marks, PlacementStatus).

Try in this order, stop at the first that works:
1. If `data/raw/placementdata.csv` already exists → use it.
2. Hugging Face mirror (no login needed): repo `Krooz/Campus_Recruitment_CSV` (repo_type="dataset").
   Use `huggingface_hub.list_repo_files` to find the CSV, then `hf_hub_download`. Save as `data/raw/placementdata.csv`.
3. Kaggle API (`kaggle datasets download`), only if `~/.kaggle/kaggle.json` exists. Search Kaggle for
   "placement prediction dataset placementdata.csv" to confirm the slug.
4. Otherwise print a clear message telling the user to download `placementdata.csv` from Kaggle into `data/raw/`
   and exit with code 1.

Validate after loading: expected columns present, ≥ 5,000 rows. Record source URL + row count in `metrics.json → data`.

### 2.2 Simulated columns (fill the two features the public dataset lacks)
No public dataset contains all 8 features. We add `ProgrammingSkills` and `Attendance` with a **documented, seeded
generative rule**. This must be stated plainly in the report (Dataset section + Limitations): their influence results
reflect our simulation assumptions, not real-world evidence.

Use `rng = np.random.default_rng(42)`; `z(x)` = standardised column; `placed` = 1/0 from the original label.
```
ProgrammingSkills = clip(round(5.5 + 1.1*z(Projects) + 0.7*z(Internships) + 0.4*z(CGPA)
                                + 0.6*placed + N(0, 1.6)), 1, 10)
Attendance        = clip(round(80 + 3.0*z(CGPA) + 2.0*placed + N(0, 7.5), 1), 45, 100)
```
Effects are deliberately modest so the simulated columns do not dominate the real ones. Don't retune these to
produce a particular result. Put the formulas in `src/config.py` comments and in the report.

### 2.3 Inject realistic data-quality issues (so preprocessing is meaningful)
The Kaggle file is clean. To show a realistic cleaning pipeline, inject a **seeded, logged** set of issues
into `data/interim/placement_raw_augmented.csv`, and say so clearly in the report:
- **Missing values:** ~3% of `CGPA`, `AptitudeScore`, `CommunicationSkills`, `Attendance`, `ProgrammingSkills` (MCAR).
- **Duplicates:** ~1% exact duplicate rows.
- **Outliers / invalid entries:** ~0.5% rows: CGPA > 10 (e.g. 12.5, 78 — a percentage mistakenly typed), Attendance > 100 or negative, AptitudeScore > 100.
- **Inconsistent labels:** target spelled as `Placed`, `placed`, ` Placed `, `NotPlaced`, `Not Placed`, `not placed`, `NP`.
- **Inconsistent categoricals:** `Yes/yes/Y/No/no/N` in ExtracurricularActivities and PlacementTraining.
- **Type noise:** a few numeric cells stored as strings like `"8.2 "`.
Save an `injection_log` (what was injected, how many) into `reports/preprocessing_log.json`.

## 3. EDA (`src/eda.py`) — figures (300 dpi PNG, consistent palette, readable in a 2-column paper)
Save to `reports/figures/`:
1. `class_balance.png` — bar chart of Placed vs Not Placed with % labels.
2. `feature_distributions.png` — grid of histograms/KDEs for the 8 features, split by class.
3. `boxplots_by_class.png` — boxplots of numeric features by class.
4. `correlation_heatmap.png` — Pearson/Spearman correlation incl. target.
5. `missing_values.png` — missing-value counts before cleaning.
6. `placement_rate_by_bins.png` — placement rate vs. binned CGPA / Internships / Projects / Aptitude (bar or line).
7. `outliers_before_after.png` — e.g. CGPA and Attendance boxplot before vs after cleaning.
Also write `reports/data_summary.csv` (describe() + missing %) for the report's dataset table.

## 4. Preprocessing (`src/preprocess.py`) — order matters, log every count
1. Standardise column names to the final names in §1.
2. Strip whitespace; coerce numeric columns with `pd.to_numeric(errors="coerce")`.
3. Normalise target to 0/1 via mapping; normalise Yes/No categoricals to 1/0.
4. Drop exact duplicates.
5. Invalid values → NaN using domain ranges (CGPA 0–10, %s 0–100, ratings within scale, counts ≥ 0).
   Exception: CGPA in (10, 100] is likely a percentage → convert `CGPA = value/9.5` (common Indian conversion), clip to 10; log how many.
6. Statistical outliers in remaining numeric columns: IQR rule (1.5×) → **cap (winsorise)**, don't drop; log counts.
7. **Train/test split happens BEFORE imputation** (stratified 80/20, random_state 42). Imputation is part of the
   sklearn Pipeline (median for numeric) so it's fit on training data only — this prevents leakage. Explain this in the report.
8. Save `data/processed/placement_clean.csv` (pre-imputation, cleaned) and the log.

## 5. Feature engineering (`src/features.py`)
- **Engineered features** (test their value in the ablation, §7.4):
  - `ExperienceScore = Internships + Projects + Certifications`
  - `SkillIndex = mean(z(ProgrammingSkills), z(CommunicationSkills), z(AptitudeScore))`
  - `AcademicConsistency = CGPA * Attendance / 100`
- **Transformation:** `StandardScaler` for linear/distance-based models (LR, SVM, KNN); trees use unscaled input
  (keep one pipeline factory that chooses per model).
- **Encoding:** binary categoricals already 0/1; if any multi-class categorical appears, OneHotEncoder.
- **Feature selection analysis** (report + figure `feature_selection.png`): mutual information scores,
  chi-square / ANOVA F-test, and RFE with Logistic Regression. Report which features each method ranks top.
- **Multicollinearity:** VIF table → `reports/vif.csv`.
- Main model uses the 8 core features (+ engineered features only if ablation shows they help ≥ 0.005 ROC-AUC in CV; record the decision in DECISIONS.md).

## 6. Models (`src/train.py`)
Train these 5 (justify each in the report in 1–2 lines):
| Model | Why |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients give direct influence + odds ratios |
| K-Nearest Neighbours | Simple instance-based, non-parametric comparison |
| SVM (RBF kernel, `probability=True`) | Strong on medium-size tabular data with non-linear boundaries |
| Random Forest | Robust ensemble, handles non-linearity, built-in importance |
| XGBoost (fallback: sklearn `HistGradientBoostingClassifier`) | State of the art for tabular data |

- Each model = full `sklearn.Pipeline` (imputer → [scaler] → model).
- Hyperparameter tuning: `RandomizedSearchCV` (n_iter 20–30, 5-fold `StratifiedKFold`, scoring `roc_auc`) on the training set only. Keep grids small so the whole run is < 5 min on a laptop CPU (`n_jobs=-1`).
- Class balance: check ratio; if minority < 40%, use `class_weight="balanced"` / `scale_pos_weight` (not SMOTE — keep it simple; mention SMOTE as alternative).
- Pick best model by **mean CV ROC-AUC** (not test score — explain why), then evaluate once on the held-out test set.
- Decision threshold: default 0.5; also report the F1-optimal threshold found on CV predictions (don't tune on test).
- Save every fitted pipeline to `models/all_models/<name>.joblib`, best to `models/best_model.joblib`, plus `models/model_meta.json` (feature list, ranges for UI, threshold, model name, training date, sklearn version).

## 7. Evaluation
### 7.1 Metrics (per model, test set) → `reports/model_comparison.csv` and `metrics.json`
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, plus CV mean ± std ROC-AUC and F1, train time.
### 7.2 Figures
- `roc_curves.png` (all models on one plot), `pr_curves.png`
- `confusion_matrices.png` (grid, all models, normalised + counts)
- `model_comparison_bar.png` (grouped bars of main metrics)
- `cv_boxplot.png` (distribution of CV ROC-AUC per model)
- `calibration_curve.png` for the best model
- `learning_curve.png` for the best model (shows over/underfitting)
### 7.3 Statistical comparison
Paired comparison of the top 2 models on the same CV folds (paired t-test on fold scores, or McNemar on test predictions). Report p-value and plain-English interpretation.
### 7.4 Ablation (small table → `reports/ablation.csv`)
CV ROC-AUC of the best model type with: (a) 8 core features, (b) core + engineered, (c) core + extra Kaggle columns, (d) core minus the 2 simulated columns. (d) shows how much the simulated columns matter — important for honesty.

## 8. Influential features (`src/explain.py`) — the "additional" requirement
Use several methods and show they agree:
1. **Logistic Regression coefficients** (on standardised data) + odds ratios with sign → `lr_coefficients.png`.
2. **Tree impurity importance** (RF / XGB) → part of `importance_comparison.png`.
3. **Permutation importance** on the test set for the best model (n_repeats=20, with std error bars) → `permutation_importance.png`.
4. **SHAP** (`TreeExplainer` if the best model is tree-based, else `LinearExplainer`/`KernelExplainer` on a 200-row sample): `shap_summary_beeswarm.png`, `shap_bar.png`, and `shap_dependence_<top3>.png` for the 3 most important features.
5. **Consensus ranking:** normalise each method's importances to [0,1], average the ranks → `reports/feature_importance.csv` and `feature_importance_consensus.png`. Compute Spearman rank correlation between methods (agreement table in report).
6. **Partial dependence** plots for the top 3 features → `pdp_top3.png` (shows *how* each feature changes placement probability, e.g. threshold effects in CGPA).

Write plain-English findings into `metrics.json → insights` (e.g. "CGPA is the strongest predictor; probability rises sharply above ~7.5"). These are generated from the numbers, not hand-written. The app and report reuse them.

## 9. `src/predict.py` (used by the app)
- `load_model()`, `predict_one(dict) -> {proba, label, shap_values}`, `predict_batch(df)`.
- `what_if(student)`: for each actionable feature (CGPA +0.5, ProgrammingSkills +1, Internships +1, Certifications +1, CommunicationSkills +0.5, AptitudeScore +10, Projects +1, Attendance +5; capped at max), compute the new probability; return sorted by gain. Used for the "How to improve your chances" panel.
- Input validation against ranges in `model_meta.json`.

## 10. `run_all.py`
Runs: get_data → inject/augment → eda (before cleaning figs) → preprocess → eda (after) → features analysis → train → explain → copy figures to `report/figures/` → print a short summary table. Use `logging` with timestamps. Total runtime target: < 5 minutes on a laptop CPU.

## 11. `requirements.txt` (pin major versions, no GPU packages)
pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn, plotly, streamlit, joblib, scipy, statsmodels, huggingface_hub, nbformat, nbconvert, ipykernel (+ kaggle optional).

# Design decisions

Short log of choices made while building the project (useful for the viva).

## Project setup
- **Repository root = project root.** The spec files were first unpacked into `ml-project/`; at the user's request everything lives at the repo root instead.
- **Python 3.12** virtual environment in `.venv/`; CPU only. Major versions pinned with ranges in `requirements.txt`.
- **No LaTeX installed** on the build machine, so the report is delivered as LaTeX source + `report_overleaf.zip` (compile on Overleaf).

## Data
- **Dataset source:** Hugging Face mirror `Krooz/Campus_Recruitment_CSV` of the Kaggle *Placement Prediction Dataset* (`placementdata.csv`, 10,000 rows). Kaggle API is only a fallback.
- **Simulated columns** (`ProgrammingSkills`, `Attendance`) use exactly the formulas in the spec (§2.2); they were not tuned.
- **Injected issues** (all seeded with 42, counts logged in `reports/preprocessing_log.json`): 3% MCAR missing values in 5 columns, 1% duplicate rows, 0.5% invalid entries, 15% inconsistent target spellings, 10% inconsistent Yes/No values per categorical, 25 whitespace-padded numeric strings per column in CGPA and AptitudeTestScore.
- **CGPA percentage rule:** values in (10, 100] are converted with `CGPA = value / 9.5` only if they are ≥ 35 (a plausible percentage). Values like 12.5 or 15 are physically impossible as either a CGPA or a pass-level percentage, so they become missing and are imputed. (The spec lists 12.5 as an example of a bad value; dividing it by 9.5 would give a meaningless CGPA of 1.3.)
- **Duplicate detection uses StudentID** before StudentID is dropped, so two different students who happen to have identical marks are never merged.

## Preprocessing / leakage
- **Train/test split is done before any statistic is learned.** The stratified 80/20 split is stored as a `Split` column in `data/processed/placement_clean.csv` so every later step uses the identical split.
- **IQR capping bounds are computed on the training rows only** (stricter than the spec's order, which places capping before the split) – this avoids even a small leak of test-set quartiles.
- **IQR capping is applied only to continuous columns** (CGPA, AptitudeScore, CommunicationSkills, Attendance, SSC/HSC marks). Small count/rating columns (Internships, Projects, …) are only range-checked, because the IQR rule would clip legitimate values such as 3 internships.
- **Median imputation lives inside the sklearn Pipeline**, so it is re-fitted on the training folds in every CV split.
- **The HF mirror has no single `placementdata.csv`:** it stores the same 10,000 rows as `train/validation/test.csv` (7,225 + 1,275 + 1,500) without the `StudentID` column. `get_data.py` concatenates them and adds a sequential `StudentID` (1..N) to recreate the Kaggle file layout.

## Models
- **Class balance:** the minority class (Placed) is 42% of the training set, above the 40% threshold, so no class weighting / resampling is used (SMOTE mentioned as an alternative in the report).
- **SVM search speed:** during RandomizedSearchCV the SVM uses `probability=False` (ROC-AUC is computed from `decision_function`, ~5× faster) and only 10 candidates; the selected configuration is then refitted with `probability=True` so the app can show probabilities.
- **XGBoost/RF use `n_jobs=1` inside the model** and `n_jobs=-1` in the search, to avoid nested parallelism.
- **Statistical test:** besides the plain paired t-test on the 5 CV folds we report the Nadeau–Bengio corrected t-test (fold scores share training data, so the plain test is over-confident) and McNemar's test on the test predictions.
- **Engineered features not used:** the ablation shows they change CV ROC-AUC by −0.0008 (rule: use only if gain ≥ 0.005).
- **Extra Kaggle columns not used in the main model** even though they add ~+0.018 CV ROC-AUC: the problem statement defines the 8 core features, and the app asks only for those. This is reported honestly in the ablation table.
- **Saved models are joblib-compressed** (`compress=3`): the Random Forest drops from 27 MB to a few MB, so `models/` (≈6.5 MB) can be committed and the app works right after cloning.

## Explainability
- **SHAP values in probability units:** for tree models we use interventional TreeSHAP with `model_output="probability"` (100-row background), so "+0.12" means "+12 percentage points". Non-tree models fall back to KernelExplainer on a 200-row sample.
- **Consensus ranking = mean rank over 5 methods** (LR |coef|, RF impurity, XGBoost gain, permutation, SHAP); the normalised mean score is reported as a tie-breaker. Method agreement is the pairwise Spearman correlation of the normalised importances.
- **Insight wording is rule-based on the numbers** (e.g. "agree strongly" only if mean ρ ≥ 0.8, "moderately" if ≥ 0.5).

## Web app
- **No emojis / decorative icons** in the UI (page names, buttons, verdicts) to keep a professional look; colour and layout carry the meaning instead.
- **`width="stretch"`** is used instead of `use_container_width` (deprecated in Streamlit 1.67); `requirements.txt` therefore requires `streamlit>=1.50`.
- **Sliders for continuous features** use the valid domain maximum but start near the observed minimum (e.g. CGPA from 6 rather than 0), so they stay usable; the backend still validates the full domain range.
- **Risk bands** on the batch page: High < 35%, Medium 35–65%, Low ≥ 65% placement probability.
- **Preset profiles:** "Strong" and "At risk" are hand-picked illustrative profiles; "Average" uses the training-data medians from `model_meta.json`.
- **Playwright** (screenshots only) is an optional dev tool and is not in `requirements.txt`.

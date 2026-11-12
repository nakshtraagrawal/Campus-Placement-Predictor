# 03 — Report Spec: IEEE Two-Column Paper (LaTeX)

## Format
- `\documentclass[conference]{IEEEtran}` — two-column IEEE conference format. If `IEEEtran.cls` isn't installed, download it from CTAN into `report/` (Overleaf has it built in).
- Packages: `graphicx, booktabs, amsmath, cite, xcolor, hyperref, url, multirow, array`.
- Title (suggested): *"Predicting Campus Placement Outcomes of Engineering Students Using Machine Learning and Explainable Feature Analysis"*.
- Authors: IEEE author block with placeholders `[Member 1 Name]`, `[Member 2 Name]`, `[Department]`, `[College]`, `[email]`.
- Length target: **6–8 pages** in two-column format. Figures single-column (`\columnwidth`) unless wide (`figure*`) for ROC/confusion grids.
- Every figure/table has a caption, a label, and is referenced in the text. Figures come from `report/figures/`.
- **All numbers come from `reports/*.json|csv`.** Write a small helper `src/make_report_tables.py` that generates LaTeX tables (`report/tables/*.tex`) from the CSVs, and `\input{}` them. Numbers inside running text: read from `metrics.json` while writing; double-check before finishing.

## Sections (map 1:1 to the college guidelines)
0. **Abstract** (150–200 words: problem, data, 5 models, best result with metric, top influential features) + **Index Terms**.
1. **Introduction** — problem, why placement prediction matters to colleges/students (early intervention, training-cell planning), objectives as a bullet list, contributions, paper organisation.
2. **Literature Survey** — **4 papers minimum (2 per team member, team of 2)**. Use web search to find **real, verifiable** papers (IEEE Xplore, Springer, Elsevier, arXiv, reputable journals; prefer 2018+) on student placement / campus recruitment / employability prediction with ML. For each: authors, year, venue, dataset, methods, key result, limitation/gap. Add 2–4 more papers as general references if helpful (e.g. SHAP — Lundberg & Lee 2017 NeurIPS; XGBoost — Chen & Guestrin 2016 KDD; Random Forests — Breiman 2001). End with a **comparison table** (Paper | Dataset | Methods | Best metric | Gap) and a paragraph on how this work addresses gaps (multiple models + multi-method explainability + deployed app).
   - Put the full BibTeX in `references.bib` with DOIs/URLs. Mark each in `REPORT_TODO.md` so the students verify and assign "reviewed by Member 1/2".
3. **Dataset Description** — source (Kaggle placementdata.csv via HF mirror, URL), size, attribute table (name, type, range, description, source: real/simulated), class balance, `data_summary` table, EDA figures (class balance, distributions, correlation heatmap, placement-rate-by-bins). **Clearly state** that `ProgrammingSkills` and `Attendance` were simulated with the given formulas because no public dataset contains them, and that data-quality issues were injected (with counts) to evaluate the cleaning pipeline.
4. **Data Preprocessing** — each step with counts from `preprocessing_log.json` (table), missing-value strategy (median imputation inside pipeline fit on train only → no leakage), duplicate removal, invalid-range handling & CGPA percentage conversion, IQR capping, label/categorical normalisation, stratified split. Before/after outlier figure.
5. **Feature Engineering** — engineered features + formulas, scaling (which models and why), encoding, feature-selection analysis (MI, ANOVA F, RFE) with figure, VIF table, ablation result and final feature set decision.
6. **Design Phase** — system architecture diagram (draw with **TikZ** in LaTeX, or generate `architecture.png` with matplotlib): Raw data → Augmentation → Cleaning → Feature Eng. → Model training & tuning (CV) → Evaluation → Explainability → Saved pipeline → Streamlit app (single/batch prediction, what-if). Describe the methodology/workflow step by step.
7. **Implementation** — language, libraries with versions (from `pip freeze` of the real env), hardware (laptop CPU; note GPU not required for tabular models of this size), experimental setup: split ratio, 5-fold stratified CV, RandomizedSearch settings, random seed, scoring metric, threshold choice. Brief description of the web app with 1–2 screenshots from `reports/screenshots/` (if they exist).
8. **Model Development** — each of the 5 models: short description, justification, tuned hyperparameters (table).
9. **Results and Interpretation**
   - Main comparison table (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, CV ROC-AUC mean±std).
   - ROC curves, confusion matrices, metric bar chart, CV box plot, calibration & learning curve of the best model.
   - Statistical significance test between the top 2.
   - **Influential features**: consensus ranking table, SHAP summary, permutation importance, LR odds ratios, PDP of top features, method-agreement (Spearman) — interpret in plain language (direction + magnitude + thresholds).
   - Ablation table, incl. effect of removing the simulated columns.
   - Strengths / limitations of each model (short table or paragraph).
   - Practical recommendations for the college's training & placement cell based on the findings.
10. **Conclusion & Future Work** — summary; future: real institutional data with true programming/attendance records, more features (domain, backlogs, company tier), fairness analysis, deep tabular models, temporal validation on a new batch.
- **References** — IEEE style (`\bibliographystyle{IEEEtran}`), every entry cited at least once.

## Writing style
- Formal academic English, third person, concise. No marketing tone. No invented claims.
- Be honest about limitations (simulated columns, injected noise, single public dataset, possible synthetic origin of the Kaggle dataset itself).

## Build
- If `latexmk` or `pdflatex` + `bibtex` is available: compile to `report/main.pdf` and fix all errors and overfull-box warnings that visibly break the layout.
- Always also produce `report_overleaf.zip` (main.tex, references.bib, figures/, tables/, IEEEtran.cls if downloaded) for upload to Overleaf.

## `REPORT_TODO.md` (for the students)
List, as a checklist:
- Fill in names, roll numbers, department, college, emails, guide name.
- Open each literature-survey paper, verify citation details and summary, and record who reviewed which 2 papers.
- Check that all numbers in text match the tables (they should — just re-read).
- Add any college-specific front matter (certificate page, declaration) if required separately.
- Prepare for viva: read `notebooks/walkthrough.ipynb` and `DECISIONS.md`.

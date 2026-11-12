# CLAUDE.md — Student Placement Prediction (ML College Project)

You are building a complete, submission-ready college Machine Learning project **end to end, autonomously**.
Read this file fully, then read every file in `docs/` before writing any code.

| File | What it defines |
|---|---|
| `docs/01_PROJECT_SPEC.md` | Problem, dataset plan, preprocessing, feature engineering, models, evaluation, feature influence |
| `docs/02_UI_SPEC.md` | The Streamlit web app (look, pages, behaviour) |
| `docs/03_REPORT_SPEC.md` | The IEEE two-column LaTeX report, section by section |

## Problem in one line
An engineering college wants to predict whether a student will be **Placed / Not Placed** from
CGPA, programming skills, internships, certifications, communication skills, aptitude score, projects and attendance —
and **identify which features are most influential**.

## Ground rules
- **Small project.** No CI, no Docker, no tests beyond a smoke check.
- **Git: the user commits manually. NEVER run any git command that changes the repository** (`git init`, `add`, `commit`, `push`, `branch`, `checkout`, `reset`, `stash`, etc.). Read-only commands like `git status` / `git diff` are fine.
  Instead, at the end of each Part (and whenever a meaningful chunk is done), **print a "Suggested commit" block** for the user to run themselves:
  ```
  git add <specific paths>
  git commit -m "<type>: <short summary>" -m "<1–3 line body>"
  ```
  Use conventional-style messages (`feat:`, `fix:`, `docs:`, `chore:`), keep them short, and list exact paths (not `git add .` unless the repo is new). On the first one, include `git init` and a `.gitignore` you created (ignore `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, LaTeX build files `*.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz`). Keep `data/raw/` and `models/` ignored too unless they are small (< 20 MB), and say so.
  Also append every suggested commit to `COMMITS.md` so the user can find them later.
- **Python 3.10–3.12**, CPU only. Nothing here needs a GPU; do not add PyTorch/TensorFlow.
- **Reproducible:** every random process uses `RANDOM_STATE = 42` from `src/config.py`.
- **One command rebuilds everything:** `python run_all.py` (data → preprocessing → training → explainability → figures → metrics → report assets).
- **No fabricated results.** Every number in the report and app must come from files the pipeline wrote (`reports/metrics.json`, `reports/*.csv`). Never type a metric by hand.
- **No fabricated citations.** Literature-survey papers must be real and verifiable (use web search). If you cannot verify a paper, don't use it.
- Work on Windows and Linux (use `pathlib`, no shell-specific commands in Python).
- Write clean, commented, readable code — students will be asked to explain it in a viva.
- Don't stop to ask questions unless truly blocked; make a sensible choice and record it in `DECISIONS.md` (short bullet list).

## Build order (3 parts)

### Part 1 — Data + ML pipeline
1. Create the folder structure below, `requirements.txt`, a virtual-env note in README.
2. Implement `src/` exactly as described in `docs/01_PROJECT_SPEC.md`.
3. Run `python run_all.py` and fix until it completes cleanly.
4. Sanity check: best model test ROC-AUC should be roughly 0.80–0.90 on this data. If it's ≥0.99, you have leakage — find and fix it. If it's < 0.70, something is broken.

### Part 2 — Web app
1. Build the Streamlit app per `docs/02_UI_SPEC.md`.
2. Start it (`streamlit run app/Home.py --server.headless true`), make sure every page loads without exceptions, then stop it. If Playwright is available, take screenshots of each page to `reports/screenshots/` (these go into the report); otherwise skip.

### Part 3 — Report + final polish
1. Write `report/main.tex` + `report/references.bib` per `docs/03_REPORT_SPEC.md`.
2. If `pdflatex`/`latexmk` exists, compile to `report/main.pdf`; otherwise leave it ready for Overleaf (zip `report/` → `report_overleaf.zip`).
3. Write the final `README.md` (setup, run, project structure, screenshots) and `REPORT_TODO.md` (things the students must personally do — see report spec).
4. Run the **Definition of Done** checklist below and fix anything failing.

## Target folder structure
```
placement-predictor/
├── CLAUDE.md, docs/                  # these spec files (keep them)
├── README.md, DECISIONS.md, REPORT_TODO.md
├── requirements.txt
├── run_all.py                        # runs the full pipeline in order
├── data/
│   ├── raw/                          # downloaded Kaggle CSV (untouched)
│   ├── interim/placement_raw_augmented.csv   # + simulated columns + injected quality issues
│   └── processed/placement_clean.csv
├── src/
│   ├── config.py                     # paths, constants, feature lists, RANDOM_STATE
│   ├── get_data.py                   # download + augment (see spec §2)
│   ├── preprocess.py                 # cleaning (spec §4)
│   ├── features.py                   # feature engineering + sklearn ColumnTransformer (spec §5)
│   ├── eda.py                        # EDA figures (spec §3)
│   ├── train.py                      # model training, tuning, evaluation (spec §6–7)
│   ├── explain.py                    # feature influence analysis (spec §8)
│   └── predict.py                    # load model, predict one/many, what-if suggestions
├── models/
│   ├── best_model.joblib             # full sklearn Pipeline (preprocess + model)
│   └── all_models/*.joblib
├── reports/
│   ├── figures/*.png                 # 300 dpi, used by the LaTeX report
│   ├── metrics.json                  # all metrics, CV results, chosen model, data stats
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── preprocessing_log.json        # counts of what cleaning fixed
│   └── screenshots/
├── notebooks/walkthrough.ipynb       # short executed notebook calling src/ (for the viva)
├── app/
│   ├── Home.py                       # Streamlit entry
│   ├── pages/…                       # see UI spec
│   └── components.py                 # shared UI helpers / CSS
├── .streamlit/config.toml            # theme (project root; run streamlit from root)
└── report/
    ├── main.tex, references.bib, IEEEtran.cls (if not available system-wide, download from CTAN)
    └── figures/ (copied from reports/figures)
```

## Definition of Done
- [ ] `python run_all.py` runs from a fresh clone (after `pip install -r requirements.txt` + dataset present) without errors.
- [ ] ≥ 4 models trained and compared (at least 3 required by guidelines), with justification text.
- [ ] `reports/metrics.json`, `model_comparison.csv`, `feature_importance.csv`, `preprocessing_log.json` exist and are consistent with each other.
- [ ] All figures listed in the specs exist in `reports/figures/`.
- [ ] Streamlit app: all pages load, prediction works, batch CSV upload works, no tracebacks shown to the user.
- [ ] `report/main.tex` covers all 10 guideline sections, uses IEEEtran two-column conference format, all numbers pulled from pipeline outputs, ≥ 4 real papers in literature survey + table.
- [ ] `notebooks/walkthrough.ipynb` executed with outputs saved.
- [ ] README explains setup + run in ≤ 10 commands.
- [ ] `REPORT_TODO.md` lists everything the students must fill in/verify themselves.

# Suggested commits

Copy-paste these in order (the repository already exists, so no `git init` is needed).

## Part 1 — data + ML pipeline

```
git add .gitignore CLAUDE.md docs/ README.md requirements.txt DECISIONS.md COMMITS.md run_all.py src/ data/ models/ reports/ report/figures/ report/tables/
git commit -m "feat: data + ML pipeline (cleaning, 5 tuned models, explainability)" -m "Downloads the placement dataset, adds 2 simulated columns and injected data issues, cleans it, tunes LR/KNN/SVM/RF/XGBoost with 5-fold CV and runs multi-method feature-influence analysis. python run_all.py rebuilds everything."
```
Note: `data/` (~1.5 MB) and `models/` (~6.5 MB, compressed) are small, so they are committed rather than ignored.

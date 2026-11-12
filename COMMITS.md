# Suggested commits

Copy-paste these in order (the repository already exists, so no `git init` is needed). Messages are kept to one line.

## Part 1 — data + ML pipeline

```
git add .gitignore CLAUDE.md docs/ README.md requirements.txt DECISIONS.md COMMITS.md run_all.py src/ data/ models/ reports/ report/figures/ report/tables/
git commit -m "feat: data + ML pipeline (cleaning, 5 tuned models, explainability)"
```
Note: `data/` (~1.5 MB) and `models/` (~6.5 MB, compressed) are small, so they are committed rather than ignored.

## Part 2 — Streamlit app (verified, no emojis)

```
git add -A app/ .streamlit/ reports/screenshots/ requirements.txt DECISIONS.md COMMITS.md
git commit -m "feat: polish Streamlit app and add screenshots"
```
`-A` on `app/` records the page renames (the old emoji file names are deleted).

## Part 3a — notebook + IEEE report

```
git add notebooks/ report/ report_overleaf.zip src/ run_all.py app/pages/5_About.py reports/ DECISIONS.md COMMITS.md
git commit -m "docs: add walkthrough notebook and IEEE LaTeX report"
```

## Part 3b — README, report TODO, final checks

```
git add README.md REPORT_TODO.md report/main.tex report_overleaf.zip COMMITS.md
git commit -m "docs: final README, report TODO and section references"
```

# Report TODO — for the team

Items the team must complete or verify personally before submission.

## 1. Personal details
- [ ] `report/main.tex`: replace `[Member 1 Name]`, `[Member 2 Name]`, `[Department]`, `[College]`, `[City]` and `[email]` in the author block.
- [ ] Add roll numbers and the guide's name if your college's format requires them (for example, in an acknowledgement or on a cover page).
- [ ] `app/pages/5_About.py`: replace `[Member 1]`, `[Member 2]`, `[Roll number]`, `[Guide name]`, `[Department]` and `[College]`.

## 2. Literature survey (2 papers per member)
Open each paper through its DOI, check the citation details and the summary written in Section II and Table I, and record who reviewed it.

| Paper | DOI | Reviewed by | Checked |
|---|---|---|---|
| Pathak et al., 2025, *Ingénierie des Systèmes d'Information* 30(7) | 10.18280/isi.300704 | Member 1 | [ ] |
| Kumar et al., 2023, *IJMECS* 15(6) | 10.5815/ijmecs.2023.06.01 | Member 1 | [ ] |
| Casuat et al., 2020, IEEE ICSET | 10.1109/ICSET51301.2020.9265371 | Member 2 | [ ] |
| Basha et al., 2023, IEEE CSITSS | 10.1109/CSITSS60515.2023.10334118 | Member 2 | [ ] |
| Sibagariang, 2025, *JNTETI* 14(3) (additional) | 10.22146/jnteti.v14i3.20516 | either | [ ] |
| Mezhoudi et al., *J. Ambient Intell. Humaniz. Comput.* 14 (survey, additional) | 10.1007/s12652-021-03276-9 | either | [ ] |

Points to check in particular:
- [ ] Pathak et al.: confirm the issue number (7) and the reported accuracy/AUC of the voting ensemble.
- [ ] Kumar et al.: the abstract gives no dataset size or numeric results. If the full text does, add them to Table I.
- [ ] Basha et al.: add page numbers from IEEE Xplore, and the dataset size and metrics if available.
- [ ] Mezhoudi et al.: published online in 2021, in volume 14 (2023). Confirm which year your college prefers.
- [ ] The general method references (SHAP, XGBoost, Random Forest, SVM, KNN, PDP, Nadeau–Bengio, McNemar, SMOTE, scikit-learn) are standard, but re-check the page numbers.

## 3. Dataset reference
- [ ] The Hugging Face mirror (`Krooz/Campus_Recruitment_CSV`) was used and verified. Find the original Kaggle page of the "Placement Prediction Dataset" (`placementdata.csv`), and add its URL and author to `references.bib` (`dataset_hf` entry).
- [ ] `src/config.py` → `KAGGLE_URL` is only an unverified fallback slug. Correct it if you find the real one.

## 4. Numbers and figures
- [ ] Compile on Overleaf (upload `report_overleaf.zip`, compiler pdfLaTeX) and check there are no errors or overfull boxes that break the layout. The LaTeX was checked statically but not compiled locally.
- [ ] Check the page count is 6–8 pages. If it is longer, the easiest cuts are the feature-selection figure (its numbers are also in the feature-selection table), the precision–recall half of the ROC/PR figure, or the strengths-and-limitations table.
- [ ] Re-read the text against the tables. All numbers come from `report/tables/macros.tex` and the generated tables, so they should match. If you re-run `python run_all.py`, re-upload the new zip.

## 5. College-specific front matter
- [ ] Add the certificate page, declaration, acknowledgement and abstract page if your college requires them separately from the IEEE paper.

## 6. Viva preparation
- [ ] Read `notebooks/walkthrough.ipynb` (executed, outputs saved) and `DECISIONS.md`.
- [ ] Be ready to explain: why the split happens before imputation (leakage), why the model is chosen by CV ROC-AUC and not the test score, what the two simulated columns are and why their influence must be treated with caution, and how to read the SHAP and partial dependence plots.
- [ ] Run the app (`streamlit run app/Home.py`) once before the demo. The "Strong", "Average" and "At risk" example profiles give a quick demonstration.

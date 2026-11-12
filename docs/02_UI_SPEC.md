# 02 — UI Spec: Streamlit Web App

**Stack:** Streamlit (multipage) + Plotly for all interactive charts. Run with `streamlit run app/Home.py`.
Chosen because it's pure Python, quick to build and reliable on demo day. It must still look like a polished product,
not a default Streamlit demo.

## Design direction
- **Feel:** clean, modern "analytics dashboard" — calm whitespace, card-based layout, one accent colour.
- **Theme** (`.streamlit/config.toml` at project root):
  - primaryColor `#4F46E5` (indigo), backgroundColor `#F8FAFC`, secondaryBackgroundColor `#FFFFFF`, textColor `#0F172A`, font `sans serif`.
- **Semantic colours** (use everywhere, incl. Plotly): Placed = `#10B981` (green), Not Placed = `#EF4444` (red), neutral/model series from a small fixed palette (indigo, teal, amber, slate, pink).
- **Custom CSS** in `app/components.py` (`inject_css()` called on every page):
  - Load **Inter** from Google Fonts; headings 600 weight; slightly tighter letter-spacing on titles.
  - Card class: white background, 12px radius, 1px `#E2E8F0` border, soft shadow, 20–24px padding.
  - Hide Streamlit's default menu/footer ("Made with Streamlit"); reduce top padding.
  - Styled metric tiles (big number, small label, optional delta).
- Plotly: template `plotly_white`, Inter font, no chart junk, titles left-aligned, hover templates formatted (e.g. `78.4%`).
- Every page has a short subtitle sentence explaining what it shows (evaluators may not be ML people).
- Responsive at 1366×768 (typical college laptop/projector). Use `layout="wide"`.
- Sidebar: project name + small logo emoji/icon, page nav, and a tiny "Model: XGBoost · ROC-AUC 0.87" badge read from `model_meta.json`/`metrics.json`.

## Pages

### 1. `Home.py` — Overview
- Hero: title "Campus Placement Predictor", one-line description, 2 buttons (`st.page_link`) → Predict, Insights.
- 4 metric tiles: dataset size, placement rate, best model, best ROC-AUC (all read from `metrics.json`).
- "How it works" — 4 small step cards: Data → Cleaning → 5 Models → Explainable prediction.
- Top 3 influential features (from `feature_importance.csv`) as small cards with a one-line insight each.

### 2. `pages/1_🎯_Predict.py` — Single-student prediction (the star of the demo)
- **Left column (form, inside a card):** 8 inputs with sensible widgets & ranges from `model_meta.json`:
  CGPA slider (step 0.1), ProgrammingSkills slider 1–10, Internships number/segmented, Certifications, CommunicationSkills slider (step 0.1), AptitudeScore slider, Projects, Attendance slider (%). Each has a `help=` tooltip.
  - Three preset buttons above the form: "Strong profile", "Average profile", "At-risk profile" that fill the form (use `st.session_state`).
  - "Predict" primary button. (Optionally live-update; but a button is fine.)
- **Right column (results):**
  - Big verdict card: ✅ "Likely to be Placed" (green) or ⚠️ "At risk of Not Being Placed" (red) + probability.
  - Plotly **gauge** of placement probability (0–100%) with red/amber/green bands and the decision threshold marked.
  - **"Why this prediction?"** — SHAP waterfall-style horizontal bar chart for this student (green pushes up, red pushes down), sorted by magnitude, with plain-English labels ("CGPA of 8.4 increases chances by +12%").
  - **"How to improve your chances"** — top 3 actions from `what_if()` as cards: "Do 1 more internship → probability 54% → 67% (+13 pts)". Skip actions with ≤ 0.5 pt gain; if already > 90%, show a congratulatory message instead.
  - Compare-to-peers mini chart: where this student's CGPA / Aptitude sit vs placed & not-placed distributions (small strip/violin).

### 3. `pages/2_📊_Insights.py` — Influential features & data exploration
- Tabs:
  - **Feature influence:** consensus importance bar chart (interactive), method-agreement table, SHAP summary image (`reports/figures/shap_summary_beeswarm.png`), dependence/PDP plot with a selectbox for the feature. Auto-generated insights from `metrics.json → insights` as bullet cards.
  - **Data explorer:** class balance donut, feature distribution by class (selectbox → Plotly histogram/violin), placement-rate-by-bin chart, correlation heatmap (Plotly).
  - **Data quality:** what the cleaning fixed (from `preprocessing_log.json`): tiles for missing values imputed, duplicates removed, outliers capped, labels normalised; before/after figure.

### 4. `pages/3_🤖_Models.py` — Model comparison
- Metrics table (from `model_comparison.csv`) with the best value in each column highlighted; best model row badge "Selected".
- Interactive ROC curves (all models), PR curves, grouped metric bar chart, CV score box plot.
- Confusion matrix with a model selector.
- Short "Why we chose X" text + the statistical test result (from `metrics.json`).
- Expander: hyperparameters of each tuned model.

### 5. `pages/4_📁_Batch_Prediction.py` — Bulk scoring for the placement cell
- Download a CSV **template** button (the 8 columns + example rows).
- File uploader → validate columns/ranges (friendly error listing bad rows, never a traceback) → predict.
- Results table with probability, label, risk band (High/Medium/Low), sortable; summary tiles (count at risk, etc.); histogram of probabilities; **download results** CSV.

### 6. `pages/5_ℹ️_About.py`
- Problem statement, team member placeholders (`[Member 1]`, `[Member 2]`), tech stack, dataset source + the honesty note about the 2 simulated columns and injected data issues, link to report PDF if present.

## Engineering rules
- Load model/data with `@st.cache_resource` / `@st.cache_data`.
- If `models/best_model.joblib` is missing, show a friendly card: "Run `python run_all.py` first."
- No hard-coded metrics — everything read from pipeline outputs.
- Wrap prediction calls in try/except and show `st.error` with a human message.
- Keep each page file < ~250 lines; shared helpers in `app/components.py` (css, cards, metric tiles, colour constants, loaders).
- After building, launch headless and open every page once to verify there are no exceptions; take screenshots to `reports/screenshots/` if Playwright is installed (`pip install playwright && playwright install chromium` is OK to try; skip on failure).

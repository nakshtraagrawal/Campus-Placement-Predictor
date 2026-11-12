"""About the project: problem, team, tech stack, data source and honesty notes."""
import streamlit as st

from components import ROOT, card, load_metrics, setup_page

ok = setup_page("About", "Problem statement, team, technology and an honest description of the data.", icon="ℹ️")
if not ok:
    st.stop()

m = load_metrics()
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    card("Problem statement",
         "An engineering college wants to predict whether a student will be <b>Placed</b> or <b>Not Placed</b> "
         "from CGPA, programming skills, internships, certifications, communication skills, aptitude score, "
         "projects and attendance — and identify which of these features are most influential, so the "
         "training &amp; placement cell can intervene early.")
    card("Dataset",
         f"Base data: the public <i>Placement Prediction Dataset</i> (<code>placementdata.csv</code>, "
         f"{m['data']['raw_rows']:,} students) from Kaggle, downloaded via the Hugging Face mirror "
         f"<a href='{m['data']['huggingface_mirror']}'>{m['data']['huggingface_mirror']}</a>. "
         f"Original Kaggle page: <a href='{m['data']['kaggle_original']}'>{m['data']['kaggle_original']}</a>.")
    card("⚠️ Honesty note",
         "<b>ProgrammingSkills</b> and <b>Attendance</b> are not in any public dataset, so they were "
         "<b>simulated</b> with a documented, seeded formula (modest dependence on projects, internships, "
         "CGPA and the outcome). Their importance therefore reflects our simulation assumptions, not real-world "
         "evidence. In addition, realistic data-quality problems (missing values, duplicates, invalid entries, "
         "inconsistent labels) were <b>deliberately injected</b> to demonstrate the cleaning pipeline. The public "
         "dataset itself may be partly synthetic. Predictions are a decision-support aid, not a verdict on any "
         "student.")
with c2:
    card("Team", "[Member 1] — [Roll number]<br>[Member 2] — [Roll number]<br>Guide: [Guide name]<br>"
         "[Department], [College]")
    card("Tech stack", "Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · statsmodels · "
         "matplotlib / seaborn · Plotly · Streamlit")
    card("Reproducibility", "Everything (data, cleaning, models, figures, report tables) is rebuilt by "
         "<code>python run_all.py</code> with a fixed random seed (42).")
    pdf = ROOT / "report" / "main.pdf"
    if pdf.exists():
        st.download_button("📄 Download the project report (PDF)", pdf.read_bytes(), "placement_report.pdf",
                           "application/pdf", use_container_width=True)
    else:
        st.caption("The project report (LaTeX source) is in the <code>report/</code> folder.")

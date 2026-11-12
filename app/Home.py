"""Home / overview page. Run with:  streamlit run app/Home.py"""
import streamlit as st

from components import card, load_csv, load_metrics, nice_feature, setup_page, tiles

ok = setup_page("Campus Placement Predictor",
                "Predict whether an engineering student is likely to be placed, understand <b>why</b>, "
                "and see which actions would improve their chances.")

if ok:
    m = load_metrics()
    tr = m["training"]
    best = tr["models"][tr["best_model"]]

    c1, c2, _ = st.columns([1, 1, 4])
    c1.page_link("pages/1_🎯_Predict.py", label="Predict a student", icon="🎯", use_container_width=True)
    c2.page_link("pages/2_📊_Insights.py", label="See insights", icon="📊", use_container_width=True)
    st.write("")

    tiles([
        ("Students in dataset", f"{m['clean_data']['rows']:,}", "after cleaning"),
        ("Placement rate", f"{100 * m['clean_data']['placement_rate']:.1f}%",
         f"{m['clean_data']['n_placed']:,} placed"),
        ("Best model", tr["best_model_name"], "chosen by 5-fold CV ROC-AUC"),
        ("Test ROC-AUC", f"{best['roc_auc']:.3f}", f"accuracy {100 * best['accuracy']:.1f}%"),
    ])

    st.subheader("How it works")
    steps = [
        ("Data", f"{m['data']['raw_rows']:,} student records from a public placement dataset, "
                 "plus two documented simulated features."),
        ("Cleaning", "Duplicates, wrong labels, impossible values and outliers are fixed; "
                     "missing values are imputed inside the model pipeline."),
        ("5 Models", "Logistic Regression, KNN, SVM, Random Forest and XGBoost are tuned with "
                     "5-fold cross-validation and compared."),
        ("Explainable prediction", "SHAP explains each prediction and a what-if engine suggests the "
                                   "most useful improvements."),
    ]
    for i, (col, (title, body)) in enumerate(zip(st.columns(4), steps), start=1):
        with col:
            card(f'<span class="step-num">{i}</span>{title}', body)

    st.subheader("Top 3 influential features")
    fi = load_csv("feature_importance.csv")
    insights = m.get("insights", [])
    for col, (_, row) in zip(st.columns(3), fi.head(3).iterrows()):
        f = row["feature"]
        line = next((s for s in insights if s.startswith("Average placement probability") and f" {f} =" in s),
                    f"Mean rank {row['mean_rank']:.1f} across 5 importance methods.")
        with col:
            card(f"#{int(row['consensus_rank'])} · {nice_feature(f)}", line, "insight")
    st.caption("Influence is measured by five independent methods (logistic-regression coefficients, "
               "random-forest and XGBoost importance, permutation importance and SHAP) and averaged.")

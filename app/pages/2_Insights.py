"""Influential features, data exploration and data-quality report."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components import (NOT_PLACED, PALETTE, PLACED, PRIMARY, card, load_clean, load_csv, load_metrics,
                        load_report_json, nice_feature, setup_page, show_figure, tiles)
from src import config

ok = setup_page("Insights", "Which student attributes matter most for placement, what the data looks like, "
                "and what the cleaning pipeline fixed.")
if not ok:
    st.stop()

m = load_metrics()
tab1, tab2, tab3 = st.tabs(["Feature influence", "Data explorer", "Data quality"])

# ------------------------------------------------------------------ feature influence
with tab1:
    fi = load_csv("feature_importance.csv")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        d = fi.sort_values("consensus_rank", ascending=False)  # rank 1 at the top
        fig = go.Figure(go.Bar(
            x=d["consensus_score"], y=[nice_feature(f) + (" *" if s else "") for f, s in zip(d["feature"], d["simulated"])],
            orientation="h", marker_color=["#94A3B8" if s else PRIMARY for s in d["simulated"]],
            customdata=d["mean_rank"], hovertemplate="%{y}<br>Score %{x:.2f}<br>Mean rank %{customdata:.1f}<extra></extra>"))
        fig.update_layout(height=380, title="Consensus importance (ordered by mean rank over 5 methods)",
                          xaxis_title="Consensus score (1 = top for every method)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Grey bars marked with an asterisk are simulated columns; their importance reflects the "
                   "simulation formula, not real-world evidence.")
    with c2:
        st.markdown("##### Key findings")
        insights = m.get("insights", [])
        for s in insights[:4]:
            st.markdown(f'<div class="card insight"><p>{s}</p></div>', unsafe_allow_html=True)
        with st.expander(f"{len(insights) - 4} more findings"):
            for s in insights[4:]:
                st.markdown(f"- {s}")

    st.markdown("##### Do the methods agree?")
    ag = load_csv("method_agreement.csv", index_col=0)
    labels = {"lr_coef": "LR", "rf_impurity": "RF", "xgb_gain": "XGBoost", "permutation": "Permutation",
              "shap": "SHAP"}
    ag.index = [labels[i] for i in ag.index]
    ag.columns = [labels[c] for c in ag.columns]
    a1, a2 = st.columns([3, 2], gap="large")
    a1.dataframe(ag.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                 width="stretch")
    ranks = fi[["consensus_rank", "feature"] + [f"{k}_rank" for k in labels]].copy()
    ranks.columns = ["Rank", "Feature"] + list(labels.values())
    a2.dataframe(ranks, hide_index=True, width="stretch")
    st.caption("Left: Spearman rank correlation between methods (LR = |logistic coefficient|, RF = random-forest "
               "impurity, XGBoost = gain, Permutation and SHAP = best model). Right: rank given by each method. "
               f"Mean pairwise ρ = "
               f"{m['explainability']['mean_pairwise_spearman']:.2f} (1 = identical ranking).")

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("##### SHAP summary (best model)")
        show_figure("shap_summary_beeswarm.png", "Each dot is a student; red = high feature value. "
                    "Right of zero = pushes towards Placed.")
    with s2:
        st.markdown("##### How does each feature change the probability?")
        pdp = load_report_json("pdp.json")
        f = st.selectbox("Feature", list(pdp.keys()), index=list(pdp.keys()).index(fi.iloc[0]["feature"]),
                         format_func=nice_feature)
        fig = go.Figure(go.Scatter(x=pdp[f]["grid"], y=100 * np.array(pdp[f]["avg_proba"]), mode="lines+markers",
                                   line=dict(color=PRIMARY, width=3),
                                   hovertemplate=f"{nice_feature(f)} = %{{x}}<br>Avg. probability %{{y:.1f}}%<extra></extra>"))
        fig.update_layout(height=360, title=f"Partial dependence: {nice_feature(f)}",
                          xaxis_title=nice_feature(f), yaxis_title="Avg. placement probability (%)")
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ data explorer
with tab2:
    df = load_clean()
    df["Status"] = df[config.TARGET].map({1: "Placed", 0: "Not Placed"})
    cmap = {"Placed": PLACED, "Not Placed": NOT_PLACED}
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        counts = df["Status"].value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6,
                               marker_colors=[cmap[k] for k in counts.index],
                               hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
        fig.update_layout(height=340, title="Class balance")
        st.plotly_chart(fig, width="stretch")
    with c2:
        f = st.selectbox("Feature distribution by class", config.CORE_FEATURES, format_func=nice_feature)
        kind = st.radio("Chart", ["Histogram", "Violin"], horizontal=True, label_visibility="collapsed")
        if kind == "Histogram":
            fig = px.histogram(df, x=f, color="Status", barmode="overlay", histnorm="percent", opacity=0.6,
                               color_discrete_map=cmap, nbins=30)
        else:
            fig = px.violin(df, x="Status", y=f, color="Status", box=True, color_discrete_map=cmap)
        fig.update_layout(height=300, legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Placement rate by feature value")
    f = st.selectbox("Feature", ["CGPA", "AptitudeScore", "Internships", "Projects", "Certifications",
                                 "ProgrammingSkills", "CommunicationSkills", "Attendance"], format_func=nice_feature,
                     key="bins")
    x = df[f]
    groups = x if x.nunique() <= 11 else pd.qcut(x, 8, duplicates="drop").astype(str)
    rate = df.groupby(groups, observed=True)[config.TARGET].agg(["mean", "size"]).reset_index()
    fig = go.Figure(go.Bar(x=rate[f].astype(str), y=100 * rate["mean"], marker_color=PRIMARY,
                           customdata=rate["size"],
                           hovertemplate="%{x}<br>Placement rate %{y:.1f}%<br>%{customdata} students<extra></extra>"))
    fig.update_layout(height=320, yaxis_title="Placement rate (%)", xaxis_title=nice_feature(f))
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Correlation heatmap")
    corr = df[config.CORE_FEATURES + [config.TARGET]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ data quality
with tab3:
    plog = load_report_json("preprocessing_log.json")
    cl, inj = plog["cleaning_log"], plog["injection_log"]
    st.markdown("Because the public dataset is already clean, realistic problems were **deliberately "
                "injected** (seeded and logged) so the cleaning pipeline could be demonstrated and verified.")
    tiles([
        ("Missing values imputed", f"{cl['missing_after_cleaning_total']:,}", "median, fitted on training data"),
        ("Duplicates removed", f"{cl['duplicates_removed']:,}", f"{inj['duplicate_rows_added']} injected"),
        ("Outliers capped", f"{cl['outliers_capped_total']:,}", "IQR rule (1.5×)"),
        ("Labels normalised", f"{cl['labels_normalised']:,}", "e.g. ' Placed ', 'NP' → 0/1"),
    ])
    st.write("")
    tiles([
        ("Invalid values → missing", f"{cl['invalid_to_nan_total']:,}", "outside valid ranges"),
        ("CGPA % converted", f"{cl['cgpa_percentage_converted']:,}", "percentage ÷ 9.5"),
        ("Text-noise cells fixed", f"{cl['type_noise_whitespace_fixed']:,}", "e.g. '8.2 ' → 8.2"),
        ("Rows after cleaning", f"{cl['rows_out']:,}", f"train {cl['train_rows']:,} / test {cl['test_rows']:,}"),
    ])
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        show_figure("outliers_before_after.png", "Invalid entries and outliers before vs after cleaning")
    with c2:
        show_figure("missing_values.png", "Missing cells in the raw (augmented) file")
    with st.expander("Full preprocessing log (JSON)"):
        st.json(plog)

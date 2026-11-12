"""Bulk scoring for the training & placement cell: upload a CSV, get probabilities and risk bands."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import AMBER, NOT_PLACED, PLACED, PRIMARY, card, load_meta, load_pipeline, setup_page, tiles
from src.predict import predict_batch, template_frame

ok = setup_page("Batch Prediction", "Upload a CSV with many students to score them all at once and find "
                "who needs support first. Risk band: High (< 35%), Medium (35–65%), Low (≥ 65%).", icon="📁")
if not ok:
    st.stop()

meta = load_meta()
c1, c2 = st.columns([2, 3], gap="large")
with c1:
    card("1 · Download the template",
         "The CSV must contain these columns: <code>" + "</code>, <code>".join(meta["features"]) + "</code>. "
         "Extra columns (e.g. a roll number) are kept in the output.")
    st.download_button("⬇️ Download CSV template", template_frame(meta).to_csv(index=False).encode(),
                       "placement_template.csv", "text/csv", use_container_width=True)
with c2:
    card("2 · Upload your file", "Rows with missing columns, non-numeric or out-of-range values are listed "
         "below and skipped; all other rows are scored.")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

if up is None:
    st.stop()

try:
    raw = pd.read_csv(up)
except Exception:
    st.error("The file could not be read as a CSV. Please save it as comma-separated values (UTF-8) and retry.")
    st.stop()
if raw.empty:
    st.warning("The uploaded file has no rows.")
    st.stop()

try:
    pipe, _ = load_pipeline()
    res, errors = predict_batch(raw, pipe, meta)
except Exception:
    st.error("Something went wrong while scoring the file. Please check that it matches the template.")
    st.stop()

if errors:
    with st.expander(f"⚠️ {len(errors)} problem(s) found — affected rows were skipped", expanded=res.empty):
        st.markdown("\n".join(f"- {e}" for e in errors[:200]))
        if len(errors) > 200:
            st.caption(f"... and {len(errors) - 200} more.")
if res.empty:
    st.error("No valid rows to score. Please fix the problems listed above.")
    st.stop()

n = len(res)
tiles([
    ("Students scored", f"{n:,}", f"{len(raw) - n} skipped" if len(raw) > n else "all rows valid"),
    ("Predicted placed", f"{(res['Prediction'] == 'Placed').sum():,}",
     f"{100 * (res['Prediction'] == 'Placed').mean():.0f}%"),
    ("High risk", f"{(res['RiskBand'] == 'High').sum():,}", "probability < 35%"),
    ("Average probability", f"{100 * res['PlacementProbability'].mean():.1f}%", ""),
])
st.write("")

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.markdown("##### Results (click a column header to sort)")
    show = res.sort_values("PlacementProbability")
    st.dataframe(show, hide_index=True, use_container_width=True, height=420,
                 column_config={"PlacementProbability": st.column_config.ProgressColumn(
                     "Placement probability", format="%.2f", min_value=0.0, max_value=1.0)})
    st.download_button("⬇️ Download results CSV", res.to_csv(index=False).encode(), "placement_predictions.csv",
                       "text/csv", type="primary")
with c2:
    fig = go.Figure(go.Histogram(x=100 * res["PlacementProbability"], nbinsx=20, marker_color=PRIMARY,
                                 hovertemplate="%{x}%<br>%{y} students<extra></extra>"))
    fig.add_vline(x=100 * meta["threshold"], line_dash="dash", line_color="#0F172A",
                  annotation_text="threshold")
    fig.update_layout(title="Distribution of placement probabilities", height=260,
                      xaxis_title="Probability (%)", yaxis_title="Students")
    st.plotly_chart(fig, use_container_width=True)
    counts = res["RiskBand"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[NOT_PLACED, AMBER, PLACED],
                           hovertemplate="%{x} risk: %{y} students<extra></extra>"))
    fig.update_layout(title="Students per risk band", height=240, yaxis_title="Students")
    st.plotly_chart(fig, use_container_width=True)

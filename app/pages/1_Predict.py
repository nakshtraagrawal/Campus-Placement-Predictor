"""Single-student prediction with explanation (SHAP) and what-if suggestions."""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import (AMBER, NOT_PLACED, PLACED, card, fmt_value, load_clean, load_explainer,
                        load_meta, load_pipeline, nice_feature, setup_page)
from src import config
from src.predict import predict_one, what_if

ok = setup_page("Predict", "Enter a student's profile to estimate their chance of placement, see which "
                "factors drive the prediction, and get concrete suggestions to improve it.")
if not ok:
    st.stop()

meta = load_meta()
FEATS = meta["features"]
R = meta["feature_ranges"]

ACTIONS = {
    "CGPA": "Raise CGPA by {step:.1f}", "ProgrammingSkills": "Improve programming skill by {step:.0f} level",
    "Internships": "Do {step:.0f} more internship", "Certifications": "Earn {step:.0f} more certification",
    "CommunicationSkills": "Improve communication rating by {step:.1f}",
    "AptitudeScore": "Score {step:.0f} more on the aptitude test", "Projects": "Build {step:.0f} more project",
    "Attendance": "Raise attendance by {step:.0f} points",
}
PRESETS = {
    "Strong": {"CGPA": 8.8, "ProgrammingSkills": 9, "Internships": 2, "Certifications": 3,
                       "CommunicationSkills": 4.7, "AptitudeScore": 90, "Projects": 3, "Attendance": 93.0},
    "Average": {f: (round(R[f]["default"]) if R[f]["integer"] else round(R[f]["default"], 1)) for f in FEATS},
    "At risk": {"CGPA": 6.6, "ProgrammingSkills": 3, "Internships": 0, "Certifications": 0,
                        "CommunicationSkills": 3.4, "AptitudeScore": 60, "Projects": 0, "Attendance": 68.0},
}


def ui_bounds(f):
    """Slider bounds: valid domain range, but start continuous sliders near the observed minimum."""
    lo, hi = R[f]["min"], R[f]["max"]
    if not R[f]["integer"] and f != "CommunicationSkills":
        lo = max(lo, math.floor(R[f]["data_min"]))
    return lo, hi


def apply_preset(name):
    for f, v in PRESETS[name].items():
        st.session_state[f"in_{f}"] = v
    st.session_state.pop("result", None)


for f in FEATS:  # initialise widget state once
    st.session_state.setdefault(f"in_{f}", PRESETS["Average"][f])

left, right = st.columns([5, 7], gap="large")

# ------------------------------------------------------------------ form
with left:
    st.markdown("##### Example profiles")
    pc = st.columns(3)
    for col, name in zip(pc, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")
    with st.form("student", border=True):
        st.markdown("##### Student profile")
        vals = {}
        lo, hi = ui_bounds("CGPA")
        vals["CGPA"] = st.slider("CGPA", float(lo), float(hi), step=0.1, key="in_CGPA",
                                 help=config.FEATURE_DESCRIPTIONS["CGPA"])
        vals["ProgrammingSkills"] = st.slider("Programming skills (1-10)", 1, 10, key="in_ProgrammingSkills",
                                              help=config.FEATURE_DESCRIPTIONS["ProgrammingSkills"])
        a, b, c = st.columns([1, 1.3, 1])
        vals["Internships"] = a.number_input("Internships", int(R["Internships"]["min"]), int(R["Internships"]["max"]),
                                             step=1, key="in_Internships", help=config.FEATURE_DESCRIPTIONS["Internships"])
        vals["Certifications"] = b.number_input("Certifications", int(R["Certifications"]["min"]),
                                                int(R["Certifications"]["max"]), step=1, key="in_Certifications",
                                                help=config.FEATURE_DESCRIPTIONS["Certifications"])
        vals["Projects"] = c.number_input("Projects", int(R["Projects"]["min"]), int(R["Projects"]["max"]),
                                          step=1, key="in_Projects", help=config.FEATURE_DESCRIPTIONS["Projects"])
        vals["CommunicationSkills"] = st.slider("Communication skills (1-5)", 1.0, 5.0, step=0.1,
                                                key="in_CommunicationSkills",
                                                help=config.FEATURE_DESCRIPTIONS["CommunicationSkills"])
        lo, hi = ui_bounds("AptitudeScore")
        vals["AptitudeScore"] = st.slider("Aptitude score", int(lo), int(hi), key="in_AptitudeScore",
                                          help=config.FEATURE_DESCRIPTIONS["AptitudeScore"])
        lo, hi = ui_bounds("Attendance")
        vals["Attendance"] = st.slider("Attendance (%)", float(lo), float(hi), step=1.0, key="in_Attendance",
                                       help=config.FEATURE_DESCRIPTIONS["Attendance"])
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        try:
            pipe, _ = load_pipeline()
            res = predict_one(vals, pipe, meta, load_explainer())
            res["what_if"] = what_if(vals, pipe, meta)
            res["inputs"] = dict(vals)
            st.session_state["result"] = res
        except ValueError as e:
            st.error(f"Please check the inputs: {e}")
        except Exception:
            st.error("Sorry, the prediction could not be computed. Please check the inputs and try again.")

# ------------------------------------------------------------------ results
with right:
    res = st.session_state.get("result")
    if not res:
        card("No prediction yet", "Choose a preset or fill in the profile, then press <b>Predict</b>.")
        st.stop()
    p, thr = res["proba"], meta["threshold"]
    good = p >= thr
    color = PLACED if good else NOT_PLACED
    verdict = "Likely to be placed" if good else "At risk of not being placed"
    st.markdown(f'<div class="verdict" style="background:{color}"><div class="big">{verdict}</div>'
                f'<div class="small">Estimated placement probability: <b>{100 * p:.1f}%</b> '
                f'(decision threshold {100 * thr:.0f}%) · model: {meta["model_name"]}</div></div>',
                unsafe_allow_html=True)

    g = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": "#0F172A", "thickness": 0.25},
               "steps": [{"range": [0, 35], "color": "#FECACA"}, {"range": [35, 65], "color": "#FDE68A"},
                         {"range": [65, 100], "color": "#A7F3D0"}],
               "threshold": {"line": {"color": "#4F46E5", "width": 4}, "thickness": 0.9, "value": 100 * thr}}))
    g.update_traces(number_font=dict(size=44, color="#0F172A"))
    g.update_layout(height=250, margin=dict(l=40, r=40, t=50, b=20), title="Placement probability")
    st.plotly_chart(g, width="stretch")

    # --- Why this prediction? (SHAP, probability units)
    st.markdown("##### Why this prediction?")
    sv = pd.Series(res["shap_values"]).reindex(FEATS).dropna()
    sv = sv.reindex(sv.abs().sort_values().index)
    labels = [f"{nice_feature(f)} = {fmt_value(f, res['inputs'][f])}" for f in sv.index]
    text = [f"{'increases' if v > 0 else 'decreases'} chances by {100 * v:+.1f} pts" for v in sv.values]
    fig = go.Figure(go.Bar(x=100 * sv.values, y=labels, orientation="h",
                           marker_color=[PLACED if v > 0 else NOT_PLACED for v in sv.values],
                           text=[f"{100 * v:+.1f}" for v in sv.values], textposition="outside",
                           customdata=text, hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
    span = max(100 * sv.abs().max(), 1.0)
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes(range=[min(0, 100 * sv.min()) - 0.3 * span, max(0, 100 * sv.max()) + 0.3 * span])
    fig.update_layout(height=320, xaxis_title="Contribution to placement probability (percentage points)",
                      title=f"Starting from the average student ({100 * res['base_value']:.0f}%)")
    st.plotly_chart(fig, width="stretch")
    top_pos = sv[sv > 0].sort_values(ascending=False).head(1)
    top_neg = sv[sv < 0].sort_values().head(1)
    msg = []
    if len(top_pos):
        f = top_pos.index[0]
        msg.append(f"{nice_feature(f)} of {fmt_value(f, res['inputs'][f])} increases chances by "
                   f"{100 * top_pos.iloc[0]:+.1f} pts")
    if len(top_neg):
        f = top_neg.index[0]
        msg.append(f"{nice_feature(f)} of {fmt_value(f, res['inputs'][f])} lowers them by "
                   f"{100 * top_neg.iloc[0]:.1f} pts")
    if msg:
        st.caption("Biggest effects: " + "; ".join(msg) + ".")

    # --- How to improve your chances
    st.markdown("##### How to improve your chances")
    if p > 0.90:
        card("Strong profile", "The estimated placement probability is already above 90%. "
             "Keep up the consistent performance and focus on interview preparation.", "action")
    else:
        acts = [a for a in res["what_if"] if a["gain"] > 0.005][:3]
        if not acts:
            card("No single quick win", "No single one-step improvement raises the probability by more "
                 "than 0.5 points. A combination of improvements is needed.")
        cols = st.columns(max(len(acts), 1))
        for col, a in zip(cols, acts):
            with col:
                card(ACTIONS[a["feature"]].format(step=a["step"]),
                     f"Probability {100 * a['proba_before']:.0f}% → <b>{100 * a['proba_after']:.0f}%</b> "
                     f"(+{100 * a['gain']:.1f} pts)", "action")

    # --- Compare to peers
    st.markdown("##### Compare to peers")
    df = load_clean()
    pc = st.columns(2)
    for col, f in zip(pc, ["CGPA", "AptitudeScore"]):
        fig = go.Figure()
        for lab, name, c in [(1, "Placed", PLACED), (0, "Not Placed", NOT_PLACED)]:
            fig.add_trace(go.Violin(x=df.loc[df[config.TARGET] == lab, f].dropna(), name=name,
                                    line_color=c, orientation="h", side="positive", width=1.6,
                                    points=False, meanline_visible=True, hoverinfo="skip"))
        fig.add_vline(x=res["inputs"][f], line_color=AMBER, line_width=3,
                      annotation_text="this student", annotation_position="top")
        fig.update_layout(height=220, showlegend=False, title=nice_feature(f), margin=dict(t=50))
        col.plotly_chart(fig, width="stretch")

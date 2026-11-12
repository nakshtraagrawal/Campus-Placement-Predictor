"""Model comparison: metrics, curves, confusion matrices, statistical test, hyper-parameters."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import PALETTE, card, load_csv, load_metrics, load_report_json, setup_page

ok = setup_page("Models", "Five machine-learning models were tuned with 5-fold cross-validation and tested "
                "once on unseen students. Higher is better for every metric.")
if not ok:
    st.stop()

m = load_metrics()
tr = m["training"]
comp = load_csv("model_comparison.csv")
names = dict(zip(comp["key"], comp["model"]))
colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(tr["models"])}

# ------------------------------------------------------------------ table
st.markdown("##### Metrics on the held-out test set")
cols = {"model": "Model", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mcc": "MCC", "cv_roc_auc_mean": "CV ROC-AUC",
        "cv_roc_auc_std": "CV SD", "train_time_s": "Train time (s)"}
table = comp[list(cols)].rename(columns=cols)
table["Model"] = [f"{n} (selected)" if s else n for n, s in zip(table["Model"], comp["selected"])]
num_cols = [c for c in table.columns if c not in ("Model", "CV SD", "Train time (s)")]
styled = (table.style.format({c: "{:.3f}" for c in num_cols} | {"CV SD": "{:.3f}", "Train time (s)": "{:.1f}"})
          .highlight_max(subset=num_cols, props="background-color:#D1FAE5;font-weight:600"))
st.dataframe(styled, hide_index=True, width="stretch")
st.caption("Green = best value in each column. The model is selected by mean cross-validated ROC-AUC on the "
           "training data (not by the test score), so the test set gives an unbiased final estimate.")

# ------------------------------------------------------------------ why + stats test
best = tr["best_model"]
st_res = tr["statistical_test"]
c1, c2 = st.columns(2, gap="large")
with c1:
    b = tr["models"][best]
    card(f"Why we chose {tr['best_model_name']}",
         f"It has the highest mean 5-fold CV ROC-AUC ({b['cv_roc_auc_mean']:.3f} ± {b['cv_roc_auc_std']:.3f}). "
         f"On the test set it reaches ROC-AUC {b['roc_auc']:.3f}, accuracy {100 * b['accuracy']:.1f}% and "
         f"F1 {b['f1']:.3f}. The F1-optimal threshold found by cross-validation is "
         f"{tr['threshold']['f1_optimal_cv']:.2f} (the app uses the default 0.50).")
with c2:
    card("Is it significantly better than the runner-up?", st_res["interpretation"])

# ------------------------------------------------------------------ curves
curves = load_report_json("curves.json")
c1, c2 = st.columns(2, gap="large")
with c1:
    fig = go.Figure()
    for k, c in curves.items():
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], name=f"{names[k]} ({tr['models'][k]['roc_auc']:.3f})",
                                 line=dict(color=colors[k], width=3 if k == best else 1.8),
                                 hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#94A3B8"), showlegend=False))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate",
                      height=420, legend=dict(x=0.45, y=0.05))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure()
    for k, c in curves.items():
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], name=f"{names[k]} ({tr['models'][k]['pr_auc']:.3f})",
                                 line=dict(color=colors[k], width=3 if k == best else 1.8),
                                 hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>"))
    fig.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision",
                      height=420, legend=dict(x=0.02, y=0.05))
    st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2, gap="large")
with c1:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
    fig = go.Figure()
    for k in comp["key"]:
        fig.add_trace(go.Bar(name=names[k], x=[cols[x] for x in metrics], y=[tr["models"][k][x] for x in metrics],
                             marker_color=colors[k], hovertemplate="%{x}: %{y:.3f}<extra>" + names[k] + "</extra>"))
    fig.update_layout(barmode="group", title="Metric comparison", height=400, yaxis_range=[0, 1],
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure()
    for k, scores in tr["cv_fold_roc_auc"].items():
        fig.add_trace(go.Box(y=scores, name=names[k], marker_color=colors[k], boxpoints="all", jitter=0.3,
                             hovertemplate="Fold ROC-AUC %{y:.4f}<extra></extra>"))
    fig.update_layout(title="Cross-validation ROC-AUC per fold", height=400, showlegend=False,
                      yaxis_title="ROC-AUC")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ confusion matrix
st.markdown("##### Confusion matrix")
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    sel = st.selectbox("Model", list(names), index=list(names).index(best), format_func=lambda k: names[k])
    cm = np.array(tr["confusion_matrices"][sel])
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"- **{tp}** placed students correctly identified\n- **{tn}** not-placed correctly identified\n"
                f"- **{fn}** placed students missed (false negatives)\n- **{fp}** false alarms (false positives)")
with c2:
    norm = cm / cm.sum(axis=1, keepdims=True)
    labels = ["Not Placed", "Placed"]
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1, showscale=False,
                               text=[[f"{100 * norm[i, j]:.1f}%<br>({cm[i, j]})" for j in range(2)] for i in range(2)],
                               texttemplate="%{text}", hovertemplate="Actual %{y}<br>Predicted %{x}<extra></extra>"))
    fig.update_layout(height=330, xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed",
                      title=f"{names[sel]} (row-normalised)")
    st.plotly_chart(fig, width="stretch")

with st.expander("Tuned hyper-parameters of each model"):
    hp = tr["hyperparameters"]
    rows = [{"Model": names[k], "Parameter": p, "Value": str(round(v, 4) if isinstance(v, float) else v)}
            for k, ps in hp.items() for p, v in ps.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"RandomizedSearchCV, {tr['search']['cv_folds']}-fold stratified CV, scoring = ROC-AUC.")

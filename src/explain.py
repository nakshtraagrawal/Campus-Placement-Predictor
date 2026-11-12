"""Step 5: which features influence placement most? (spec §8)

Five independent views of feature importance, then a consensus ranking:
  1. Logistic Regression coefficients (standardised inputs) + odds ratios
  2. Random Forest impurity importance
  3. XGBoost (gain) importance
  4. Permutation importance of the best model on the TEST set (ROC-AUC drop, 20 repeats)
  5. Mean |SHAP| of the best model on the test set
Plus partial dependence plots that show HOW each feature changes the placement probability.
All plain-English insights are generated from the numbers and stored in metrics.json -> insights.
"""
from __future__ import annotations

import itertools

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence, permutation_importance

from src import config
from src.predict import load_background, load_model, make_explainer
from src.preprocess import train_test_frames
from src.utils import get_logger, load_json, save_json, savefig, set_plot_style, update_metrics

log = get_logger("explain")

METHOD_LABELS = {
    "lr_coef": "LR |coefficient|",
    "rf_impurity": "RF impurity",
    "xgb_gain": "XGBoost gain",
    "permutation": "Permutation (best model)",
    "shap": "Mean |SHAP| (best model)",
}


def _load(name):
    return joblib.load(config.ALL_MODELS_DIR / f"{name}.joblib")


def lr_coefficients() -> pd.DataFrame:
    pipe = _load("logistic_regression")
    coef = pipe.named_steps["model"].coef_.ravel()
    names = pipe[:-1].get_feature_names_out()
    d = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    d = d.sort_values("coef")
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [config.COLOR_PLACED if c > 0 else config.COLOR_NOT_PLACED for c in d["coef"]]
    ax.barh(d["feature"], d["coef"], color=colors)
    span = d["coef"].abs().max()
    for i, (c, o) in enumerate(zip(d["coef"], d["odds_ratio"])):
        ax.text(c + (0.02 if c >= 0 else -0.02) * span * 5, i, f"OR {o:.2f}",
                va="center", ha="left" if c >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#0F172A", lw=0.8)
    ax.set_xlim(min(0, d["coef"].min()) - 0.35 * span, d["coef"].max() + 0.35 * span)
    ax.set_xlabel("Coefficient (per 1 SD increase, log-odds)")
    ax.set_title("Logistic Regression coefficients and odds ratios", loc="left")
    savefig(fig, "lr_coefficients.png")
    return d.sort_values("coef", key=abs, ascending=False).reset_index(drop=True)


def tree_importances() -> dict[str, pd.Series]:
    out = {}
    for key, name in [("rf_impurity", "random_forest"), ("xgb_gain", "xgboost")]:
        pipe = _load(name)
        out[key] = pd.Series(pipe.named_steps["model"].feature_importances_,
                             index=pipe[:-1].get_feature_names_out())
    return out


def permutation(pipe, X_test, y_test) -> pd.DataFrame:
    r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=20,
                               random_state=config.RANDOM_STATE, n_jobs=-1)
    d = pd.DataFrame({"feature": X_test.columns, "mean": r.importances_mean, "std": r.importances_std})
    d = d.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(d["feature"], d["mean"], xerr=d["std"], color=config.MODEL_PALETTE[0], capsize=3)
    ax.set_xlabel("Mean drop in test ROC-AUC when the feature is shuffled (20 repeats, ±1 SD)")
    ax.set_title("Permutation importance (best model, test set)", loc="left")
    savefig(fig, "permutation_importance.png")
    return d.sort_values("mean", ascending=False).reset_index(drop=True)


def shap_analysis(pipe, meta, X_test) -> tuple[pd.Series, list[str]]:
    expl = make_explainer(pipe, load_background(), meta.get("is_tree_model", False))
    n = 500 if expl.kind == "tree" else 200  # KernelExplainer is slow -> 200-row sample (spec)
    Xs = X_test.sample(min(n, len(X_test)), random_state=config.RANDOM_STATE)
    sv, base = expl.explain(Xs, nsamples=300)
    # Show raw (imputed, unscaled) values in the plots when the feature space allows it
    Xdisp = pipe[0].transform(Xs)
    if list(Xdisp.columns) != expl.feature_names:
        Xdisp = expl.transformed(Xs)
    imp = pd.Series(np.abs(sv).mean(0), index=expl.feature_names).sort_values(ascending=False)

    shap.summary_plot(sv, Xdisp, show=False, plot_size=(7, 4.5))
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (change in placement probability)")
    fig.axes[0].set_title("SHAP summary - best model (test sample)", loc="left")
    savefig(fig, "shap_summary_beeswarm.png")

    shap.summary_plot(sv, Xdisp, plot_type="bar", show=False, plot_size=(7, 4.5),
                      color=config.MODEL_PALETTE[0])
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Mean |SHAP value| (probability units)")
    fig.axes[0].set_title("Global SHAP importance", loc="left")
    savefig(fig, "shap_bar.png")

    top3 = list(imp.index[:3])
    for f in top3:
        shap.dependence_plot(f, sv, Xdisp, interaction_index=None, show=False,
                             color=config.MODEL_PALETTE[0], alpha=0.5)
        fig = plt.gcf()
        fig.set_size_inches(5.5, 4)
        fig.axes[0].set_ylabel(f"SHAP value for {f}")
        fig.axes[0].set_title(f"SHAP dependence - {f}", loc="left")
        savefig(fig, f"shap_dependence_{f}.png")
    update_metrics("shap", {"base_value": base, "explainer": expl.kind, "n_explained": len(Xs),
                            "mean_abs": imp.to_dict(), "top3": top3})
    return imp, top3


def pdp(pipe, X_train, meta) -> dict:
    """Partial dependence for every feature (json for the app), figure for the top 3."""
    Xs = X_train.dropna().sample(1000, random_state=config.RANDOM_STATE)
    out = {}
    for f in meta["features"]:
        r = partial_dependence(pipe, Xs, [f], kind="average", grid_resolution=25,
                               percentiles=(0.02, 0.98))
        out[f] = {"grid": r["grid_values"][0], "avg_proba": r["average"][0]}
    return out


def plot_pdp(pdp_data: dict, feats: list[str]) -> None:
    fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 3.6), sharey=True)
    for ax, f in zip(np.atleast_1d(axes), feats):
        g, p = np.array(pdp_data[f]["grid"]), 100 * np.array(pdp_data[f]["avg_proba"])
        ax.plot(g, p, marker="o", ms=3, color=config.MODEL_PALETTE[0])
        ax.set_title(f, loc="left")
        ax.set_xlabel(f)
    np.atleast_1d(axes)[0].set_ylabel("Avg. predicted placement probability (%)")
    fig.suptitle("Partial dependence of the best model on the top-3 consensus features", x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "pdp_top3.png")


def consensus(methods: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = config.CORE_FEATURES
    norm, ranks = {}, {}
    for m, s in methods.items():
        s = s.reindex(feats).clip(lower=0).fillna(0)
        norm[m] = s / s.max() if s.max() > 0 else s
        ranks[m] = s.rank(ascending=False, method="min")
    d = pd.DataFrame({"feature": feats})
    for m in methods:
        d[f"{m}_norm"] = norm[m].values
    for m in methods:
        d[f"{m}_rank"] = ranks[m].values.astype(int)
    d["mean_rank"] = d[[f"{m}_rank" for m in methods]].mean(1)
    d["consensus_score"] = d[[f"{m}_norm" for m in methods]].mean(1)
    d = d.sort_values(["mean_rank", "consensus_score"], ascending=[True, False]).reset_index(drop=True)
    d.insert(1, "consensus_rank", np.arange(1, len(d) + 1))
    d["simulated"] = d["feature"].isin(config.SIMULATED_FEATURES)
    d.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)

    agree = pd.DataFrame(index=list(methods), columns=list(methods), dtype=float)
    for a, b in itertools.product(methods, methods):
        agree.loc[a, b] = spearmanr(norm[a], norm[b]).statistic
    agree.to_csv(config.METHOD_AGREEMENT_CSV)

    # Figures: consensus bar + per-method comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    dd = d.iloc[::-1]
    colors = ["#94A3B8" if s else config.MODEL_PALETTE[0] for s in dd["simulated"]]
    ax.barh(dd["feature"], dd["consensus_score"], color=colors)
    for i, (sc, r) in enumerate(zip(dd["consensus_score"], dd["mean_rank"])):
        ax.text(sc + 0.01, i, f"mean rank {r:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Consensus score (mean of normalised importances)")
    ax.set_title("Consensus feature influence (grey = simulated column)", loc="left")
    savefig(fig, "feature_importance_consensus.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = d["feature"].tolist()[::-1]
    h = 0.8 / len(methods)
    y = np.arange(len(order))
    for i, (m, color) in enumerate(zip(methods, config.MODEL_PALETTE)):
        ax.barh(y + i * h - 0.4 + h / 2, norm[m].reindex(order), h, label=METHOD_LABELS[m], color=color)
    ax.set_yticks(y, order)
    ax.set_xlabel("Normalised importance (1 = most important for that method)")
    ax.set_title("Feature importance by method", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "importance_comparison.png")
    return d, agree


def _fmt(f, v):
    if f in ("CGPA", "CommunicationSkills"):
        return f"{v:.1f}"
    if f == "Attendance":
        return f"{v:.0f}%"
    return f"{v:.0f}"


def _agreement_word(rho: float) -> str:
    return "strongly" if rho >= 0.8 else "moderately" if rho >= 0.5 else "only weakly"


def build_insights(cons, agree, lr, perm, pdp_data, metrics) -> list[str]:
    ins = []
    top = cons.iloc[0]
    n_methods = len(METHOD_LABELS)
    n_first = int(sum(top[f"{m}_rank"] == 1 for m in METHOD_LABELS))
    ins.append(f"{top['feature']} is the most influential feature overall (consensus rank 1; ranked first by "
               f"{n_first} of {n_methods} methods).")
    top3 = cons["feature"].head(3).tolist()
    ins.append(f"The three most influential features are {top3[0]}, {top3[1]} and {top3[2]}.")
    for f in top3:
        g, p = np.array(pdp_data[f]["grid"]), np.array(pdp_data[f]["avg_proba"])
        if len(g) < 2:
            continue
        k = int(np.argmax(np.abs(np.diff(p))))  # grid step with the largest probability change
        direction = "rises" if p[-1] > p[0] else "falls"
        ins.append(f"Average placement probability {direction} from {100 * p[0]:.0f}% at {f} = {_fmt(f, g[0])} "
                   f"to {100 * p[-1]:.0f}% at {f} = {_fmt(f, g[-1])}; the steepest change is between "
                   f"{_fmt(f, g[k])} and {_fmt(f, g[k + 1])}.")
    lr_top = lr.iloc[0]
    ins.append(f"In the logistic regression model, a one-standard-deviation increase in {lr_top['feature']} "
               f"multiplies the odds of placement by {lr_top['odds_ratio']:.2f}.")
    neg = lr[lr["coef"] < 0]
    if not neg.empty:
        verb = "has a negative coefficient" if len(neg) == 1 else "have negative coefficients"
        ins.append(f"{', '.join(neg['feature'])} {verb} in the logistic model: holding the other features "
                   f"fixed, higher values slightly reduce the odds of placement (odds ratio "
                   f"{', '.join(f'{o:.2f}' for o in neg['odds_ratio'])}).")
    vals = agree.to_numpy()
    mean_rho = vals[np.triu_indices_from(vals, k=1)].mean()
    ins.append(f"The five importance methods agree {_agreement_word(mean_rho)} on the ranking: mean pairwise "
               f"Spearman rank correlation = {mean_rho:.2f}.")
    low = cons.iloc[-1]
    ins.append(f"{low['feature']} is the least influential feature (consensus rank {len(cons)}).")
    sim = cons[cons["simulated"]]
    abl = {r["variant"]: r for r in metrics["training"]["ablation"]}
    d_sim = [r for k, r in abl.items() if k.startswith("(d)")][0]["delta_vs_core"]
    effect = "only a small" if abs(d_sim) < 0.01 else "a noticeable"
    ins.append("The simulated columns rank " + " and ".join(
        f"{r.feature} #{r.consensus_rank}" for r in sim.itertuples()) +
        f" (their influence reflects the simulation formula, not real evidence); removing both has "
        f"{effect} effect on CV ROC-AUC ({d_sim:+.4f}).")
    d_extra = [r for k, r in abl.items() if k.startswith("(c)")][0]["delta_vs_core"]
    ins.append(f"Adding the extra Kaggle columns (extracurriculars, placement training, SSC/HSC marks) changes "
               f"CV ROC-AUC by {d_extra:+.4f}; they are left out of the main model because the problem "
               f"statement defines the 8 core features.")
    eng = metrics["training"]["engineered_features"]
    ins.append(f"Engineered features changed CV ROC-AUC by {eng['cv_gain']:+.4f} "
               f"({'used' if eng['used'] else 'not used'}; rule: gain >= 0.005).")
    return ins


def main() -> dict:
    set_plot_style()
    pipe, meta = load_model()
    X_train, X_test, y_train, y_test = train_test_frames()
    feats = meta["features"]
    X_train, X_test = X_train[feats], X_test[feats]

    lr = lr_coefficients()
    trees = tree_importances()
    perm = permutation(pipe, X_test, y_test)
    shap_imp, shap_top3 = shap_analysis(pipe, meta, X_test)

    methods = {
        "lr_coef": lr.set_index("feature")["coef"].abs(),
        "rf_impurity": trees["rf_impurity"],
        "xgb_gain": trees["xgb_gain"],
        "permutation": perm.set_index("feature")["mean"],
        "shap": shap_imp,
    }
    cons, agree = consensus(methods)
    pdp_data = pdp(pipe, X_train, meta)
    save_json(pdp_data, config.PDP_JSON)
    plot_pdp(pdp_data, cons["feature"].head(3).tolist())

    metrics = load_json(config.METRICS_JSON)
    insights = build_insights(cons, agree, lr, perm, pdp_data, metrics)
    vals = agree.to_numpy()
    update_metrics("explainability", {
        "methods": METHOD_LABELS,
        "consensus_ranking": cons["feature"].tolist(),
        "top3": cons["feature"].head(3).tolist(),
        "mean_pairwise_spearman": float(vals[np.triu_indices_from(vals, k=1)].mean()),
        "lr_odds_ratios": lr.set_index("feature")["odds_ratio"].to_dict(),
        "permutation": perm.set_index("feature")[["mean", "std"]].to_dict(orient="index"),
    })
    update_metrics("insights", insights)
    for s in insights:
        log.info("Insight: %s", s)
    return {"consensus": cons, "insights": insights}


if __name__ == "__main__":
    main()

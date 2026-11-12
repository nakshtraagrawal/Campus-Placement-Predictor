"""Step 4: train, tune and evaluate five models (spec §6-7).

* Every model is a full sklearn Pipeline (imputer -> [scaler] -> model).
* Hyper-parameters: RandomizedSearchCV, 5-fold StratifiedKFold, scoring = ROC-AUC, training set only.
* The best model is chosen by MEAN CV ROC-AUC (never by the test score - choosing on the test
  set would make the test estimate optimistic). The test set is used exactly once at the end.
"""
from __future__ import annotations

import datetime as dt
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, learning_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar

from src import config
from src.features import build_pipeline
from src.preprocess import train_test_frames
from src.utils import get_logger, save_json, savefig, set_plot_style, update_metrics

log = get_logger("train")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # spec fallback
    HAS_XGB = False

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "svm": "SVM (RBF)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost" if HAS_XGB else "HistGradientBoosting",
}
TREE_MODELS = {"random_forest", "xgboost"}


def model_specs(class_weight):
    """(estimator, search space, n_iter) per model. Grids are small to keep runtime < 5 min."""
    rs = config.RANDOM_STATE
    specs = {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, class_weight=class_weight, random_state=rs),
            {"model__C": loguniform(1e-3, 1e2), "model__penalty": ["l1", "l2"],
             "model__solver": ["liblinear"]},
            config.N_ITER_SEARCH),
        "knn": (
            KNeighborsClassifier(),
            {"model__n_neighbors": randint(5, 151), "model__weights": ["uniform", "distance"],
             "model__p": [1, 2]},
            config.N_ITER_SEARCH),
        # probability=False during the search (ROC-AUC uses decision_function, ~5x faster);
        # the final model is refitted with probability=True so the app can show probabilities.
        "svm": (
            SVC(kernel="rbf", probability=False, class_weight=class_weight, random_state=rs),
            {"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-3, 1e0)},
            10),
        "random_forest": (
            RandomForestClassifier(class_weight=class_weight, random_state=rs, n_jobs=1),
            {"model__n_estimators": randint(150, 400), "model__max_depth": [4, 6, 8, 10, 12, None],
             "model__min_samples_leaf": randint(1, 30), "model__max_features": ["sqrt", "log2", 0.5]},
            config.N_ITER_SEARCH),
    }
    if HAS_XGB:
        specs["xgboost"] = (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          random_state=rs, n_jobs=1, verbosity=0),
            {"model__n_estimators": randint(100, 500), "model__max_depth": randint(2, 7),
             "model__learning_rate": loguniform(0.01, 0.3), "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.6, 0.4), "model__min_child_weight": randint(1, 20),
             "model__reg_lambda": loguniform(0.1, 10)},
            config.N_ITER_SEARCH)
    else:
        specs["xgboost"] = (
            HistGradientBoostingClassifier(random_state=rs),
            {"model__learning_rate": loguniform(0.01, 0.3), "model__max_depth": randint(2, 8),
             "model__max_iter": randint(100, 500), "model__min_samples_leaf": randint(10, 100)},
            config.N_ITER_SEARCH)
    return specs


def cv_splitter() -> StratifiedKFold:
    # The SAME folds for every model -> fold scores are paired for the statistical test
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def test_metrics(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
        "mcc": matthews_corrcoef(y_true, pred), "brier": brier_score_loss(y_true, proba),
    }


def tune_model(name, estimator, space, n_iter, X, y, features):
    pipe = build_pipeline(name, estimator, features)
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv_splitter(),
                                scoring={"roc_auc": "roc_auc", "f1": "f1"}, refit="roc_auc",
                                n_jobs=-1, random_state=config.RANDOM_STATE)
    t0 = time.perf_counter()
    search.fit(X[features], y)
    best = search.best_estimator_
    if name == "svm":  # refit with probability estimates for the app / ROC curves
        best = clone(best).set_params(model__probability=True).fit(X[features], y)
    train_time = time.perf_counter() - t0
    i = search.best_index_
    res = search.cv_results_
    fold_auc = [res[f"split{k}_test_roc_auc"][i] for k in range(config.CV_FOLDS)]
    fold_f1 = [res[f"split{k}_test_f1"][i] for k in range(config.CV_FOLDS)]
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return best, fold_auc, fold_f1, params, train_time


def paired_tests(name_a, name_b, folds_a, folds_b, y_test, pred_a, pred_b, n_train) -> dict:
    """Paired t-test on the 5 CV folds (+ Nadeau-Bengio correction) and McNemar on the test set."""
    diff = np.array(folds_a) - np.array(folds_b)
    t_stat, p_t = stats.ttest_rel(folds_a, folds_b)
    # Corrected resampled t-test (Nadeau & Bengio 2003): folds share training data, so the
    # naive variance is too small. Correction factor (1/k + n_test/n_train).
    k = len(diff)
    n_val = n_train / k
    var = diff.var(ddof=1) * (1 / k + n_val / (n_train - n_val))
    t_corr = diff.mean() / np.sqrt(var) if var > 0 else np.inf
    p_corr = 2 * stats.t.sf(abs(t_corr), df=k - 1)
    # McNemar: do the two models make errors on different test students?
    a_ok, b_ok = pred_a == y_test, pred_b == y_test
    table = [[int((a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())],
             [int((~a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum())]]
    mc = mcnemar(table, exact=False, correction=True)
    sig = p_corr < 0.05
    text = (f"{DISPLAY_NAMES[name_a]} vs {DISPLAY_NAMES[name_b]}: mean CV ROC-AUC difference "
            f"{diff.mean():+.4f}. Paired t-test p = {p_t:.3f} (corrected p = {p_corr:.3f}); "
            f"McNemar test on the test set p = {mc.pvalue:.3f}. ")
    text += ("The difference is statistically significant at the 5% level."
             if sig else "The difference is NOT statistically significant at the 5% level, so the "
             "two models perform practically the same on this data.")
    return {"model_a": name_a, "model_b": name_b, "mean_diff_cv_roc_auc": diff.mean(),
            "ttest_t": t_stat, "ttest_p": p_t, "corrected_ttest_t": t_corr,
            "corrected_ttest_p": p_corr, "mcnemar_table": table, "mcnemar_stat": mc.statistic,
            "mcnemar_p": mc.pvalue, "significant_at_0.05": bool(sig), "interpretation": text}


def ablation(best_name, tuned_estimator, X, y) -> pd.DataFrame:
    """CV ROC-AUC of the best model type (with its tuned hyper-parameters) on different feature sets (spec §7.4)."""
    est = clone(tuned_estimator)
    core_minus_sim = [f for f in config.CORE_FEATURES if f not in config.SIMULATED_FEATURES]
    variants = [
        ("(a) 8 core features", config.CORE_FEATURES, False),
        ("(b) core + engineered", config.CORE_FEATURES, True),
        ("(c) core + extra Kaggle columns", config.CORE_FEATURES + config.EXTRA_FEATURES, False),
        ("(d) core - simulated columns", core_minus_sim, False),
    ]
    rows = []
    for label, feats, eng in variants:
        pipe = build_pipeline(best_name, clone(est), feats, engineered=eng)
        cv = cross_validate(pipe, X[feats], y, cv=cv_splitter(), scoring="roc_auc", n_jobs=-1)
        n_feat = len(feats) + (len(config.ENGINEERED_FEATURES) if eng else 0)
        rows.append({"variant": label, "n_features": n_feat,
                     "cv_roc_auc_mean": cv["test_score"].mean(), "cv_roc_auc_std": cv["test_score"].std()})
    df = pd.DataFrame(rows)
    df["delta_vs_core"] = df["cv_roc_auc_mean"] - df.loc[0, "cv_roc_auc_mean"]
    df.to_csv(config.ABLATION_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_curves(y_test, probas: dict) -> dict:
    curves = {}
    fig_r, ax_r = plt.subplots(figsize=(5.5, 5))
    fig_p, ax_p = plt.subplots(figsize=(5.5, 5))
    for (name, p), color in zip(probas.items(), config.MODEL_PALETTE):
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        auc, ap = roc_auc_score(y_test, p), average_precision_score(y_test, p)
        ax_r.plot(fpr, tpr, color=color, lw=1.6, label=f"{DISPLAY_NAMES[name]} (AUC={auc:.3f})")
        ax_p.plot(rec, prec, color=color, lw=1.6, label=f"{DISPLAY_NAMES[name]} (AP={ap:.3f})")
        # down-sample curve points for the app
        sel_r = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)
        sel_p = np.linspace(0, len(rec) - 1, min(len(rec), 200)).astype(int)
        curves[name] = {"fpr": fpr[sel_r], "tpr": tpr[sel_r], "precision": prec[sel_p], "recall": rec[sel_p]}
    ax_r.plot([0, 1], [0, 1], ls="--", color="#94A3B8", lw=1)
    ax_r.set(xlabel="False positive rate", ylabel="True positive rate")
    ax_r.set_title("ROC curves (test set)", loc="left")
    ax_r.legend(loc="lower right", fontsize=8)
    base = float(np.mean(y_test))
    ax_p.axhline(base, ls="--", color="#94A3B8", lw=1, label=f"Chance ({base:.2f})")
    ax_p.set(xlabel="Recall", ylabel="Precision")
    ax_p.set_title("Precision-recall curves (test set)", loc="left")
    ax_p.legend(loc="lower left", fontsize=8)
    savefig(fig_r, "roc_curves.png")
    savefig(fig_p, "pr_curves.png")
    return curves


def plot_confusion_matrices(cms: dict) -> None:
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4))
    for ax, (name, cm) in zip(axes, cms.items()):
        cm = np.array(cm)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "#0F172A", fontsize=9)
        ax.set_xticks([0, 1], ["Not Placed", "Placed"])
        ax.set_yticks([0, 1], ["Not Placed", "Placed"])
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("Actual")
        ax.set_title(DISPLAY_NAMES[name], fontsize=10)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test set (row-normalised, counts in brackets)", x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "confusion_matrices.png")


def plot_comparison_bar(comp: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(comp)
    x = np.arange(len(metrics))
    for i, (_, row) in enumerate(comp.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, row[metrics].astype(float), w, label=row["model"],
               color=config.MODEL_PALETTE[i % len(config.MODEL_PALETTE)])
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison on the held-out test set", loc="left")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    savefig(fig, "model_comparison_bar.png")


def plot_cv_box(folds: dict) -> None:
    d = pd.DataFrame([(DISPLAY_NAMES[m], s) for m, v in folds.items() for s in v], columns=["model", "auc"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    import seaborn as sns
    sns.boxplot(data=d, x="model", y="auc", hue="model", palette=config.MODEL_PALETTE[:len(folds)],
                ax=ax, width=0.5, legend=False)
    sns.stripplot(data=d, x="model", y="auc", color="#0F172A", size=4, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("ROC-AUC per fold")
    ax.set_title("5-fold cross-validation ROC-AUC (training set)", loc="left")
    savefig(fig, "cv_boxplot.png")


def plot_calibration(name, y_test, proba) -> None:
    frac, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color="#94A3B8", label="Perfectly calibrated")
    ax.plot(mean_pred, frac, marker="o", color=config.MODEL_PALETTE[0], label=DISPLAY_NAMES[name])
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction placed")
    ax.set_title(f"Calibration curve - {DISPLAY_NAMES[name]} (test set)", loc="left")
    ax.legend()
    savefig(fig, "calibration_curve.png")


def plot_learning_curve(name, pipe, X, y) -> dict:
    sizes, tr, va = learning_curve(pipe, X, y, cv=cv_splitter(), scoring="roc_auc", n_jobs=-1,
                                   train_sizes=np.linspace(0.1, 1.0, 6), random_state=config.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for scores, label, c in [(tr, "Training", config.MODEL_PALETTE[0]), (va, "Validation (CV)", config.MODEL_PALETTE[1])]:
        m, s = scores.mean(1), scores.std(1)
        ax.plot(sizes, m, marker="o", color=c, label=label)
        ax.fill_between(sizes, m - s, m + s, color=c, alpha=0.15)
    ax.set(xlabel="Training examples", ylabel="ROC-AUC")
    ax.set_title(f"Learning curve - {DISPLAY_NAMES[name]}", loc="left")
    ax.legend()
    savefig(fig, "learning_curve.png")
    return {"train_sizes": sizes, "train_mean": tr.mean(1), "val_mean": va.mean(1),
            "final_gap": float(tr.mean(1)[-1] - va.mean(1)[-1])}


# ---------------------------------------------------------------------------
def main() -> dict:
    set_plot_style()
    X_train, X_test, y_train, y_test = train_test_frames()
    feats = config.CORE_FEATURES

    minority_share = float(min(y_train.mean(), 1 - y_train.mean()))
    class_weight = "balanced" if minority_share < 0.40 else None
    log.info("Minority class share %.1f%% -> class_weight=%s", 100 * minority_share, class_weight)
    specs = model_specs(class_weight)

    fitted, folds_auc, folds_f1, params, times, probas = {}, {}, {}, {}, {}, {}
    rows, cms = [], {}
    for name, (est, space, n_iter) in specs.items():
        log.info("Tuning %s (%d candidates x %d folds)...", DISPLAY_NAMES[name], n_iter, config.CV_FOLDS)
        best, fa, ff, p, t = tune_model(name, est, space, n_iter, X_train, y_train, feats)
        fitted[name], folds_auc[name], folds_f1[name], params[name], times[name] = best, fa, ff, p, t
        proba = best.predict_proba(X_test[feats])[:, 1]
        probas[name] = proba
        m = test_metrics(y_test, proba)
        cms[name] = confusion_matrix(y_test, (proba >= 0.5).astype(int)).tolist()
        rows.append({"key": name, "model": DISPLAY_NAMES[name], **m,
                     "cv_roc_auc_mean": np.mean(fa), "cv_roc_auc_std": np.std(fa),
                     "cv_f1_mean": np.mean(ff), "cv_f1_std": np.std(ff), "train_time_s": t})
        joblib.dump(best, config.ALL_MODELS_DIR / f"{name}.joblib", compress=3)
        log.info("  %-20s CV AUC %.4f +- %.4f | test AUC %.4f | %.1fs",
                 DISPLAY_NAMES[name], np.mean(fa), np.std(fa), m["roc_auc"], t)

    comp = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)
    best_name = comp.loc[0, "key"]
    second_name = comp.loc[1, "key"]
    comp["selected"] = comp["key"] == best_name
    comp.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    pd.DataFrame(folds_auc).to_csv(config.CV_SCORES_CSV, index_label="fold")
    log.info("Best model by mean CV ROC-AUC: %s", DISPLAY_NAMES[best_name])

    # --- Statistical comparison of the top two models
    test_res = paired_tests(best_name, second_name, folds_auc[best_name], folds_auc[second_name],
                            y_test.to_numpy(), (probas[best_name] >= 0.5).astype(int),
                            (probas[second_name] >= 0.5).astype(int), len(X_train))

    # --- F1-optimal threshold from out-of-fold CV predictions (never from the test set)
    oof = cross_val_predict(clone(fitted[best_name]), X_train[feats], y_train, cv=cv_splitter(),
                            method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    f1_thr = float(thr[np.argmax(f1s)])
    test_at_f1_thr = test_metrics(y_test, probas[best_name], f1_thr)

    # --- Ablation + engineered-feature decision (needs >= 0.005 CV ROC-AUC gain)
    abl = ablation(best_name, fitted[best_name].named_steps["model"], X_train, y_train)
    eng_gain = float(abl.loc[1, "delta_vs_core"])
    use_engineered = eng_gain >= 0.005
    final_pipe = fitted[best_name]
    if use_engineered:
        log.info("Engineered features improve CV ROC-AUC by %.4f -> refitting with them", eng_gain)
        final_pipe = build_pipeline(best_name, clone(fitted[best_name].named_steps["model"]), feats, engineered=True)
        final_pipe.fit(X_train[feats], y_train)
        probas[best_name] = final_pipe.predict_proba(X_test[feats])[:, 1]
    joblib.dump(final_pipe, config.BEST_MODEL_PATH, compress=3)

    # --- Figures
    curves = plot_curves(y_test, probas)
    save_json(curves, config.CURVES_JSON)
    plot_confusion_matrices(cms)
    plot_comparison_bar(comp)
    plot_cv_box(folds_auc)
    plot_calibration(best_name, y_test, probas[best_name])
    lc = plot_learning_curve(best_name, clone(final_pipe), X_train[feats], y_train)

    # --- model_meta.json (used by the app and predict.py)
    clean = pd.concat([X_train, X_test])
    ranges = {}
    for f in feats:
        lo, hi = config.FEATURE_RANGES[f]
        ranges[f] = {"min": lo, "max": hi, "default": float(clean[f].median()),
                     "integer": f in config.INTEGER_FEATURES,
                     "data_min": float(clean[f].min()), "data_max": float(clean[f].max())}
    best_row = comp.loc[0]
    meta = {
        "model_key": best_name, "model_name": DISPLAY_NAMES[best_name],
        "features": feats, "uses_engineered_features": use_engineered,
        "feature_ranges": ranges, "threshold": 0.5, "f1_optimal_threshold_cv": f1_thr,
        "test_roc_auc": float(roc_auc_score(y_test, probas[best_name])),
        "cv_roc_auc_mean": float(best_row["cv_roc_auc_mean"]),
        "trained_on": dt.datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "random_state": config.RANDOM_STATE,
        "is_tree_model": best_name in TREE_MODELS,
    }
    save_json(meta, config.MODEL_META_PATH)

    # Background sample for SHAP explanations in the app
    X_train[feats].sample(200, random_state=config.RANDOM_STATE).to_csv(
        config.MODELS_DIR / "shap_background.csv", index=False)

    results = {
        "best_model": best_name, "best_model_name": DISPLAY_NAMES[best_name],
        "selection_criterion": "mean 5-fold CV ROC-AUC on the training set",
        "class_balance": {"minority_share_train": minority_share, "class_weight": class_weight},
        "search": {"method": "RandomizedSearchCV", "cv_folds": config.CV_FOLDS,
                   "n_iter": {k: v[2] for k, v in specs.items()}, "scoring": "roc_auc"},
        "models": {r["key"]: {k: v for k, v in r.items() if k != "key"} for r in rows},
        "hyperparameters": params,
        "confusion_matrices": cms,
        "cv_fold_roc_auc": folds_auc,
        "statistical_test": test_res,
        "threshold": {"default": 0.5, "f1_optimal_cv": f1_thr, "best_model_test_at_f1_threshold": test_at_f1_thr},
        "ablation": abl.to_dict(orient="records"),
        "engineered_features": {"cv_gain": eng_gain, "used": use_engineered, "rule": ">= 0.005 CV ROC-AUC"},
        "learning_curve": lc,
        "xgboost_available": HAS_XGB,
    }
    update_metrics("training", results)
    return results


if __name__ == "__main__":
    main()

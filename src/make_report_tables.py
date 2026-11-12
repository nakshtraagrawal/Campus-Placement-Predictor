"""Generate LaTeX tables + number macros for the IEEE report from the pipeline outputs.

Nothing in the report is typed by hand: tables are written to report/tables/*.tex and every number
that appears in running text is a macro defined in report/tables/macros.tex (e.g. \\BestTestAUC).
Also draws the system-architecture diagram (reports/figures/architecture.png).
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config
from src.utils import get_logger, load_json, savefig, set_plot_style

log = get_logger("report_tables")
T = config.REPORT_TABLE_DIR


def esc(s) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def write(name: str, text: str) -> None:
    T.mkdir(parents=True, exist_ok=True)
    (T / name).write_text(text, encoding="utf-8")


def tabular(df: pd.DataFrame, colspec: str, header: list[str], size=r"\footnotesize") -> str:
    lines = [size, r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def f3(x) -> str:
    return f"{x:.3f}"


# ---------------------------------------------------------------------------
def table_attributes() -> None:
    summ = pd.read_csv(config.DATA_SUMMARY_CSV, index_col="feature")
    types = {"CGPA": "float", "CommunicationSkills": "float", "Attendance": "float (\\%)"}
    rows = []
    for f in config.CORE_FEATURES:
        lo, hi = config.FEATURE_RANGES[f]
        rows.append([esc(f), types.get(f, "int"), f"{lo:g}--{hi:g}",
                     f"{summ.loc[f, 'min']:.1f}--{summ.loc[f, 'max']:.1f}",
                     f"{summ.loc[f, 'mean']:.2f} $\\pm$ {summ.loc[f, 'std']:.2f}",
                     f"{summ.loc[f, 'missing_pct']:.1f}",
                     "Simulated" if f in config.SIMULATED_FEATURES else "Real"])
    for f in config.EXTRA_FEATURES:
        rows.append([esc(f), "binary" if f in config.BINARY_CATEGORICALS else "int",
                     "0/1" if f in config.BINARY_CATEGORICALS else "0--100",
                     f"{summ.loc[f, 'min']:.0f}--{summ.loc[f, 'max']:.0f}",
                     f"{summ.loc[f, 'mean']:.2f} $\\pm$ {summ.loc[f, 'std']:.2f}",
                     f"{summ.loc[f, 'missing_pct']:.1f}", "Real (ablation)"])
    df = pd.DataFrame(rows)
    write("attributes.tex", tabular(df, "lllllrl",
          ["Attribute", "Type", "Valid", "Observed", "Mean $\\pm$ SD", "Miss.\\%", "Source"], r"\scriptsize"))


def table_preprocessing(plog: dict) -> None:
    inj, cl = plog["injection_log"], plog["cleaning_log"]
    rows = [
        ["Rows in augmented file", "--", cl["rows_in"]],
        ["Whitespace/type noise fixed", sum(inj["type_noise_strings"].values()), cl["type_noise_whitespace_fixed"]],
        ["Target labels normalised", inj["inconsistent_labels"], cl["labels_normalised"]],
        ["Yes/No categoricals normalised", sum(inj["inconsistent_categoricals"].values()),
         sum(cl["categoricals_normalised"].values())],
        ["Exact duplicates removed", inj["duplicate_rows_added"], cl["duplicates_removed"]],
        ["CGPA percentage $\\rightarrow$ CGPA", inj["invalid_values"].get("cgpa_percentage", 0),
         cl["cgpa_percentage_converted"]],
        ["Out-of-range values $\\rightarrow$ NaN", inj["invalid_values_total"] - inj["invalid_values"].get("cgpa_percentage", 0),
         cl["invalid_to_nan_total"]],
        ["Outliers capped (IQR, 1.5$\\times$)", "--", cl["outliers_capped_total"]],
        ["Missing cells imputed (median, in pipeline)", sum(inj["missing_values"].values()),
         cl["missing_after_cleaning_total"]],
        ["Rows after cleaning", "--", cl["rows_out"]],
        ["Train / test rows (stratified 80/20)", "--", f"{cl['train_rows']} / {cl['test_rows']}"],
    ]
    write("preprocessing.tex", tabular(pd.DataFrame(rows), "lrr", ["Step", "Injected", "Fixed / result"]))


def table_vif_fs() -> None:
    vif = pd.read_csv(config.VIF_CSV)
    fs = pd.read_csv(config.FEATURE_SELECTION_CSV).set_index("feature")
    rows = []
    for r in vif.itertuples():
        s = fs.loc[r.feature]
        rows.append([esc(r.feature), f"{s['mutual_info']:.3f} ({int(s['rank_mutual_info'])})",
                     f"{s['anova_f']:.0f} ({int(s['rank_anova_f'])})",
                     f"{s['chi2']:.1f} ({int(s['rank_chi2'])})", int(s["rfe_rank"]), f"{r.VIF:.2f}"])
    df = pd.DataFrame(rows).sort_values(1, key=lambda c: c.str.extract(r"\((\d+)\)")[0].astype(int))
    write("feature_selection.tex", tabular(df, "lrrrrr",
          ["Feature", "MI (rank)", "ANOVA F (rank)", "$\\chi^2$ (rank)", "RFE", "VIF"]))


def table_models(m: dict) -> None:
    comp = pd.read_csv(config.MODEL_COMPARISON_CSV)
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
    best = {c: comp[c].max() for c in cols + ["cv_roc_auc_mean"]}
    rows = []
    for r in comp.itertuples():
        name = esc(r.model) + (r"$^\star$" if r.selected else "")
        vals = [rf"\textbf{{{f3(getattr(r, c))}}}" if getattr(r, c) == best[c] else f3(getattr(r, c)) for c in cols]
        cv = f"{r.cv_roc_auc_mean:.3f}$\\pm${r.cv_roc_auc_std:.3f}"
        if r.cv_roc_auc_mean == best["cv_roc_auc_mean"]:
            cv = r"\textbf{" + cv + "}"
        rows.append([name, *vals, cv, f"{r.train_time_s:.0f}"])
    write("model_comparison.tex", tabular(pd.DataFrame(rows), "lrrrrrrrrr",
          ["Model", "Acc", "Prec", "Rec", "F1", "ROC", "PR", "MCC", "CV ROC-AUC", "Time (s)"], r"\scriptsize"))

    # Hyper-parameters
    hp = m["training"]["hyperparameters"]
    names = dict(zip(comp["key"], comp["model"]))
    rows = []
    for key, params in hp.items():
        txt = ", ".join(f"{esc(k)}={v:.3g}" if isinstance(v, float) else f"{esc(k)}={esc(v)}"
                        for k, v in params.items() if k not in ("solver",))
        rows.append([esc(names[key]), txt])
    lines = [r"\footnotesize", r"\begin{tabular}{p{0.24\columnwidth}p{0.68\columnwidth}}", r"\toprule",
             r"Model & Tuned hyper-parameters (best of RandomizedSearchCV) \\", r"\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("hyperparameters.tex", "\n".join(lines) + "\n")

    abl = pd.read_csv(config.ABLATION_CSV)
    rows = [[esc(r.variant), r.n_features, f"{r.cv_roc_auc_mean:.4f} $\\pm$ {r.cv_roc_auc_std:.4f}",
             f"{r.delta_vs_core:+.4f}"] for r in abl.itertuples()]
    write("ablation.tex", tabular(pd.DataFrame(rows), "lrrr", ["Feature set", "\\#", "CV ROC-AUC", "$\\Delta$"]))


def table_importance() -> None:
    fi = pd.read_csv(config.FEATURE_IMPORTANCE_CSV)
    methods = ["lr_coef", "rf_impurity", "xgb_gain", "permutation", "shap"]
    rows = [[r["consensus_rank"], esc(r["feature"]) + (r"$^\dagger$" if r["simulated"] else ""),
             *[int(r[f"{m}_rank"]) for m in methods], f"{r['mean_rank']:.1f}", f"{r['consensus_score']:.2f}"]
            for _, r in fi.iterrows()]
    write("feature_importance.tex", tabular(pd.DataFrame(rows), "rlrrrrrrr",
          ["\\#", "Feature", "LR", "RF", "XGB", "Perm", "SHAP", "Mean", "Score"]))
    ag = pd.read_csv(config.METHOD_AGREEMENT_CSV, index_col=0)
    short = {"lr_coef": "LR", "rf_impurity": "RF", "xgb_gain": "XGB", "permutation": "Perm", "shap": "SHAP"}
    rows = [[short[i], *[f"{ag.loc[i, j]:.2f}" for j in ag.columns]] for i in ag.index]
    write("method_agreement.tex", tabular(pd.DataFrame(rows), "lrrrrr", ["", *[short[c] for c in ag.columns]]))


def table_environment() -> None:
    pkgs = ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "scipy", "statsmodels",
            "matplotlib", "seaborn", "plotly", "streamlit"]
    rows = [["Python", platform.python_version()]]
    for p in pkgs:
        try:
            rows.append([p, md.version(p)])
        except md.PackageNotFoundError:
            pass
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:] + [["", ""]] * (2 * half - len(rows))
    df = pd.DataFrame([[*a, *b] for a, b in zip(left, right)])
    write("environment.tex", tabular(df, "llll", ["Library", "Version", "Library", "Version"]))


# ---------------------------------------------------------------------------
def macros(m: dict, plog: dict) -> None:
    tr = m["training"]
    best = tr["best_model"]
    bm = tr["models"][best]
    second = tr["statistical_test"]["model_b"]
    st = tr["statistical_test"]
    cl = plog["cleaning_log"]
    inj = plog["injection_log"]
    ex = m["explainability"]
    abl = {r["variant"][:3]: r for r in tr["ablation"]}
    fi = pd.read_csv(config.FEATURE_IMPORTANCE_CSV)
    corr = pd.read_csv(config.REPORTS_DIR / "correlation_pearson.csv", index_col=0)
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    vals = {
        "RawRows": f"{m['data']['raw_rows']:,}".replace(",", "{,}"),
        "AugRows": f"{m['data']['augmented_rows']:,}".replace(",", "{,}"),
        "CleanRows": f"{cl['rows_out']:,}".replace(",", "{,}"),
        "TrainRows": f"{cl['train_rows']:,}".replace(",", "{,}"),
        "TestRows": f"{cl['test_rows']:,}".replace(",", "{,}"),
        "PlacementRate": pct(m["clean_data"]["placement_rate"]),
        "NPlaced": f"{m['clean_data']['n_placed']:,}".replace(",", "{,}"),
        "NNotPlaced": f"{m['clean_data']['n_not_placed']:,}".replace(",", "{,}"),
        "LabelsInjected": inj["inconsistent_labels"],
        "DupAdded": inj["duplicate_rows_added"], "DupRemoved": cl["duplicates_removed"],
        "LabelsFixed": cl["labels_normalised"], "InvalidNaN": cl["invalid_to_nan_total"],
        "CgpaConverted": cl["cgpa_percentage_converted"], "Capped": cl["outliers_capped_total"],
        "MissingInjectedPerCol": list(inj["missing_values"].values())[0],
        "MissingTotal": cl["missing_after_cleaning_total"],
        "InvalidInjected": inj["invalid_values_total"],
        "TypeNoise": cl["type_noise_whitespace_fixed"],
        "MinorityShare": pct(tr["class_balance"]["minority_share_train"]),
        "BestModel": esc(tr["best_model_name"]),
        "SecondModel": esc({r: v["model"] for r, v in tr["models"].items()}[second]),
        "BestTestAUC": f3(bm["roc_auc"]), "BestTestAcc": f3(bm["accuracy"]), "BestTestF": f3(bm["f1"]),
        "BestTestPrec": f3(bm["precision"]), "BestTestRec": f3(bm["recall"]),
        "BestTestPRAUC": f3(bm["pr_auc"]), "BestTestMCC": f3(bm["mcc"]),
        "BestCVAUC": f3(bm["cv_roc_auc_mean"]), "BestCVAUCsd": f3(bm["cv_roc_auc_std"]),
        "WorstModel": esc(min(tr["models"].values(), key=lambda v: v["cv_roc_auc_mean"])["model"]),
        "WorstCVAUC": f3(min(v["cv_roc_auc_mean"] for v in tr["models"].values())),
        "CVAUCRange": f3(max(v["cv_roc_auc_mean"] for v in tr["models"].values())
                         - min(v["cv_roc_auc_mean"] for v in tr["models"].values())),
        "StatDiff": f"{st['mean_diff_cv_roc_auc']:.4f}", "StatP": f3(st["ttest_p"]),
        "StatPcorr": f3(st["corrected_ttest_p"]), "McNemarP": f3(st["mcnemar_p"]),
        "StatSig": "is" if st["significant_at_0.05"] else "is not",
        "FOneThr": f"{tr['threshold']['f1_optimal_cv']:.2f}",
        "FOneThrTestF": f3(tr["threshold"]["best_model_test_at_f1_threshold"]["f1"]),
        "FOneThrTestRec": f3(tr["threshold"]["best_model_test_at_f1_threshold"]["recall"]),
        "LCGap": f3(tr["learning_curve"]["final_gap"]),
        "AblCore": f"{abl['(a)']['cv_roc_auc_mean']:.4f}", "AblEng": f"{abl['(b)']['delta_vs_core']:+.4f}",
        "AblExtra": f"{abl['(c)']['delta_vs_core']:+.4f}", "AblNoSim": f"{abl['(d)']['delta_vs_core']:+.4f}",
        "EngUsed": "were" if tr["engineered_features"]["used"] else "were not",
        "TopOne": esc(ex["top3"][0]), "TopTwo": esc(ex["top3"][1]), "TopThree": esc(ex["top3"][2]),
        "LeastFeature": esc(ex["consensus_ranking"][-1]),
        "MeanRho": f"{ex['mean_pairwise_spearman']:.2f}",
        "RankProg": int(fi.set_index("feature").loc["ProgrammingSkills", "consensus_rank"]),
        "RankAtt": int(fi.set_index("feature").loc["Attendance", "consensus_rank"]),
        "MaxVIF": f"{m['vif']['max_vif']:.2f}", "MaxVIFFeature": esc(m["vif"]["max_vif_feature"]),
        "OddsTop": f"{max(ex['lr_odds_ratios'].items(), key=lambda kv: abs(__import__('math').log(kv[1])))[1]:.2f}",
        "OddsTopFeature": esc(max(ex['lr_odds_ratios'].items(), key=lambda kv: abs(__import__('math').log(kv[1])))[0]),
        "XGBName": "XGBoost" if tr["xgboost_available"] else "HistGradientBoosting",
        "NIter": config.N_ITER_SEARCH,
        "CorrInternProg": f"{corr.loc['Internships', 'ProgrammingSkills']:.2f}",
    }
    lines = ["% AUTO-GENERATED by src/make_report_tables.py - do not edit by hand"]
    for k, v in vals.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    # Generated insights as an itemize list
    ins = "\n".join(r"\item " + esc(s) for s in m["insights"])
    write("macros.tex", "\n".join(lines) + "\n")
    write("insights.tex", "\\begin{itemize}\n" + ins + "\n\\end{itemize}\n")


# ---------------------------------------------------------------------------
def architecture_figure() -> None:
    set_plot_style()
    steps = [
        ("Raw data\n(Kaggle CSV,\nHF mirror)", "#E0E7FF"),
        ("Augmentation\n(2 simulated cols,\ninjected issues)", "#E0E7FF"),
        ("Cleaning\n(labels, dupes,\nranges, IQR cap)", "#CCFBF1"),
        ("Stratified\n80/20 split", "#CCFBF1"),
        ("Feature eng. +\nselection\n(MI, ANOVA, RFE, VIF)", "#CCFBF1"),
        ("Pipeline: impute\n-> [scale] -> model\n5 models, RandomSearch\n5-fold CV", "#FEF3C7"),
        ("Evaluation\n(test set, stats test,\nablation)", "#FEF3C7"),
        ("Explainability\n(LR, RF, XGB, perm.,\nSHAP, PDP)", "#FCE7F3"),
        ("Saved pipeline\n(best_model.joblib\n+ metadata)", "#F1F5F9"),
        ("Streamlit app\n(single / batch\nprediction, what-if)", "#F1F5F9"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    ax.axis("off")
    w, h = 0.86, 0.72
    pos = [(0.08 + i * 1.0, 1.15) for i in range(5)] + [(0.08 + (4 - i) * 1.0, 0.12) for i in range(5)]
    for (label, color), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=color, ec="#334155", lw=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)
    for i in range(len(pos) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 == y1:
            start = (x0 + w, y0 + h / 2) if x1 > x0 else (x0, y0 + h / 2)
            end = (x1, y1 + h / 2) if x1 > x0 else (x1 + w, y1 + h / 2)
        else:
            start, end = (x0 + w / 2, y0), (x1 + w / 2, y1 + h)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#334155", lw=1.2))
    ax.set_title("System architecture and workflow", loc="left", fontsize=11)
    savefig(fig, "architecture.png")


def main() -> None:
    m = load_json(config.METRICS_JSON)
    plog = load_json(config.PREPROCESSING_LOG_JSON)
    table_attributes()
    table_preprocessing(plog)
    table_vif_fs()
    table_models(m)
    table_importance()
    table_environment()
    macros(m, plog)
    architecture_figure()
    log.info("Wrote LaTeX tables + macros to %s (python %s on %s)", T, sys.version.split()[0], platform.system())


if __name__ == "__main__":
    main()

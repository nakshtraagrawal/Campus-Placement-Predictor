"""Exploratory data analysis figures (spec §3). 300-dpi PNGs in reports/figures/.

`before_cleaning()` runs on the raw augmented file, `after_cleaning()` on the clean file.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils import CLASS_NAMES, CLASS_PALETTE, get_logger, savefig, set_plot_style

log = get_logger("eda")

LABEL_ORDER = ["Not Placed", "Placed"]
LABEL_PALETTE = {"Not Placed": config.COLOR_NOT_PLACED, "Placed": config.COLOR_PLACED}


def _raw_numeric() -> pd.DataFrame:
    """Raw augmented data with final column names and numeric coercion only (no cleaning)."""
    raw = pd.read_csv(config.AUGMENTED_CSV, dtype=str)
    raw = raw.rename(columns=config.COLUMN_RENAME)
    for c in config.CORE_FEATURES:
        raw[c] = pd.to_numeric(raw[c].str.strip(), errors="coerce")
    return raw


# ---------------------------------------------------------------------------
# Before cleaning
# ---------------------------------------------------------------------------
def missing_values_plot() -> None:
    raw = pd.read_csv(config.AUGMENTED_CSV, dtype=str).rename(columns=config.COLUMN_RENAME)
    miss = raw.drop(columns=["StudentID"]).isna().sum().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = [config.MODEL_PALETTE[0] if v else "#CBD5E1" for v in miss.values]
    ax.barh(miss.index, miss.values, color=colors)
    for i, v in enumerate(miss.values):
        ax.text(v + 3, i, f"{v} ({100 * v / len(raw):.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Missing cells")
    ax.set_title("Missing values before cleaning", loc="left")
    ax.set_xlim(0, miss.max() * 1.3 + 1)
    savefig(fig, "missing_values.png")


def before_cleaning() -> None:
    set_plot_style()
    missing_values_plot()
    log.info("Saved before-cleaning figures")


# ---------------------------------------------------------------------------
# After cleaning
# ---------------------------------------------------------------------------
def class_balance_plot(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].map(CLASS_NAMES).value_counts().reindex(LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(counts.index, counts.values, color=[LABEL_PALETTE[k] for k in counts.index], width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100 * v / counts.sum():.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Students")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_title("Class balance (cleaned data)", loc="left")
    savefig(fig, "class_balance.png")


def feature_distributions_plot(df: pd.DataFrame) -> None:
    d = df.assign(Status=df[config.TARGET].map(CLASS_NAMES))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, f in zip(axes.ravel(), config.CORE_FEATURES):
        discrete = d[f].nunique() <= 12
        sns.histplot(data=d, x=f, hue="Status", hue_order=LABEL_ORDER, palette=LABEL_PALETTE,
                     stat="density", common_norm=False, element="step", alpha=0.35,
                     discrete=discrete, kde=not discrete, ax=ax, legend=(f == config.CORE_FEATURES[0]))
        ax.set_title(f + (" (simulated)" if f in config.SIMULATED_FEATURES else ""), loc="left")
        ax.set_ylabel("")
    fig.suptitle("Feature distributions by placement status", x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "feature_distributions.png")


def boxplots_plot(df: pd.DataFrame) -> None:
    d = df.assign(Status=df[config.TARGET].map(CLASS_NAMES))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, f in zip(axes.ravel(), config.CORE_FEATURES):
        sns.boxplot(data=d, x="Status", y=f, hue="Status", order=LABEL_ORDER, palette=LABEL_PALETTE,
                    ax=ax, width=0.5, fliersize=1.5, legend=False)
        ax.set_xlabel("")
        ax.set_title(f, loc="left")
    fig.suptitle("Feature boxplots by placement status", x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "boxplots_by_class.png")


def correlation_plot(df: pd.DataFrame) -> pd.DataFrame:
    cols = config.CORE_FEATURES + [config.TARGET]
    pear = df[cols].corr(method="pearson")
    spear = df[cols].corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, m, title in [(axes[0], pear, "Pearson"), (axes[1], spear, "Spearman")]:
        mask = np.triu(np.ones_like(m, dtype=bool), k=1)
        sns.heatmap(m, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    square=True, cbar=ax is axes[1], ax=ax, annot_kws={"size": 8})
        ax.set_title(f"{title} correlation (incl. target)", loc="left")
    fig.tight_layout()
    savefig(fig, "correlation_heatmap.png")
    pear.to_csv(config.REPORTS_DIR / "correlation_pearson.csv")
    return pear


def placement_rate_by_bins_plot(df: pd.DataFrame) -> None:
    specs = [
        ("CGPA", pd.cut(df["CGPA"], bins=[0, 6.5, 7.0, 7.5, 8.0, 8.5, 10], include_lowest=True)),
        ("Internships", df["Internships"]),
        ("Projects", df["Projects"]),
        ("AptitudeScore", pd.cut(df["AptitudeScore"], bins=[0, 60, 70, 80, 90, 100], include_lowest=True)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (name, bins) in zip(axes, specs):
        g = df.groupby(bins, observed=True)[config.TARGET].agg(["mean", "size"])
        labels = [str(i) if not isinstance(i, float) else f"{i:g}" for i in g.index]
        bars = ax.bar(labels, 100 * g["mean"], color=config.MODEL_PALETTE[0], alpha=0.85)
        for b, (rate, n) in zip(bars, g.values):
            ax.text(b.get_x() + b.get_width() / 2, 100 * rate + 1.5, f"{100 * rate:.0f}%",
                    ha="center", fontsize=8)
        ax.set_title(name, loc="left")
        ax.set_xlabel("bin")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Placement rate (%)")
    axes[0].set_ylim(0, 105)
    fig.suptitle("Placement rate by binned feature value", x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "placement_rate_by_bins.png")


def outliers_before_after_plot(df: pd.DataFrame) -> None:
    raw = _raw_numeric()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, f in zip(axes, ["CGPA", "Attendance", "AptitudeScore"]):
        data = pd.DataFrame({
            "value": pd.concat([raw[f], df[f]], ignore_index=True),
            "stage": ["Before"] * len(raw) + ["After"] * len(df),
        }).dropna()
        sns.boxplot(data=data, x="stage", y="value", hue="stage", ax=ax, width=0.5, fliersize=2,
                    palette={"Before": "#94A3B8", "After": config.MODEL_PALETTE[0]}, legend=False)
        ax.set_yscale("symlog") if f != "AptitudeScore" else None
        ax.set_title(f, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Outliers and invalid entries before vs after cleaning (symlog scale for CGPA/Attendance)",
                 x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "outliers_before_after.png")


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = config.CORE_FEATURES + config.EXTRA_FEATURES
    desc = df[cols].describe().T
    desc["missing_pct"] = 100 * df[cols].isna().mean()
    desc.index.name = "feature"
    desc.to_csv(config.DATA_SUMMARY_CSV)
    return desc


def after_cleaning() -> None:
    set_plot_style()
    df = pd.read_csv(config.CLEAN_CSV)
    class_balance_plot(df)
    feature_distributions_plot(df)
    boxplots_plot(df)
    correlation_plot(df)
    placement_rate_by_bins_plot(df)
    outliers_before_after_plot(df)
    data_summary(df)
    log.info("Saved after-cleaning EDA figures + data_summary.csv")


if __name__ == "__main__":
    before_cleaning()
    after_cleaning()

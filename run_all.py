"""Rebuild everything with one command:  python run_all.py

data -> augment/inject -> EDA (before) -> preprocess -> EDA (after) -> feature analysis
-> train -> explain -> report tables/figures -> summary table
"""
from __future__ import annotations

import os
import shutil
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd

from src import config
from src.utils import get_logger, load_json

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

log = get_logger("run_all")


def step(title, fn):
    log.info("=" * 70)
    log.info("STEP: %s", title)
    t0 = time.perf_counter()
    out = fn()
    log.info("done in %.1fs", time.perf_counter() - t0)
    return out


def copy_figures_to_report() -> None:
    config.REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_dir in [config.FIG_DIR, config.SCREENSHOT_DIR]:
        for p in src_dir.glob("*.png"):
            shutil.copy(p, config.REPORT_FIG_DIR / p.name)
            n += 1
    log.info("Copied %d figures to %s", n, config.REPORT_FIG_DIR)


def make_overleaf_zip() -> None:
    """Zip the LaTeX sources (no build files) so the report can be uploaded to Overleaf."""
    import zipfile
    keep = {".tex", ".bib", ".cls", ".bst", ".png"}
    out = config.ROOT / "report_overleaf.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(config.REPORT_DIR.rglob("*")):
            if p.is_file() and p.suffix in keep:
                z.write(p, p.relative_to(config.REPORT_DIR))
    log.info("Wrote %s", out.name)


def print_summary() -> None:
    comp = pd.read_csv(config.MODEL_COMPARISON_CSV)
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc",
            "cv_roc_auc_mean", "cv_roc_auc_std", "train_time_s", "selected"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print("\n" + "=" * 100)
        print("MODEL COMPARISON (test set; CV = 5-fold on training set)")
        print(comp[cols].to_string(index=False))
        fi = pd.read_csv(config.FEATURE_IMPORTANCE_CSV)
        print("\nCONSENSUS FEATURE RANKING")
        print(fi[["consensus_rank", "feature", "mean_rank", "consensus_score"]].to_string(index=False))
        print("\nINSIGHTS")
        for s in load_json(config.METRICS_JSON).get("insights", []):
            print(" -", s)
        print("=" * 100)


def main() -> None:
    t0 = time.perf_counter()
    config.ensure_dirs()
    # Fresh metrics file each run so no stale numbers survive
    for p in [config.METRICS_JSON, config.PREPROCESSING_LOG_JSON]:
        p.unlink(missing_ok=True)

    from src import eda, explain, features, get_data, make_report_tables, preprocess, train

    step("1/8 Download + augment + inject data issues", get_data.main)
    step("2/8 EDA before cleaning", eda.before_cleaning)
    step("3/8 Preprocessing / cleaning", preprocess.main)
    step("4/8 EDA after cleaning", eda.after_cleaning)
    step("5/8 Feature selection + VIF", features.main)
    step("6/8 Train, tune and evaluate models", train.main)
    step("7/8 Feature influence (explainability)", explain.main)
    step("8/8 Report tables, architecture diagram, copy figures",
         lambda: (make_report_tables.main(), copy_figures_to_report(), make_overleaf_zip()))
    print_summary()
    log.info("ALL DONE in %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()

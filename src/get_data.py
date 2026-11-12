"""Step 1: obtain the raw dataset, add the two simulated columns and inject data-quality issues.

Outputs
-------
data/raw/placementdata.csv                    untouched download
data/interim/placement_raw_augmented.csv      + ProgrammingSkills, Attendance + injected issues
reports/preprocessing_log.json -> "injection_log"
reports/metrics.json           -> "data"
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, load_json, save_json, update_metrics

log = get_logger("get_data")


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def _try_huggingface() -> bool:
    """Download the CSV from the Hugging Face mirror (no login required)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        log.warning("huggingface_hub not installed - skipping HF mirror")
        return False
    try:
        files = list_repo_files(config.HF_REPO_ID, repo_type="dataset")
        csvs = sorted(f for f in files if f.lower().endswith(".csv"))
        if not csvs:
            log.warning("No CSV found in HF repo %s", config.HF_REPO_ID)
            return False
        single = [f for f in csvs if Path(f).name.lower() == "placementdata.csv"]
        if single:
            shutil.copy(hf_hub_download(config.HF_REPO_ID, single[0], repo_type="dataset"), config.RAW_CSV)
        else:
            # The mirror stores the 10,000 Kaggle rows as train/validation/test files without the
            # StudentID column -> concatenate them and add a sequential StudentID (1..N).
            parts = [pd.read_csv(hf_hub_download(config.HF_REPO_ID, f, repo_type="dataset")) for f in csvs]
            df = pd.concat(parts, ignore_index=True)
            if "StudentID" not in df.columns:
                df.insert(0, "StudentID", range(1, len(df) + 1))
            df.to_csv(config.RAW_CSV, index=False)
        log.info("Downloaded %s from Hugging Face (%s)", ", ".join(single or csvs), config.HF_URL)
        return True
    except Exception as exc:  # network errors, repo moved, ...
        log.warning("Hugging Face download failed: %s", exc)
        return False


def _try_kaggle() -> bool:
    """Fallback: Kaggle API, only if the user has credentials configured."""
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        return False
    slug = config.KAGGLE_URL.split("/datasets/")[-1]
    try:
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug,
                        "-p", str(config.RAW_DIR), "--unzip"], check=True)
        found = list(config.RAW_DIR.glob("*.csv"))
        if found and not config.RAW_CSV.exists():
            found[0].rename(config.RAW_CSV)
        return config.RAW_CSV.exists()
    except Exception as exc:
        log.warning("Kaggle download failed: %s", exc)
        return False


def download_raw() -> str:
    """Return the source used. Exits with code 1 if no source works (spec §2.1)."""
    config.ensure_dirs()
    if config.RAW_CSV.exists():
        log.info("Using existing %s", config.RAW_CSV)
        return "local file (originally from " + config.HF_URL + ")"
    if _try_huggingface():
        return config.HF_URL
    if _try_kaggle():
        return config.KAGGLE_URL
    log.error("Could not download the dataset automatically.\n"
              "Please download 'placementdata.csv' from Kaggle (%s)\n"
              "and save it as %s, then re-run `python run_all.py`.",
              config.KAGGLE_URL, config.RAW_CSV)
    sys.exit(1)


def load_and_validate_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in config.EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing expected columns: {missing}")
    if len(df) < config.MIN_ROWS:
        raise ValueError(f"Raw CSV has only {len(df)} rows (expected >= {config.MIN_ROWS})")
    return df[config.EXPECTED_RAW_COLUMNS]


# ---------------------------------------------------------------------------
# 2. Simulated columns (spec §2.2) - formulas documented in src/config.py
# ---------------------------------------------------------------------------
def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def add_simulated_columns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    placed = (df["PlacementStatus"].astype(str).str.strip().str.lower() == "placed").astype(int)
    n = len(df)
    prog = (5.5 + 1.1 * _z(df["Projects"]) + 0.7 * _z(df["Internships"]) + 0.4 * _z(df["CGPA"])
            + 0.6 * placed + rng.normal(0, 1.6, n))
    df["ProgrammingSkills"] = np.clip(np.round(prog), 1, 10).astype(int)
    att = 80 + 3.0 * _z(df["CGPA"]) + 2.0 * placed + rng.normal(0, 7.5, n)
    df["Attendance"] = np.clip(np.round(att, 1), 45, 100)
    return df


# ---------------------------------------------------------------------------
# 3. Inject realistic data-quality issues (spec §2.3) - seeded and logged
# ---------------------------------------------------------------------------
def inject_issues(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    df = df.copy().reset_index(drop=True)
    n = len(df)
    inj: dict = {"rows_before_injection": n}

    # Object dtype so that we can store strings such as "8.2 " in numeric columns
    for c in ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "Attendance", "ProgrammingSkills"]:
        df[c] = df[c].astype(object)

    # (a) Outliers / invalid entries in ~0.5% of rows
    n_out = int(round(0.005 * n))
    out_rows = rng.choice(n, n_out, replace=False)
    kinds = rng.choice(["cgpa_percentage", "cgpa_impossible", "attendance_over_100",
                        "attendance_negative", "aptitude_over_100"], n_out)
    outlier_counts: dict[str, int] = {}
    for r, k in zip(out_rows, kinds):
        if k == "cgpa_percentage":       # percentage typed instead of CGPA, e.g. 78
            df.at[r, "CGPA"] = round(float(df.at[r, "CGPA"]) * 9.5, 1)
        elif k == "cgpa_impossible":     # impossible value such as 12.5
            df.at[r, "CGPA"] = float(rng.choice([12.5, 15.0, 11.2, -1.0]))
        elif k == "attendance_over_100":
            df.at[r, "Attendance"] = float(rng.choice([105.0, 120.0, 150.0, 999.0]))
        elif k == "attendance_negative":
            df.at[r, "Attendance"] = float(rng.choice([-5.0, -10.0, -1.0]))
        else:
            df.at[r, "AptitudeTestScore"] = int(rng.choice([110, 125, 150, 200]))
        outlier_counts[k] = outlier_counts.get(k, 0) + 1
    inj["invalid_values"] = dict(sorted(outlier_counts.items()))
    inj["invalid_values_total"] = int(n_out)

    # (b) Missing values (MCAR) in ~3% of five columns; avoid overwriting injected outliers
    missing_counts = {}
    candidates = np.setdiff1d(np.arange(n), out_rows)
    for c in ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "Attendance", "ProgrammingSkills"]:
        idx = rng.choice(candidates, int(round(0.03 * n)), replace=False)
        df.loc[idx, c] = np.nan
        missing_counts[c] = int(len(idx))
    inj["missing_values"] = missing_counts

    # (c) Type noise: numeric cells stored as strings with stray whitespace, e.g. "8.2 "
    type_noise = {}
    for c in ["CGPA", "AptitudeTestScore"]:
        ok = df.index[df[c].notna()].to_numpy()
        idx = rng.choice(ok, 25, replace=False)
        for r in idx:
            df.at[r, c] = rng.choice([f"{df.at[r, c]} ", f" {df.at[r, c]}", f" {df.at[r, c]} "])
        type_noise[c] = 25
    inj["type_noise_strings"] = type_noise

    # (d) Inconsistent target labels (~15% of rows get a non-canonical spelling)
    placed_variants = ["placed", " Placed ", "Placed "]
    not_variants = ["Not Placed", "not placed", "NP", "Notplaced"]
    lab_idx = rng.choice(n, int(round(0.15 * n)), replace=False)
    for r in lab_idx:
        is_placed = str(df.at[r, "PlacementStatus"]).strip().lower() == "placed"
        df.at[r, "PlacementStatus"] = rng.choice(placed_variants if is_placed else not_variants)
    inj["inconsistent_labels"] = int(len(lab_idx))

    # (e) Inconsistent Yes/No categoricals (~10% of rows per column)
    cat_counts = {}
    for c in ["ExtracurricularActivities", "PlacementTraining"]:
        idx = rng.choice(n, int(round(0.10 * n)), replace=False)
        for r in idx:
            yes = str(df.at[r, c]).strip().lower() in {"yes", "y"}
            df.at[r, c] = rng.choice(["yes", "Y", "YES "] if yes else ["no", "N", " No"])
        cat_counts[c] = int(len(idx))
    inj["inconsistent_categoricals"] = cat_counts

    # (f) Exact duplicate rows (~1%), appended then shuffled
    dup_idx = rng.choice(n, int(round(0.01 * n)), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df = df.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
    inj["duplicate_rows_added"] = int(len(dup_idx))
    inj["rows_after_injection"] = int(len(df))
    return df, inj


def main() -> pd.DataFrame:
    source = download_raw()
    raw = load_and_validate_raw()
    log.info("Raw data: %d rows x %d columns", *raw.shape)

    rng = np.random.default_rng(config.RANDOM_STATE)
    aug = add_simulated_columns(raw, rng)
    dirty, injection_log = inject_issues(aug, rng)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    dirty.to_csv(config.AUGMENTED_CSV, index=False)
    log.info("Wrote %s (%d rows)", config.AUGMENTED_CSV.name, len(dirty))

    # Start a fresh preprocessing log with the injection details
    plog = load_json(config.PREPROCESSING_LOG_JSON)
    plog["injection_log"] = injection_log
    save_json(plog, config.PREPROCESSING_LOG_JSON)

    status = raw["PlacementStatus"].astype(str).str.strip().str.lower()
    update_metrics("data", {
        "source": source,
        "huggingface_mirror": config.HF_URL,
        "kaggle_original": config.KAGGLE_URL,
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "raw_placement_rate": float((status == "placed").mean()),
        "augmented_rows": int(len(dirty)),
        "simulated_columns": config.SIMULATED_FEATURES,
    })
    return dirty


if __name__ == "__main__":
    main()

"""Step 2: clean the augmented raw data (spec §4). Every step logs what it changed.

Order of operations
-------------------
1. Standardise column names               5. Invalid values -> NaN (CGPA % -> CGPA conversion)
2. Strip whitespace, coerce numerics      6. Stratified 80/20 train/test split
3. Normalise target + Yes/No to 1/0       7. IQR capping (bounds learned on TRAIN rows only)
4. Drop exact duplicates                  8. Save (missing values are NOT imputed here)

Imputation happens later *inside* the sklearn Pipeline, so the median is learned from the
training folds only. This prevents information from the test set leaking into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import get_logger, load_json, save_json, update_metrics

log = get_logger("preprocess")

NUMERIC_COLS = config.CORE_FEATURES + ["SSC_Marks", "HSC_Marks"]
# Columns that are counts/ratings with a small fixed scale: range checks are enough, IQR
# capping would wrongly clip legitimate values such as 3 internships.
IQR_COLS = ["CGPA", "AptitudeScore", "CommunicationSkills", "Attendance", "SSC_Marks", "HSC_Marks"]

TARGET_MAP = {"placed": 1, "notplaced": 0, "np": 0}
YES_NO_MAP = {"yes": 1, "y": 1, "no": 0, "n": 0}
CGPA_PERCENT_MIN = 35.0  # values in [35, 100] are read as a percentage typed into the CGPA field


def load_raw_augmented() -> pd.DataFrame:
    # dtype=str: read everything as text so we can see (and log) the type noise
    return pd.read_csv(config.AUGMENTED_CSV, dtype=str, keep_default_na=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    clog: dict = {"rows_in": int(len(df))}

    # 1. Standardise column names
    df = df.rename(columns={c: config.COLUMN_RENAME.get(c.strip(), c.strip()) for c in df.columns})

    # Remember the raw spellings so we can count every non-canonical label/category
    raw_target = df[config.TARGET].copy()
    raw_cats = {c: df[c].copy() for c in config.BINARY_CATEGORICALS}

    # 2. Strip whitespace and coerce numeric columns
    type_noise = 0
    for c in df.columns:
        s = df[c]
        stripped = s.str.strip()
        if c in NUMERIC_COLS:
            type_noise += int(((s != stripped) & s.notna()).sum())
        df[c] = stripped
    clog["type_noise_whitespace_fixed"] = type_noise
    unparsable = {}
    for c in NUMERIC_COLS + ["StudentID"]:
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        unparsable[c] = int(df[c].isna().sum() - before_na)
    clog["unparsable_numeric_to_nan"] = {k: v for k, v in unparsable.items() if v}

    # 3. Normalise the target and the Yes/No categoricals
    target_key = df[config.TARGET].str.lower().str.replace(r"[\s_-]", "", regex=True)
    canonical = raw_target.isin(["Placed", "NotPlaced"])
    clog["labels_normalised"] = int((~canonical & raw_target.notna()).sum())
    df[config.TARGET] = target_key.map(TARGET_MAP)
    n_bad_target = int(df[config.TARGET].isna().sum())
    df = df[df[config.TARGET].notna()].copy()
    df[config.TARGET] = df[config.TARGET].astype(int)
    clog["rows_dropped_unknown_target"] = n_bad_target

    cat_fixed = {}
    for c in config.BINARY_CATEGORICALS:
        cat_fixed[c] = int((~raw_cats[c].isin(["Yes", "No"]) & raw_cats[c].notna()).sum())
        df[c] = df[c].str.lower().map(YES_NO_MAP)
    clog["categoricals_normalised"] = cat_fixed

    # 4. Exact duplicates (StudentID included, so genuinely different students are never merged)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    clog["duplicates_removed"] = int(n0 - len(df))
    df = df.drop(columns=["StudentID"])

    # 5. Invalid values -> NaN using domain ranges
    cgpa = df["CGPA"]
    pct_mask = (cgpa > 10) & (cgpa >= CGPA_PERCENT_MIN) & (cgpa <= 100)
    df.loc[pct_mask, "CGPA"] = np.minimum(cgpa[pct_mask] / 9.5, 10.0).round(2)
    clog["cgpa_percentage_converted"] = int(pct_mask.sum())
    invalid = {}
    for c, (lo, hi) in config.FEATURE_RANGES.items():
        bad = (df[c] < lo) | (df[c] > hi)
        invalid[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    clog["invalid_to_nan"] = {k: v for k, v in invalid.items() if v}
    clog["invalid_to_nan_total"] = int(sum(invalid.values()))

    # 6. Stratified train/test split BEFORE any statistic is learned from the data
    idx_train, idx_test = train_test_split(df.index, test_size=config.TEST_SIZE,
                                           stratify=df[config.TARGET],
                                           random_state=config.RANDOM_STATE)
    df["Split"] = "train"
    df.loc[idx_test, "Split"] = "test"

    # 7. IQR (1.5x) capping = winsorising; bounds computed from training rows only
    capped, bounds = {}, {}
    train = df[df["Split"] == "train"]
    for c in IQR_COLS:
        q1, q3 = train[c].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_cap = int(((df[c] < lo) | (df[c] > hi)).sum())
        df[c] = df[c].clip(lower=lo, upper=hi)
        capped[c], bounds[c] = n_cap, [round(lo, 3), round(hi, 3)]
    clog["outliers_capped"] = capped
    clog["outliers_capped_total"] = int(sum(capped.values()))
    clog["iqr_bounds_train"] = bounds

    # Missing values that the pipeline's median imputer will fill
    feature_cols = config.CORE_FEATURES + config.EXTRA_FEATURES
    clog["missing_after_cleaning"] = {c: int(v) for c, v in df[feature_cols].isna().sum().items() if v}
    clog["missing_after_cleaning_total"] = int(df[feature_cols].isna().sum().sum())
    clog["rows_out"] = int(len(df))
    clog["train_rows"] = int((df["Split"] == "train").sum())
    clog["test_rows"] = int((df["Split"] == "test").sum())

    ordered = config.CORE_FEATURES + config.EXTRA_FEATURES + [config.TARGET, "Split"]
    return df[ordered], clog


def main() -> pd.DataFrame:
    raw = load_raw_augmented()
    clean_df, clog = clean(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.CLEAN_CSV, index=False)

    plog = load_json(config.PREPROCESSING_LOG_JSON)
    plog["cleaning_log"] = clog
    save_json(plog, config.PREPROCESSING_LOG_JSON)

    rate = float(clean_df[config.TARGET].mean())
    update_metrics("clean_data", {
        "rows": clog["rows_out"], "train_rows": clog["train_rows"], "test_rows": clog["test_rows"],
        "placement_rate": rate, "n_placed": int(clean_df[config.TARGET].sum()),
        "n_not_placed": int((1 - clean_df[config.TARGET]).sum()),
    })
    log.info("Clean data: %d rows (train %d / test %d), placement rate %.1f%%",
             clog["rows_out"], clog["train_rows"], clog["test_rows"], 100 * rate)
    log.info("Fixed: %d duplicates, %d labels, %d invalid->NaN, %d CGPA%% converted, %d capped",
             clog["duplicates_removed"], clog["labels_normalised"], clog["invalid_to_nan_total"],
             clog["cgpa_percentage_converted"], clog["outliers_capped_total"])
    return clean_df


def load_clean() -> pd.DataFrame:
    """Used by every later step."""
    return pd.read_csv(config.CLEAN_CSV)


def train_test_frames(df: pd.DataFrame | None = None):
    """Return X_train, X_test, y_train, y_test (all feature columns; callers pick a subset)."""
    df = load_clean() if df is None else df
    feats = config.CORE_FEATURES + config.EXTRA_FEATURES
    tr, te = df[df["Split"] == "train"], df[df["Split"] == "test"]
    return (tr[feats].reset_index(drop=True), te[feats].reset_index(drop=True),
            tr[config.TARGET].reset_index(drop=True), te[config.TARGET].reset_index(drop=True))


if __name__ == "__main__":
    main()

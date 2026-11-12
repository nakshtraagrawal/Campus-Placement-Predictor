"""Load the saved model and make predictions / explanations / what-if suggestions (spec §9).

Used by the Streamlit app and by explain.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src import config
from src.utils import load_json

# Risk bands for the batch page (probability of being placed)
RISK_BANDS = [(0.35, "High"), (0.65, "Medium"), (1.01, "Low")]


def load_model():
    """Return (pipeline, meta). Raises FileNotFoundError if the pipeline has not been run."""
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError("models/best_model.joblib not found - run `python run_all.py` first.")
    return joblib.load(config.BEST_MODEL_PATH), load_json(config.MODEL_META_PATH)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_frame(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, list[str]]:
    """Check columns and ranges. Returns (rows that passed, list of human-readable errors).

    Rows with a problem are excluded and reported as 'Row 5: CGPA = 12 is outside 0-10'.
    Row numbers are 1-based like in a spreadsheet (header = row 0).
    """
    feats = meta["features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        return df.iloc[0:0], [f"Missing required column(s): {', '.join(missing)}. "
                              f"Expected columns: {', '.join(feats)}."]
    errors, ok = [], np.ones(len(df), dtype=bool)
    num = pd.DataFrame(index=df.index)
    for f in feats:
        num[f] = pd.to_numeric(df[f].astype(str).str.strip(), errors="coerce")
        lo, hi = meta["feature_ranges"][f]["min"], meta["feature_ranges"][f]["max"]
        for pos, (raw, val) in enumerate(zip(df[f], num[f])):
            if pd.isna(val):
                errors.append(f"Row {pos + 1}: {f} = '{raw}' is not a number")
                ok[pos] = False
            elif not lo <= val <= hi:
                errors.append(f"Row {pos + 1}: {f} = {val:g} is outside the valid range {lo:g}-{hi:g}")
                ok[pos] = False
    return num[ok], errors


def validate_one(student: dict, meta: dict) -> dict:
    df, errors = validate_frame(pd.DataFrame([student]), meta)
    if errors:
        raise ValueError("; ".join(e.replace("Row 1: ", "") for e in errors))
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# SHAP explainer (probability units, so +0.12 means +12 percentage points)
# ---------------------------------------------------------------------------
@dataclass
class ProbaExplainer:
    pipe: object
    explainer: object
    kind: str            # "tree" or "kernel"
    feature_names: list

    def explain(self, X_raw: pd.DataFrame, nsamples: int | str = "auto"):
        """Return (shap_values [n x p], base_value) in probability units."""
        Z = self.pipe[:-1].transform(X_raw)
        if self.kind == "tree":
            sv = self.explainer.shap_values(Z)
        else:
            sv = self.explainer.shap_values(Z.to_numpy(), nsamples=nsamples, silent=True)
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, p, classes) -> class "Placed"
            sv = sv[..., 1]
        base = np.asarray(self.explainer.expected_value).ravel()
        base = float(base[-1])
        return sv, base

    def transformed(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        return self.pipe[:-1].transform(X_raw)


def make_explainer(pipe, background: pd.DataFrame, is_tree: bool) -> ProbaExplainer:
    import shap
    pre, model = pipe[:-1], pipe[-1]
    bg = pre.transform(background)
    names = list(bg.columns)
    if is_tree:
        bg_small = bg.sample(min(100, len(bg)), random_state=config.RANDOM_STATE)
        try:  # interventional TreeSHAP with probability output (XGBoost)
            expl = shap.TreeExplainer(model, data=bg_small, model_output="probability",
                                      feature_perturbation="interventional")
        except Exception:  # Random Forest: raw output is already a probability
            expl = shap.TreeExplainer(model, data=bg_small, feature_perturbation="interventional")
        return ProbaExplainer(pipe, expl, "tree", names)

    def f(Z):
        return model.predict_proba(pd.DataFrame(Z, columns=names))[:, 1]
    expl = shap.KernelExplainer(f, shap.kmeans(bg.to_numpy(), 20))
    return ProbaExplainer(pipe, expl, "kernel", names)


def load_background() -> pd.DataFrame:
    return pd.read_csv(config.MODELS_DIR / "shap_background.csv")


# ---------------------------------------------------------------------------
# Prediction API
# ---------------------------------------------------------------------------
def risk_band(p: float) -> str:
    for upper, band in RISK_BANDS:
        if p < upper:
            return band
    return "Low"


def predict_one(student: dict, pipe=None, meta=None, explainer: ProbaExplainer | None = None) -> dict:
    """Return {'proba', 'label', 'shap_values', 'base_value'} for one student dict."""
    if pipe is None:
        pipe, meta = load_model()
    clean = validate_one(student, meta)
    X = pd.DataFrame([clean])[meta["features"]]
    proba = float(pipe.predict_proba(X)[0, 1])
    out = {"proba": proba, "label": int(proba >= meta["threshold"]),
           "shap_values": None, "base_value": None}
    if explainer is None:
        explainer = make_explainer(pipe, load_background(), meta.get("is_tree_model", False))
    sv, base = explainer.explain(X, nsamples=300)
    out["shap_values"] = dict(zip(explainer.feature_names, map(float, sv[0])))
    out["base_value"] = base
    return out


def predict_batch(df: pd.DataFrame, pipe=None, meta=None) -> tuple[pd.DataFrame, list[str]]:
    """Validate + score many students. Returns (results for valid rows, list of errors)."""
    if pipe is None:
        pipe, meta = load_model()
    df = df.reset_index(drop=True)
    valid, errors = validate_frame(df, meta)
    if valid.empty:
        return valid, errors
    proba = pipe.predict_proba(valid[meta["features"]])[:, 1]
    res = df.loc[valid.index].copy()
    res.insert(0, "Row", valid.index + 1)
    res["PlacementProbability"] = np.round(proba, 4)
    res["Prediction"] = np.where(proba >= meta["threshold"], "Placed", "Not Placed")
    res["RiskBand"] = [risk_band(p) for p in proba]
    return res.reset_index(drop=True), errors


def what_if(student: dict, pipe=None, meta=None) -> list[dict]:
    """Improve one actionable feature at a time; return the effect, best first (spec §9)."""
    if pipe is None:
        pipe, meta = load_model()
    base = validate_one(student, meta)
    feats = meta["features"]
    p0 = float(pipe.predict_proba(pd.DataFrame([base])[feats])[0, 1])
    rows = []
    for f, step in config.WHAT_IF_STEPS.items():
        if f not in feats:
            continue
        new = min(base[f] + step, meta["feature_ranges"][f]["max"])
        if new == base[f]:
            continue  # already at the maximum
        changed = {**base, f: new}
        p1 = float(pipe.predict_proba(pd.DataFrame([changed])[feats])[0, 1])
        rows.append({"feature": f, "from": base[f], "to": new, "step": new - base[f],
                     "proba_before": p0, "proba_after": p1, "gain": p1 - p0})
    return sorted(rows, key=lambda r: r["gain"], reverse=True)


def template_frame(meta: dict | None = None) -> pd.DataFrame:
    """Example rows for the batch-upload CSV template."""
    return pd.DataFrame([
        {"CGPA": 8.6, "ProgrammingSkills": 8, "Internships": 2, "Certifications": 2,
         "CommunicationSkills": 4.6, "AptitudeScore": 88, "Projects": 3, "Attendance": 92},
        {"CGPA": 7.5, "ProgrammingSkills": 6, "Internships": 1, "Certifications": 1,
         "CommunicationSkills": 4.2, "AptitudeScore": 78, "Projects": 2, "Attendance": 82},
        {"CGPA": 6.6, "ProgrammingSkills": 4, "Internships": 0, "Certifications": 0,
         "CommunicationSkills": 3.6, "AptitudeScore": 62, "Projects": 1, "Attendance": 70},
    ])[config.CORE_FEATURES]

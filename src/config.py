"""Central configuration: paths, constants, feature lists and valid ranges.

Every other module imports from here so that there is exactly ONE place to
change a path, the random seed or the list of features.

Simulated columns (the public dataset does not contain them; see spec §2.2):

    rng = np.random.default_rng(42); z(x) = (x - mean) / std; placed = 1/0

    ProgrammingSkills = clip(round(5.5 + 1.1*z(Projects) + 0.7*z(Internships)
                                   + 0.4*z(CGPA) + 0.6*placed + N(0, 1.6)), 1, 10)
    Attendance        = clip(round(80 + 3.0*z(CGPA) + 2.0*placed + N(0, 7.5), 1), 45, 100)

The effects are deliberately modest so the simulated columns do not dominate
the real ones. They are NOT tuned to produce any particular result.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 20  # RandomizedSearchCV iterations per model

# --------------------------------------------------------------------------
# Paths (pathlib => works on Windows and Linux)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_CSV = RAW_DIR / "placementdata.csv"
AUGMENTED_CSV = INTERIM_DIR / "placement_raw_augmented.csv"
CLEAN_CSV = PROCESSED_DIR / "placement_clean.csv"

MODELS_DIR = ROOT / "models"
ALL_MODELS_DIR = MODELS_DIR / "all_models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"

REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SCREENSHOT_DIR = REPORTS_DIR / "screenshots"
METRICS_JSON = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
PREPROCESSING_LOG_JSON = REPORTS_DIR / "preprocessing_log.json"
DATA_SUMMARY_CSV = REPORTS_DIR / "data_summary.csv"
VIF_CSV = REPORTS_DIR / "vif.csv"
ABLATION_CSV = REPORTS_DIR / "ablation.csv"
FEATURE_SELECTION_CSV = REPORTS_DIR / "feature_selection.csv"
METHOD_AGREEMENT_CSV = REPORTS_DIR / "method_agreement.csv"
CV_SCORES_CSV = REPORTS_DIR / "cv_fold_scores.csv"
CURVES_JSON = REPORTS_DIR / "curves.json"  # ROC/PR points for the interactive app
PDP_JSON = REPORTS_DIR / "pdp.json"        # partial dependence values for the app

REPORT_DIR = ROOT / "report"
REPORT_FIG_DIR = REPORT_DIR / "figures"
REPORT_TABLE_DIR = REPORT_DIR / "tables"

# --------------------------------------------------------------------------
# Dataset source
# --------------------------------------------------------------------------
HF_REPO_ID = "Krooz/Campus_Recruitment_CSV"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"
# Kaggle API fallback only. NOTE: this slug could not be verified online - check it before relying on it.
KAGGLE_URL = "https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset"
MIN_ROWS = 5000

# Original Kaggle column -> final column name used everywhere in the project
COLUMN_RENAME = {
    "StudentID": "StudentID",
    "CGPA": "CGPA",
    "Internships": "Internships",
    "Projects": "Projects",
    "Workshops/Certifications": "Certifications",
    "AptitudeTestScore": "AptitudeScore",
    "SoftSkillsRating": "CommunicationSkills",
    "ExtracurricularActivities": "ExtracurricularActivities",
    "PlacementTraining": "PlacementTraining",
    "SSC_Marks": "SSC_Marks",
    "HSC_Marks": "HSC_Marks",
    "PlacementStatus": "Placed",
}
EXPECTED_RAW_COLUMNS = list(COLUMN_RENAME.keys())

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
TARGET = "Placed"

# The 8 features named in the problem statement (used by the main model + app)
CORE_FEATURES = [
    "CGPA",
    "ProgrammingSkills",
    "Internships",
    "Certifications",
    "CommunicationSkills",
    "AptitudeScore",
    "Projects",
    "Attendance",
]
SIMULATED_FEATURES = ["ProgrammingSkills", "Attendance"]

# Extra Kaggle columns: kept in the clean data, only used in the ablation study
EXTRA_FEATURES = ["ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
BINARY_CATEGORICALS = ["ExtracurricularActivities", "PlacementTraining"]

ENGINEERED_FEATURES = ["ExperienceScore", "SkillIndex", "AcademicConsistency"]

# Valid domain ranges (min, max). Values outside become NaN during cleaning.
FEATURE_RANGES = {
    "CGPA": (0.0, 10.0),
    "ProgrammingSkills": (1, 10),
    "Internships": (0, 3),
    "Certifications": (0, 5),
    "CommunicationSkills": (1.0, 5.0),
    "AptitudeScore": (0, 100),
    "Projects": (0, 5),
    "Attendance": (0.0, 100.0),
    "SSC_Marks": (0, 100),
    "HSC_Marks": (0, 100),
}
INTEGER_FEATURES = ["ProgrammingSkills", "Internships", "Certifications", "AptitudeScore", "Projects"]

# Human-readable descriptions (used by the app tooltips and report table)
FEATURE_DESCRIPTIONS = {
    "CGPA": "Cumulative grade point average on a 10-point scale",
    "ProgrammingSkills": "Programming skill rating (1 = beginner, 10 = expert)",
    "Internships": "Number of internships completed",
    "Certifications": "Number of workshops / certifications completed",
    "CommunicationSkills": "Soft / communication skills rating (1-5)",
    "AptitudeScore": "Score in the aptitude test (0-100)",
    "Projects": "Number of academic / personal projects",
    "Attendance": "Average class attendance (%)",
}

# what-if step sizes for "How to improve your chances" (spec §9)
WHAT_IF_STEPS = {
    "CGPA": 0.5,
    "ProgrammingSkills": 1,
    "Internships": 1,
    "Certifications": 1,
    "CommunicationSkills": 0.5,
    "AptitudeScore": 10,
    "Projects": 1,
    "Attendance": 5,
}

# Colours shared by matplotlib figures and the Streamlit app
COLOR_PLACED = "#10B981"
COLOR_NOT_PLACED = "#EF4444"
MODEL_PALETTE = ["#4F46E5", "#14B8A6", "#F59E0B", "#64748B", "#EC4899"]
FIG_DPI = 300


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, ALL_MODELS_DIR,
              REPORTS_DIR, FIG_DIR, SCREENSHOT_DIR, REPORT_FIG_DIR, REPORT_TABLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

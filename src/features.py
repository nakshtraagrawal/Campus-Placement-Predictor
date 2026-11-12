"""Step 3: feature engineering, the preprocessing Pipeline factory and feature-selection analysis.

Pipeline built for every model (spec §5-6):

    ColumnTransformer(median imputer for numeric, most-frequent for binary categoricals)
      -> [FeatureEngineer]   (only if engineered features are switched on)
      -> [StandardScaler]    (only for LR / SVM / KNN - distance or gradient based models)
      -> model

Because imputation and scaling live inside the Pipeline, they are re-fitted on the training
folds only during cross-validation - no information leaks from validation/test rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import config
from src.utils import get_logger, savefig, set_plot_style, update_metrics

log = get_logger("features")

# Models that need standardised input
SCALED_MODELS = {"logistic_regression", "knn", "svm"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the three engineered features (spec §5).

    ExperienceScore     = Internships + Projects + Certifications
    SkillIndex          = mean(z(ProgrammingSkills), z(CommunicationSkills), z(AptitudeScore))
    AcademicConsistency = CGPA * Attendance / 100

    The means/stds for the z-scores are learned in fit(), i.e. on training data only.
    """

    SKILL_COLS = ["ProgrammingSkills", "CommunicationSkills", "AptitudeScore"]

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.means_ = X[self.SKILL_COLS].mean()
        self.stds_ = X[self.SKILL_COLS].std().replace(0, 1)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        X["ExperienceScore"] = X["Internships"] + X["Projects"] + X["Certifications"]
        X["SkillIndex"] = ((X[self.SKILL_COLS] - self.means_) / self.stds_).mean(axis=1)
        X["AcademicConsistency"] = X["CGPA"] * X["Attendance"] / 100
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.feature_names_in_) + config.ENGINEERED_FEATURES)


def make_imputer(features: list[str]) -> ColumnTransformer:
    """Median for numeric columns; most frequent value for the 0/1 categoricals."""
    num = [f for f in features if f not in config.BINARY_CATEGORICALS]
    cat = [f for f in features if f in config.BINARY_CATEGORICALS]
    transformers = [("num", SimpleImputer(strategy="median"), num)]
    if cat:
        # Binary columns are already 0/1, so no encoder is needed. A multi-class categorical
        # would get a OneHotEncoder here.
        transformers.append(("cat", SimpleImputer(strategy="most_frequent"), cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def build_pipeline(model_name: str, estimator, features: list[str],
                   engineered: bool = False) -> Pipeline:
    steps = [("impute", make_imputer(features))]
    if engineered:
        steps.append(("engineer", FeatureEngineer()))
    if model_name in SCALED_MODELS:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")  # keep column names through the steps
    return pipe


# ---------------------------------------------------------------------------
# Feature-selection analysis + multicollinearity (report only; does not change the model)
# ---------------------------------------------------------------------------
def _imputed_train(X_train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return make_imputer(features).set_output(transform="pandas").fit_transform(X_train[features])


def feature_selection_analysis(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    feats = config.CORE_FEATURES
    Xi = _imputed_train(X_train, feats)

    mi = mutual_info_classif(Xi, y_train, random_state=config.RANDOM_STATE)
    f_stat, f_p = f_classif(Xi, y_train)
    chi_stat, chi_p = chi2(MinMaxScaler().fit_transform(Xi), y_train)  # chi2 needs non-negative input
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=1)
    rfe.fit(StandardScaler().fit_transform(Xi), y_train)

    res = pd.DataFrame({
        "feature": feats,
        "mutual_info": mi,
        "anova_f": f_stat, "anova_p": f_p,
        "chi2": chi_stat, "chi2_p": chi_p,
        "rfe_rank": rfe.ranking_,  # 1 = last feature kept = most important
    })
    for col, asc in [("mutual_info", False), ("anova_f", False), ("chi2", False), ("rfe_rank", True)]:
        res[f"rank_{col}"] = res[col].rank(ascending=asc, method="min").astype(int)
    res = res.sort_values("rank_mutual_info").reset_index(drop=True)
    res.to_csv(config.FEATURE_SELECTION_CSV, index=False)

    # Figure: one panel per method
    import matplotlib.pyplot as plt
    set_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    panels = [("mutual_info", "Mutual information"), ("anova_f", "ANOVA F-statistic"),
              ("chi2", "Chi-square (min-max scaled)"), ("rfe_rank", "RFE rank (LR, 1 = best)")]
    for ax, (col, title) in zip(axes, panels):
        d = res.sort_values(col, ascending=(col != "rfe_rank"))
        vals = d[col] if col != "rfe_rank" else (len(feats) + 1 - d[col])
        ax.barh(d["feature"], vals, color=config.MODEL_PALETTE[0])
        ax.set_title(title, loc="left")
        if col == "rfe_rank":
            ax.set_xticks(range(1, len(feats) + 1))
            ax.set_xticklabels(range(len(feats), 0, -1))
    fig.suptitle("Filter and wrapper feature-selection scores (training set)", x=0.01, ha="left")
    fig.tight_layout()
    savefig(fig, "feature_selection.png")

    top = {m: res.sort_values(f"rank_{m}")["feature"].head(3).tolist()
           for m in ["mutual_info", "anova_f", "chi2", "rfe_rank"]}
    update_metrics("feature_selection", {"top3_by_method": top})
    log.info("Feature selection top-3: %s", top)
    return res


def vif_table(X_train: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor: > 5 suggests problematic multicollinearity."""
    import statsmodels.api as sm
    Xi = sm.add_constant(_imputed_train(X_train, config.CORE_FEATURES))
    vif = pd.DataFrame({
        "feature": config.CORE_FEATURES,
        "VIF": [variance_inflation_factor(Xi.values, i + 1) for i in range(len(config.CORE_FEATURES))],
    }).sort_values("VIF", ascending=False)
    vif.to_csv(config.VIF_CSV, index=False)
    update_metrics("vif", {"max_vif": float(vif["VIF"].max()),
                           "max_vif_feature": vif.iloc[0]["feature"]})
    log.info("VIF max = %.2f (%s)", vif["VIF"].max(), vif.iloc[0]["feature"])
    return vif


def main() -> None:
    from src.preprocess import train_test_frames
    X_train, _, y_train, _ = train_test_frames()
    feature_selection_analysis(X_train, y_train)
    vif_table(X_train)


if __name__ == "__main__":
    main()

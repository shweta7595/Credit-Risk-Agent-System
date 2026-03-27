"""Feature engineering for credit risk scoring.

Implements WOE/IV analysis, income ratios, credit utilisation proxies,
log transforms for skewed features, employment quantiles, risk buckets,
and one-hot encoding.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

GRADE_BUCKET_MAP = {
    "A": "prime",
    "B": "prime",
    "C": "near_prime",
    "D": "subprime",
    "E": "subprime",
    "F": "deep_subprime",
    "G": "deep_subprime",
}

CATEGORICAL_VARS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "loan_grade_bucket",
    "cb_person_default_on_file",
    "emp_length_quantile",
    "age_bucket",
    "cred_hist_bucket",
]

# Applied BEFORE log transforms
RAW_CONTINUOUS_COLS = [
    "person_age",
    "person_emp_length",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "DTI",
]

# Heavily skewed features that get log1p transformed
LOG_TRANSFORM_COLS = [
    "person_income",
    "loan_amnt",
    "monthly_payment_est",
]

# After log transform, these become the columns to scale
CONTINUOUS_COLS = RAW_CONTINUOUS_COLS + [
    "log_person_income",
    "log_loan_amnt",
    "log_monthly_payment_est",
    "PTI",
    "loan_to_income",
]


# ── WOE / IV ────────────────────────────────────────────────────────────────


def calc_woe_iv(df: pd.DataFrame, feature: str, target: str = "loan_status") -> pd.DataFrame:
    """Calculate Weight of Evidence and Information Value for a categorical feature."""
    lst = []
    for val in df[feature].unique():
        good = len(df[(df[feature] == val) & (df[target] == 0)])
        bad = len(df[(df[feature] == val) & (df[target] == 1)])
        lst.append([val, good, bad])

    woe_df = pd.DataFrame(lst, columns=["Value", "Good", "Bad"])
    woe_df["Distr_Good"] = woe_df["Good"] / woe_df["Good"].sum()
    woe_df["Distr_Bad"] = woe_df["Bad"] / woe_df["Bad"].sum()
    woe_df["WOE"] = np.log(woe_df["Distr_Good"] / woe_df["Distr_Bad"])
    woe_df["IV"] = (woe_df["Distr_Good"] - woe_df["Distr_Bad"]) * woe_df["WOE"]
    return woe_df


# ── Feature Creation ────────────────────────────────────────────────────────


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive income ratios, log transforms, risk buckets, and employment quantiles."""
    df = df.copy()

    # Income ratio features (computed BEFORE log transform on raw values)
    df["DTI"] = df["loan_percent_income"]
    df["monthly_payment_est"] = (df["loan_amnt"] * (df["loan_int_rate"] / 100)) / 12
    df["PTI"] = df["monthly_payment_est"] / (df["person_income"] / 12)
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"]

    # Log transforms for heavily right-skewed features
    df["log_person_income"] = np.log1p(df["person_income"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
    df["log_monthly_payment_est"] = np.log1p(df["monthly_payment_est"])

    for col in LOG_TRANSFORM_COLS:
        before_skew = df[col].skew()
        after_skew = np.log1p(df[col]).skew()
        logger.info("Log1p(%s): skew %.2f → %.2f", col, before_skew, after_skew)

    # Drop raw skewed columns (replaced by log versions)
    df = df.drop(columns=LOG_TRANSFORM_COLS)

    # Employment length quantiles
    df["emp_length_quantile"] = pd.qcut(
        df["person_emp_length"],
        q=4,
        labels=["short", "medium", "long", "very_long"],
        duplicates="drop",
    )

    # Risk buckets
    df["loan_grade_bucket"] = df["loan_grade"].map(GRADE_BUCKET_MAP)

    df["age_bucket"] = pd.cut(
        df["person_age"],
        bins=[0, 25, 35, 50, 65, 120],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"],
    )

    df["cred_hist_bucket"] = pd.cut(
        df["cb_person_cred_hist_length"],
        bins=[0, 2, 5, 10, 20, 50],
        labels=["0-2", "3-5", "6-10", "11-20", "20+"],
    )

    logger.info("Created engineered features – shape now %s", df.shape)
    return df


# ── Model Data Preparation ──────────────────────────────────────────────────


def prepare_model_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], StandardScaler, List[str]]:
    """One-hot encode, scale continuous features, split train/test."""
    target = "loan_status"

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_VARS, drop_first=True)

    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaled_columns = [c for c in CONTINUOUS_COLS if c in X_train.columns]

    scaler = StandardScaler()
    X_train[scaled_columns] = scaler.fit_transform(X_train[scaled_columns])
    X_test[scaled_columns] = scaler.transform(X_test[scaled_columns])

    # Log class imbalance
    class_counts = y_train.value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]
    logger.info(
        "Class imbalance – 0: %d, 1: %d (ratio %.2f:1) → handled by balanced sample_weight",
        class_counts[0], class_counts[1], imbalance_ratio,
    )

    _save_quantile_edges(df)

    # Save log transform column list for inference
    joblib.dump(LOG_TRANSFORM_COLS, MODEL_DIR / "log_columns.joblib")

    logger.info(
        "Train: %d  Test: %d  Features: %d  Scaled: %d",
        len(X_train), len(X_test), len(feature_names), len(scaled_columns),
    )
    return X_train, X_test, y_train, y_test, feature_names, scaler, scaled_columns


def _save_quantile_edges(df: pd.DataFrame) -> None:
    """Persist the emp_length_quantile bin edges so inference can reuse them."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _, bin_edges = pd.qcut(
        df["person_emp_length"], q=4, retbins=True, duplicates="drop"
    )
    joblib.dump(bin_edges.tolist(), MODEL_DIR / "emp_quantile_edges.joblib")
    logger.info("Saved emp_length quantile edges: %s", bin_edges.tolist())


def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a summary dict describing the engineered features."""
    iv_results = {}
    for cat in ["loan_grade", "person_home_ownership", "loan_intent", "cb_person_default_on_file"]:
        if cat in df.columns:
            woe_df = calc_woe_iv(df, cat)
            iv_results[cat] = round(woe_df["IV"].sum(), 4)

    return {
        "total_features": len(df.columns),
        "engineered": [
            "DTI", "PTI", "loan_to_income",
            "log_person_income", "log_loan_amnt", "log_monthly_payment_est",
            "emp_length_quantile", "loan_grade_bucket", "age_bucket", "cred_hist_bucket",
        ],
        "log_transformed": LOG_TRANSFORM_COLS,
        "information_values": iv_results,
        "shape": df.shape,
    }

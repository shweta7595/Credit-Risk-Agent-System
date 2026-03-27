"""Feature Engineering Agent – computes the full feature vector at inference.

Mirrors training: outlier capping → derived ratios → log transforms →
risk buckets → one-hot encoding.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

GRADE_BUCKET_MAP = {
    "A": "prime", "B": "prime", "C": "near_prime",
    "D": "subprime", "E": "subprime",
    "F": "deep_subprime", "G": "deep_subprime",
}

# One-hot dummies kept after drop_first=True (first alphabetically is dropped)
HOME_OWNERSHIP_DUMMIES = ["OTHER", "OWN", "RENT"]
LOAN_INTENT_DUMMIES = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
LOAN_GRADE_DUMMIES = ["B", "C", "D", "E", "F", "G"]
GRADE_BUCKET_DUMMIES = ["near_prime", "prime", "subprime"]
DEFAULT_DUMMIES = ["Y"]
EMP_QUANTILE_DUMMIES = ["medium", "long", "very_long"]
AGE_BUCKET_DUMMIES = ["26-35", "36-50", "51-65", "65+"]
CRED_HIST_DUMMIES = ["3-5", "6-10", "11-20", "20+"]

AGE_BINS = [0, 25, 35, 50, 65, 120]
AGE_LABELS = ["18-25", "26-35", "36-50", "51-65", "65+"]
CRED_HIST_BINS = [0, 2, 5, 10, 20, 50]
CRED_HIST_LABELS = ["0-2", "3-5", "6-10", "11-20", "20+"]


def _load_outlier_caps() -> Dict[str, Dict[str, float]]:
    try:
        return joblib.load(MODEL_DIR / "outlier_caps.joblib")
    except FileNotFoundError:
        return {}


def _get_emp_quantile(emp_length: float) -> str:
    try:
        edges = joblib.load(MODEL_DIR / "emp_quantile_edges.joblib")
    except FileNotFoundError:
        edges = [0, 2, 4, 7, 123]
    labels = ["short", "medium", "long", "very_long"]
    for i in range(len(edges) - 1):
        if emp_length <= edges[i + 1]:
            return labels[min(i, len(labels) - 1)]
    return labels[-1]


def _bucket(value: float, bins: list, labels: list) -> str:
    for i in range(len(bins) - 1):
        if value <= bins[i + 1]:
            return labels[i]
    return labels[-1]


def _one_hot(actual: str, categories: list, prefix: str) -> Dict[str, int]:
    return {f"{prefix}_{cat}": int(actual == cat) for cat in categories}


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw applicant data into the model feature vector."""
    raw = state["raw_data"]
    trace_entry = {
        "agent": "FeatureEngineeringAgent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    # ── Step 1: Apply outlier caps (same as training) ────────────────────
    caps = _load_outlier_caps()
    age = raw["person_age"]
    income = raw["person_income"]
    emp_len = raw["person_emp_length"]
    loan_amnt = raw["loan_amnt"]
    cred_hist = raw["cb_person_cred_hist_length"]
    int_rate = raw["loan_int_rate"]

    for col, val in [("person_age", age), ("person_income", income),
                     ("person_emp_length", emp_len), ("loan_amnt", loan_amnt),
                     ("cb_person_cred_hist_length", cred_hist)]:
        if col in caps:
            lo, hi = caps[col]["lower"], caps[col]["upper"]
            if col == "person_age": age = max(lo, min(hi, val))
            elif col == "person_income": income = max(lo, min(hi, val))
            elif col == "person_emp_length": emp_len = max(lo, min(hi, val))
            elif col == "loan_amnt": loan_amnt = max(lo, min(hi, val))
            elif col == "cb_person_cred_hist_length": cred_hist = max(lo, min(hi, val))

    # ── Step 2: Derived features (on capped values) ──────────────────────
    dti = raw["loan_percent_income"]
    monthly_payment_est = (loan_amnt * (int_rate / 100)) / 12
    pti = monthly_payment_est / (income / 12) if income > 0 else 0.0
    loan_to_income = loan_amnt / income if income > 0 else 0.0

    # ── Step 3: Log transforms (same as training) ────────────────────────
    log_income = float(np.log1p(income))
    log_loan_amnt = float(np.log1p(loan_amnt))
    log_payment = float(np.log1p(monthly_payment_est))

    # ── Step 4: Bucket assignments ───────────────────────────────────────
    emp_quantile = _get_emp_quantile(emp_len)
    grade_bucket = GRADE_BUCKET_MAP.get(raw["loan_grade"], "subprime")
    age_bkt = _bucket(age, AGE_BINS, AGE_LABELS)
    cred_bkt = _bucket(cred_hist, CRED_HIST_BINS, CRED_HIST_LABELS)
    default_flag_str = "Y" if raw["cb_person_default_on_file"] == 1 else "N"

    # ── Step 5: Build feature vector ─────────────────────────────────────
    feature_vector = {
        "person_age": age,
        "person_emp_length": emp_len,
        "loan_int_rate": int_rate,
        "loan_percent_income": dti,
        "cb_person_cred_hist_length": cred_hist,
        "emp_length_missing": raw.get("emp_length_missing", 0),
        "DTI": dti,
        "PTI": pti,
        "loan_to_income": loan_to_income,
        "log_person_income": log_income,
        "log_loan_amnt": log_loan_amnt,
        "log_monthly_payment_est": log_payment,
    }

    # ── Step 6: One-hot encoded dummies ──────────────────────────────────
    feature_vector.update(_one_hot(raw["person_home_ownership"], HOME_OWNERSHIP_DUMMIES, "person_home_ownership"))
    feature_vector.update(_one_hot(raw["loan_intent"], LOAN_INTENT_DUMMIES, "loan_intent"))
    feature_vector.update(_one_hot(raw["loan_grade"], LOAN_GRADE_DUMMIES, "loan_grade"))
    feature_vector.update(_one_hot(grade_bucket, GRADE_BUCKET_DUMMIES, "loan_grade_bucket"))
    feature_vector.update(_one_hot(default_flag_str, DEFAULT_DUMMIES, "cb_person_default_on_file"))
    feature_vector.update(_one_hot(emp_quantile, EMP_QUANTILE_DUMMIES, "emp_length_quantile"))
    feature_vector.update(_one_hot(age_bkt, AGE_BUCKET_DUMMIES, "age_bucket"))
    feature_vector.update(_one_hot(cred_bkt, CRED_HIST_DUMMIES, "cred_hist_bucket"))

    engineered = {
        "DTI": dti, "PTI": pti, "loan_to_income": loan_to_income,
        "log_person_income": log_income, "log_loan_amnt": log_loan_amnt,
        "log_monthly_payment_est": log_payment,
        "outlier_caps_applied": list(caps.keys()),
    }

    trace_entry["status"] = "completed"
    trace_entry["features_created"] = len(feature_vector)

    logger.info("FeatureEngineeringAgent: built %d features", len(feature_vector))

    return {
        "engineered_features": engineered,
        "feature_vector": feature_vector,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }

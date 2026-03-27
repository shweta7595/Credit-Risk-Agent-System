"""Data ingestion pipeline for the Credit Risk dataset.

Downloads the Kaggle dataset, validates schema, handles outliers with
percentile capping (Winsorization), and imputes missing values.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

OUTLIER_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "cb_person_cred_hist_length",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def download_dataset(force: bool = False) -> Path:
    """Download the credit risk dataset from Kaggle via kagglehub."""
    raw_path = DATA_DIR / "credit_risk_dataset.csv"
    if raw_path.exists() and not force:
        logger.info("Dataset already exists at %s", raw_path)
        return raw_path

    import kagglehub

    cache_path = kagglehub.dataset_download("laotse/credit-risk-dataset")
    src = Path(cache_path) / "credit_risk_dataset.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.read_csv(src).to_csv(raw_path, index=False)
    logger.info("Dataset downloaded to %s", raw_path)
    return raw_path


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if the dataframe doesn't match expected schema."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logger.info("Schema validation passed – %d columns, %d rows", len(df.columns), len(df))


def cap_outliers(df: pd.DataFrame, lower_pct: float = 0.01, upper_pct: float = 0.99) -> Tuple[pd.DataFrame, Dict]:
    """Winsorize outliers at given percentiles and return the cap values."""
    df = df.copy()
    caps = {}

    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        lo = float(s.quantile(lower_pct))
        hi = float(s.quantile(upper_pct))
        before_lo = int((s < lo).sum())
        before_hi = int((s > hi).sum())

        df[col] = df[col].clip(lower=lo, upper=hi)
        caps[col] = {"lower": lo, "upper": hi}
        logger.info(
            "Capped %s to [%.1f, %.1f] — clipped %d low, %d high",
            col, lo, hi, before_lo, before_hi,
        )

    # Persist caps for inference
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(caps, MODEL_DIR / "outlier_caps.joblib")
    return df, caps


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers, create missingness flags, and impute missing values."""
    df = df.copy()

    # Percentile-based outlier capping
    df, caps = cap_outliers(df)

    # Missingness flags BEFORE imputation
    df["emp_length_missing"] = df["person_emp_length"].isnull().astype(int)

    # Impute missing numerics with median
    for col in ["person_emp_length", "loan_int_rate", "person_age"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        logger.info("Imputed %s with median=%.2f", col, median_val)

    return df


def ingest(force_download: bool = False) -> Tuple[pd.DataFrame, Path]:
    """Full ingestion pipeline: download → validate → clean → save."""
    raw_path = download_dataset(force=force_download)
    df = pd.read_csv(raw_path)
    validate_schema(df)
    df = clean_data(df)

    clean_path = DATA_DIR / "credit_risk_clean.csv"
    df.to_csv(clean_path, index=False)
    logger.info("Clean dataset saved to %s (%d rows)", clean_path, len(df))
    return df, clean_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, path = ingest()
    print(f"Ingested {len(df)} rows → {path}")

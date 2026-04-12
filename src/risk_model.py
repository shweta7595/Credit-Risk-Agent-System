"""Risk scoring model: Gradient Boosting (production) + Random Forest (benchmark).

Precision-recall threshold tuning and plots use the production GB model only.
SHAP TreeExplainer is applied to the saved Gradient Boosting classifier.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 0.5

ClassifierModel = Union[GradientBoostingClassifier, RandomForestClassifier]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[GradientBoostingClassifier, "CalibratedClassifierCV", Dict[str, Any]]:
    """Train production Gradient Boosting with balanced sample weights.

    Returns both the raw model (for SHAP TreeExplainer) and a calibrated
    wrapper (for probability outputs). Isotonic calibration corrects the
    score inflation caused by compute_sample_weight('balanced').
    """
    sample_weights = compute_sample_weight("balanced", y_train)

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Split X_test into calibration set (50%) and evaluation set (50%).
    # Calibration corrects score inflation from balanced sample weights.
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X_cal, y_cal)

    metrics = evaluate_model(calibrated, X_eval, y_eval)
    logger.info(
        "GradientBoosting (calibrated) – AUC: %.4f, F1: %.4f", metrics["auc"], metrics["f1"]
    )
    return model, calibrated, metrics


def train_benchmark_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Train Random Forest for benchmark comparison only (not saved for inference)."""
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test)
    logger.info(
        "RandomForest (benchmark) – AUC: %.4f, F1: %.4f @ t=0.50",
        metrics["auc"], metrics["f1"],
    )
    return metrics


def evaluate_model(
    model: ClassifierModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Compute classification metrics at a given threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def find_optimal_threshold(
    model: ClassifierModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_recall: float = 0.85,
) -> Dict[str, Any]:
    """Select the threshold that hits min_recall with maximum precision."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    mask = recalls >= min_recall
    if mask.any():
        valid_indices = np.where(mask)[0]
        best_idx = valid_indices[np.argmax(precisions[mask])]
        strategy = f"max precision where recall >= {min_recall}"
    else:
        best_idx = np.argmax(f1_scores)
        strategy = "max F1 (recall target not achievable)"

    optimal_threshold = float(thresholds[best_idx])

    sweep = []
    for t in sorted(set([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, round(optimal_threshold, 2)])):
        idx = np.argmin(np.abs(thresholds - t))
        sweep.append({
            "threshold": round(t, 2),
            "precision": round(float(precisions[idx]), 4),
            "recall": round(float(recalls[idx]), 4),
            "f1": round(float(f1_scores[idx]), 4),
        })

    return {
        "optimal_threshold": round(optimal_threshold, 4),
        "strategy": strategy,
        "at_optimal": {
            "precision": round(float(precisions[best_idx]), 4),
            "recall": round(float(recalls[best_idx]), 4),
            "f1": round(float(f1_scores[best_idx]), 4),
        },
        "threshold_sweep": sweep,
    }


def plot_precision_recall_curve(
    model: ClassifierModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_threshold: float,
    save_path=None,
) -> Path:
    """Generate and save PR / ROC curves for the production model."""
    y_proba = model.predict_proba(X_test)[:, 1]

    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.plot(recalls, precisions, "b-", linewidth=2, label=f"AP = {avg_prec:.4f}")
    opt_idx = np.argmin(np.abs(pr_thresholds - optimal_threshold))
    ax.plot(recalls[opt_idx], precisions[opt_idx], "r*", markersize=15,
            label=f"Optimal (t={optimal_threshold:.2f})")
    def_idx = np.argmin(np.abs(pr_thresholds - 0.5))
    ax.plot(recalls[def_idx], precisions[def_idx], "go", markersize=10, label="Default (t=0.50)")
    ax.set_xlabel("Recall (Default Capture Rate)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall (Gradient Boosting — production)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])

    ax = axes[1]
    ax.plot(pr_thresholds, precisions[:-1], "b-", linewidth=2, label="Precision")
    ax.plot(pr_thresholds, recalls[:-1], "r-", linewidth=2, label="Recall")
    f1v = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    ax.plot(pr_thresholds, f1v, "g--", linewidth=2, label="F1")
    ax.axvline(x=optimal_threshold, color="purple", linestyle="--", linewidth=1.5,
               label=f"Optimal = {optimal_threshold:.2f}")
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1.5, label="Default = 0.50")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve (Gradient Boosting)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = MODEL_DIR.parent / "data" / "precision_recall_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR / ROC curves saved to %s", save_path)
    return save_path


def explain_prediction(
    model: GradientBoostingClassifier,
    X_sample: pd.DataFrame,
    feature_names: list,
) -> Dict[str, Any]:
    """SHAP TreeExplainer for the production Gradient Boosting model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = np.asarray(shap_values[1], dtype=float)
    else:
        shap_vals = np.asarray(shap_values, dtype=float)

    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals, dtype=float).ravel()

    row_vals = np.asarray(X_sample.iloc[0].to_numpy(dtype=float), dtype=float).ravel()

    feature_impacts = sorted(
        zip(feature_names, shap_vals, row_vals),
        key=lambda x: abs(float(np.asarray(x[1]).squeeze())),
        reverse=True,
    )

    top_factors = []
    for name, impact, value in feature_impacts[:8]:
        imp = float(np.asarray(impact).squeeze())
        val = float(np.asarray(value).squeeze())
        direction = "increases" if imp > 0 else "decreases"
        top_factors.append({
            "feature": name,
            "value": round(val, 4),
            "shap_impact": round(imp, 4),
            "direction": f"{direction} default risk",
        })

    expected = explainer.expected_value
    if isinstance(expected, np.ndarray):
        base_val = float(expected[0]) if expected.size == 1 else float(expected.flat[0])
    else:
        base_val = float(expected)

    return {
        "base_value": round(base_val, 4),
        "top_risk_factors": top_factors,
    }


def save_model(
    model: GradientBoostingClassifier,
    scaler,
    feature_names: List[str],
    scaled_columns: List[str],
    threshold: float = DEFAULT_THRESHOLD,
    calibrated_model: "CalibratedClassifierCV | None" = None,
    metrics: "Dict[str, Any] | None" = None,
) -> Dict[str, str]:
    """Persist production model artifacts.

    Saves calibrated model for scoring (risk_model.joblib) and raw model
    for SHAP TreeExplainer (risk_model_raw.joblib).
    Optionally saves evaluation metrics (model_metrics.joblib) for the monitoring UI.
    """
    paths = {
        "model":       MODEL_DIR / "risk_model.joblib",
        "model_raw":   MODEL_DIR / "risk_model_raw.joblib",
        "scaler":      MODEL_DIR / "scaler.joblib",
        "features":    MODEL_DIR / "feature_names.joblib",
        "scaled_cols": MODEL_DIR / "scaled_columns.joblib",
        "threshold":   MODEL_DIR / "threshold.joblib",
    }
    joblib.dump(calibrated_model if calibrated_model is not None else model, paths["model"])
    joblib.dump(model, paths["model_raw"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(feature_names, paths["features"])
    joblib.dump(scaled_columns, paths["scaled_cols"])
    joblib.dump(threshold, paths["threshold"])

    if metrics is not None:
        metrics_path = MODEL_DIR / "model_metrics.joblib"
        joblib.dump(metrics, metrics_path)
        paths["metrics"] = metrics_path

    logger.info("Model artifacts saved to %s (threshold=%.4f, calibrated=%s)",
                MODEL_DIR, threshold, calibrated_model is not None)
    return {k: str(v) for k, v in paths.items()}


def load_model_metrics() -> "Dict[str, Any] | None":
    """Load persisted evaluation metrics saved at training time."""
    path = MODEL_DIR / "model_metrics.joblib"
    return joblib.load(path) if path.exists() else None


def load_model() -> Tuple[Any, Any, List[str], List[str], GradientBoostingClassifier]:
    """Load persisted production model artifacts.

    Returns (scoring_model, scaler, feature_names, scaled_columns, shap_model)
    where scoring_model is calibrated (if available) and shap_model is raw GB.
    """
    model = joblib.load(MODEL_DIR / "risk_model.joblib")
    raw_path = MODEL_DIR / "risk_model_raw.joblib"
    shap_model = joblib.load(raw_path) if raw_path.exists() else model
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    scaled_columns = joblib.load(MODEL_DIR / "scaled_columns.joblib")
    return model, scaler, feature_names, scaled_columns, shap_model


def load_threshold() -> float:
    """Load the persisted optimal threshold."""
    path = MODEL_DIR / "threshold.joblib"
    return joblib.load(path) if path.exists() else DEFAULT_THRESHOLD

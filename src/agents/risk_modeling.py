"""Risk Modeling Agent – runs the trained model and generates SHAP explanations."""

import logging
from typing import Dict, Any
from datetime import datetime, timezone

import pandas as pd

from src.risk_model import load_model, load_threshold, explain_prediction

logger = logging.getLogger(__name__)


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Score the applicant using the trained Gradient Boosting production model."""
    trace_entry = {
        "agent": "RiskModelingAgent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    model, scaler, feature_names, scaled_columns = load_model()
    threshold = load_threshold()

    feature_vector = state["feature_vector"]

    # Align features in the exact order the model expects
    ordered = {fname: feature_vector.get(fname, 0) for fname in feature_names}
    X = pd.DataFrame([ordered])

    # Scale only the columns the scaler was fit on
    X_scaled = X.copy()
    cols_to_scale = [c for c in scaled_columns if c in X_scaled.columns]
    X_scaled[cols_to_scale] = scaler.transform(X_scaled[cols_to_scale])

    proba = float(model.predict_proba(X_scaled)[:, 1][0])
    prediction = int(proba >= threshold)

    if proba < 0.3:
        risk_tier = "LOW"
    elif proba < 0.6:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "HIGH"

    confidence = round(abs(proba - threshold) / max(threshold, 1 - threshold), 4)

    explanation = explain_prediction(model, X_scaled, feature_names)

    trace_entry["status"] = "completed"
    trace_entry["risk_score"] = round(proba, 4)
    trace_entry["risk_tier"] = risk_tier

    logger.info(
        "RiskModelingAgent: score=%.4f, tier=%s, confidence=%.4f",
        proba, risk_tier, confidence,
    )

    return {
        "risk_score": round(proba, 4),
        "risk_tier": risk_tier,
        "prediction": prediction,
        "confidence": confidence,
        "shap_explanation": explanation,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }

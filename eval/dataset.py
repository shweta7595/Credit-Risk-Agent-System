"""Evaluation dataset for the credit risk pipeline.

v1 (credit-risk-golden-set)  — extreme clear-cut cases, all score 1.0 by design (sanity check only)
v2 (credit-risk-eval-v2)     — challenging borderline cases that expose real model gaps
"""

from __future__ import annotations
from typing import Any, Dict, List

# ── v1: sanity-check extremes ─────────────────────────────────────────────────
GOLDEN_EXAMPLES: List[Dict[str, Any]] = [
    {
        "inputs": {"applicant": {
            "person_age": 35, "person_income": 120000,
            "person_home_ownership": "MORTGAGE", "person_emp_length": 10.0,
            "loan_intent": "HOMEIMPROVEMENT", "loan_grade": "A",
            "loan_amnt": 8000, "loan_int_rate": 7.5, "loan_percent_income": 0.07,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 12,
        }},
        "outputs": {"expected_decision": "APPROVE", "expected_risk_tier": "LOW",
                    "expected_policy_passed": True, "expected_score_max": 0.30},
    },
    {
        "inputs": {"applicant": {
            "person_age": 28, "person_income": 32000,
            "person_home_ownership": "RENT", "person_emp_length": 1.0,
            "loan_intent": "PERSONAL", "loan_grade": "F",
            "loan_amnt": 18000, "loan_int_rate": 22.5, "loan_percent_income": 0.56,
            "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 2,
        }},
        "outputs": {"expected_decision": "DECLINE", "expected_risk_tier": "HIGH",
                    "expected_policy_passed": False, "expected_score_min": 0.60},
    },
    {
        "inputs": {"applicant": {
            "person_age": 42, "person_income": 65000,
            "person_home_ownership": "RENT", "person_emp_length": 5.0,
            "loan_intent": "EDUCATION", "loan_grade": "C",
            "loan_amnt": 14000, "loan_int_rate": 13.5, "loan_percent_income": 0.22,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6,
        }},
        "outputs": {"expected_decision": "MANUAL_REVIEW", "expected_risk_tier": "MEDIUM",
                    "expected_policy_passed": True,
                    "expected_score_min": 0.30, "expected_score_max": 0.70},
    },
    {
        "inputs": {"applicant": {
            "person_age": 24, "person_income": 28000,
            "person_home_ownership": "RENT", "person_emp_length": 0.5,
            "loan_intent": "VENTURE", "loan_grade": "G",
            "loan_amnt": 25000, "loan_int_rate": 26.0, "loan_percent_income": 0.89,
            "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 1,
        }},
        "outputs": {"expected_decision": "DECLINE", "expected_risk_tier": "HIGH",
                    "expected_policy_passed": False, "expected_score_min": 0.85},
    },
    {
        "inputs": {"applicant": {
            "person_age": 50, "person_income": 95000,
            "person_home_ownership": "OWN", "person_emp_length": 15.0,
            "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "B",
            "loan_amnt": 12000, "loan_int_rate": 10.0, "loan_percent_income": 0.13,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 20,
        }},
        "outputs": {"expected_decision": "APPROVE", "expected_risk_tier": "LOW",
                    "expected_policy_passed": True, "expected_score_max": 0.35},
    },
]

# ── v2: challenging borderline cases ──────────────────────────────────────────
# These expose model weaknesses: borderline PTI, mixed signals, policy edge cases.
CHALLENGING_EXAMPLES: List[Dict[str, Any]] = [
    # Good income, but grade D + moderate PTI — borderline APPROVE vs MANUAL_REVIEW
    {
        "inputs": {"applicant": {
            "person_age": 38, "person_income": 72000,
            "person_home_ownership": "RENT", "person_emp_length": 4.0,
            "loan_intent": "PERSONAL", "loan_grade": "D",
            "loan_amnt": 15000, "loan_int_rate": 16.0, "loan_percent_income": 0.21,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 7,
        }},
        "outputs": {"expected_decision": "MANUAL_REVIEW", "expected_risk_tier": "MEDIUM",
                    "expected_policy_passed": True,
                    "expected_score_min": 0.25, "expected_score_max": 0.65},
    },
    # Prior default BUT high income + long history — model should penalise but not hard-decline
    {
        "inputs": {"applicant": {
            "person_age": 45, "person_income": 110000,
            "person_home_ownership": "MORTGAGE", "person_emp_length": 12.0,
            "loan_intent": "HOMEIMPROVEMENT", "loan_grade": "C",
            "loan_amnt": 20000, "loan_int_rate": 12.0, "loan_percent_income": 0.18,
            "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 15,
        }},
        "outputs": {"expected_decision": "MANUAL_REVIEW", "expected_risk_tier": "MEDIUM",
                    "expected_policy_passed": True,
                    "expected_score_min": 0.25, "expected_score_max": 0.70},
    },
    # High PTI (0.45) but grade A — policy may flag, model may approve
    {
        "inputs": {"applicant": {
            "person_age": 30, "person_income": 44000,
            "person_home_ownership": "RENT", "person_emp_length": 3.0,
            "loan_intent": "MEDICAL", "loan_grade": "A",
            "loan_amnt": 20000, "loan_int_rate": 8.0, "loan_percent_income": 0.45,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 5,
        }},
        "outputs": {"expected_decision": "MANUAL_REVIEW", "expected_risk_tier": "MEDIUM",
                    "expected_policy_passed": True,
                    "expected_score_min": 0.15, "expected_score_max": 0.60},
    },
    # Young applicant, short history, grade E, but small loan + low PTI
    {
        "inputs": {"applicant": {
            "person_age": 22, "person_income": 38000,
            "person_home_ownership": "RENT", "person_emp_length": 1.0,
            "loan_intent": "EDUCATION", "loan_grade": "E",
            "loan_amnt": 5000, "loan_int_rate": 18.0, "loan_percent_income": 0.13,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 1,
        }},
        "outputs": {"expected_decision": "MANUAL_REVIEW", "expected_risk_tier": "MEDIUM",
                    "expected_policy_passed": True,
                    "expected_score_min": 0.30, "expected_score_max": 0.75},
    },
    # Edge: score just above auto-decline threshold (0.85) — must DECLINE
    {
        "inputs": {"applicant": {
            "person_age": 32, "person_income": 35000,
            "person_home_ownership": "RENT", "person_emp_length": 2.0,
            "loan_intent": "VENTURE", "loan_grade": "F",
            "loan_amnt": 22000, "loan_int_rate": 24.0, "loan_percent_income": 0.63,
            "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 3,
        }},
        "outputs": {"expected_decision": "DECLINE", "expected_risk_tier": "HIGH",
                    "expected_policy_passed": False, "expected_score_min": 0.75},
    },
    # Strong profile except very new employment — tests model sensitivity to emp_length
    {
        "inputs": {"applicant": {
            "person_age": 29, "person_income": 85000,
            "person_home_ownership": "OWN", "person_emp_length": 0.0,
            "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "B",
            "loan_amnt": 10000, "loan_int_rate": 9.5, "loan_percent_income": 0.12,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 4,
        }},
        "outputs": {"expected_decision": "MANUAL_REVIEW", "expected_risk_tier": "LOW",
                    "expected_policy_passed": True,
                    "expected_score_max": 0.45},
    },
]


def create_or_get_dataset(client, dataset_name: str = "credit-risk-golden-set",
                          examples: List[Dict[str, Any]] = None):
    """Push examples to LangSmith; return existing dataset if already present."""
    if examples is None:
        examples = GOLDEN_EXAMPLES

    existing = [d for d in client.list_datasets() if d.name == dataset_name]
    if existing:
        return existing[0]

    dataset = client.create_dataset(
        dataset_name,
        description="Credit risk underwriting evaluation dataset.",
    )
    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id,
    )
    return dataset

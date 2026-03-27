"""Policy Validation Agent – checks the application against underwriting policies."""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


def _load_policy_text(filename: str) -> str:
    path = DOCS_DIR / filename
    if path.exists():
        return path.read_text()
    return ""


def _check_credit_policy(raw: Dict[str, Any]) -> List[str]:
    """Validate against credit_policy.md rules."""
    violations = []

    age = raw.get("person_age", 0)
    if age < 21:
        violations.append(f"POLICY_VIOLATION: Applicant age {age} below minimum 21")
    if age > 65:
        violations.append(f"POLICY_VIOLATION: Applicant age {age} above maximum 65")

    income = raw.get("person_income", 0)
    if income < 12000:
        violations.append(f"POLICY_VIOLATION: Annual income ${income:,} below minimum $12,000")

    emp_length = raw.get("person_emp_length", 0)
    if emp_length < 1:
        violations.append(f"POLICY_WARNING: Employment length {emp_length} years below recommended 1 year")

    dti = raw.get("loan_percent_income", 0)
    if dti > 0.43:
        violations.append(f"POLICY_VIOLATION: DTI ratio {dti:.2f} exceeds maximum 0.43")
    if dti > 0.50:
        violations.append(f"POLICY_HARD_STOP: DTI ratio {dti:.2f} exceeds absolute limit 0.50")

    loan_amnt = raw.get("loan_amnt", 0)
    if loan_amnt < 500:
        violations.append(f"POLICY_VIOLATION: Loan amount ${loan_amnt:,} below minimum $500")
    if loan_amnt > 35000:
        violations.append(f"POLICY_VIOLATION: Loan amount ${loan_amnt:,} above maximum $35,000")

    lpi = raw.get("loan_percent_income", 0)
    if lpi > 0.50:
        violations.append(f"POLICY_VIOLATION: Loan-to-income {lpi:.2f} exceeds limit 0.50")

    grade = raw.get("loan_grade", "")
    if grade in ("E", "F", "G"):
        violations.append(f"POLICY_WARNING: Loan grade {grade} requires additional review")

    default_flag = raw.get("cb_person_default_on_file", 0)
    if default_flag == 1 and grade in ("D", "E", "F", "G"):
        violations.append("POLICY_WARNING: Prior default + low grade triggers manual review")

    return violations


def _check_regulatory(raw: Dict[str, Any]) -> List[str]:
    """Check regulatory constraints."""
    violations = []
    # Age should not be sole decision factor
    age = raw.get("person_age", 30)
    if age < 25 or age > 60:
        violations.append(
            f"REGULATORY_NOTE: Age {age} is at boundary; ensure it is not the sole factor"
        )
    return violations


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the application against all policies and regulations."""
    raw = state["raw_data"]
    risk_score = state.get("risk_score", 0.5)

    trace_entry = {
        "agent": "PolicyValidationAgent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    violations = []
    violations.extend(_check_credit_policy(raw))
    violations.extend(_check_regulatory(raw))

    # Risk-score-based auto-rules
    if risk_score > 0.85:
        violations.append(f"POLICY_HARD_STOP: Risk score {risk_score:.4f} exceeds auto-decline threshold 0.85")

    # Determine preliminary decision
    hard_stops = [v for v in violations if "HARD_STOP" in v]
    policy_violations_only = [v for v in violations if "VIOLATION" in v]
    warnings = [v for v in violations if "WARNING" in v or "NOTE" in v]

    if hard_stops:
        decision = "DECLINE"
    elif not policy_violations_only and not warnings and risk_score < 0.25:
        grade = raw.get("loan_grade", "C")
        dti = raw.get("loan_percent_income", 0.5)
        default_flag = raw.get("cb_person_default_on_file", 1)
        emp = raw.get("person_emp_length", 0)
        if grade in ("A", "B") and dti <= 0.20 and default_flag == 0 and emp >= 3:
            decision = "APPROVE"
        else:
            decision = "MANUAL_REVIEW"
    elif policy_violations_only:
        decision = "DECLINE"
    else:
        decision = "MANUAL_REVIEW"

    policy_passed = len(hard_stops) == 0 and len(policy_violations_only) == 0

    trace_entry["status"] = "completed"
    trace_entry["violations_count"] = len(violations)
    trace_entry["decision"] = decision

    logger.info(
        "PolicyValidationAgent: %d violations, decision=%s", len(violations), decision
    )

    return {
        "policy_violations": violations,
        "policy_passed": policy_passed,
        "decision": decision,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }

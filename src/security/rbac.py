"""Role-based access control for underwriting UI and API consumers.

Roles (least privilege):
- viewer: high-level outcome only
- underwriter: full operational detail (SHAP, explanation, violations)
- auditor: underwriter + LLM judge + agent trace
- admin: same as auditor (extend later for config)

User directory: JSON in env APP_RBAC_USERS, e.g.
{"jsmith":{"password":"secret","role":"underwriter"},"guest":{"password":"x","role":"viewer"}}

For production, replace password values with bcrypt or delegate to IdP; this is a demo-friendly layout.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Role(str, Enum):
    VIEWER = "viewer"
    UNDERWRITER = "underwriter"
    AUDITOR = "auditor"
    ADMIN = "admin"


def _load_user_db() -> Dict[str, Dict[str, str]]:
    raw = os.getenv("APP_RBAC_USERS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {k.lower(): v for k, v in data.items() if isinstance(v, dict)}
    except json.JSONDecodeError:
        logger.warning("APP_RBAC_USERS is not valid JSON; RBAC login disabled")
        return {}


def authenticate(username: str, password: str) -> Optional[Role]:
    """Return Role if credentials match user db; else None."""
    if not username or not password:
        return None
    db = _load_user_db()
    key = username.strip().lower()
    rec = db.get(key)
    if not rec:
        return None
    if rec.get("password") != password:
        return None
    role_str = (rec.get("role") or "viewer").lower()
    try:
        return Role(role_str)
    except ValueError:
        return Role.VIEWER


def filter_pipeline_result_for_role(result: Dict[str, Any], role: Role) -> Dict[str, Any]:
    """Return a copy of pipeline result fields allowed for this role."""
    out: Dict[str, Any] = {}

    out["decision"] = result.get("decision")
    out["risk_tier"] = result.get("risk_tier")
    out["risk_score"] = result.get("risk_score")
    out["prediction"] = result.get("prediction")
    out["confidence"] = result.get("confidence")
    out["policy_passed"] = result.get("policy_passed")

    if role == Role.VIEWER:
        violations = result.get("policy_violations") or []
        out["policy_violation_count"] = len(violations)
        out["policy_violations"] = []  # detail hidden
        out["applicant"] = _mask_applicant_viewer(result.get("applicant") or {})
        out["shap_explanation"] = {}
        out["explanation_report"] = (
            "[Withheld at VIEWER role — request an underwriter for narrative detail.]"
        )
        out["llm_judge_verdict"] = None
        out["llm_judge_rationale"] = ""
        out["llm_judge_concerns"] = []
        out["llm_judge_compliance_notes"] = ""
        out["agent_trace"] = []
        return out

    if role == Role.UNDERWRITER:
        out["policy_violations"] = result.get("policy_violations") or []
        out["applicant"] = result.get("applicant")
        out["shap_explanation"] = result.get("shap_explanation") or {}
        out["explanation_report"] = result.get("explanation_report") or ""
        out["llm_judge_verdict"] = result.get("llm_judge_verdict")
        out["llm_judge_rationale"] = (result.get("llm_judge_rationale") or "")[:800]
        out["llm_judge_concerns"] = result.get("llm_judge_concerns") or []
        out["llm_judge_compliance_notes"] = ""
        out["agent_trace"] = []
        return out

    # auditor / admin
    out["policy_violations"] = result.get("policy_violations") or []
    out["applicant"] = result.get("applicant")
    out["shap_explanation"] = result.get("shap_explanation") or {}
    out["explanation_report"] = result.get("explanation_report") or ""
    out["llm_judge_verdict"] = result.get("llm_judge_verdict")
    out["llm_judge_rationale"] = result.get("llm_judge_rationale") or ""
    out["llm_judge_concerns"] = result.get("llm_judge_concerns") or []
    out["llm_judge_compliance_notes"] = result.get("llm_judge_compliance_notes") or ""
    out["agent_trace"] = result.get("agent_trace") or []
    return out


def _mask_applicant_viewer(applicant: Dict[str, Any]) -> Dict[str, Any]:
    """Show coarse bands only for sensitive numerics."""
    masked = dict(applicant)
    inc = applicant.get("person_income")
    if isinstance(inc, (int, float)):
        if inc < 40000:
            masked["person_income_band"] = "<$40k"
        elif inc < 80000:
            masked["person_income_band"] = "$40k–$80k"
        else:
            masked["person_income_band"] = ">$80k"
        masked.pop("person_income", None)
    lam = applicant.get("loan_amnt")
    if isinstance(lam, (int, float)):
        masked["loan_amnt_band"] = "<$10k" if lam < 10000 else "≥$10k"
        masked.pop("loan_amnt", None)
    return masked


def role_display_name(role: Role) -> str:
    return role.value.replace("_", " ").title()

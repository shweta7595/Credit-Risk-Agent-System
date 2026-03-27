"""Streamlit chat UI for credit underwriting (RBAC + pipeline + PII redaction on chat).

Run from repository root:
  streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.agents.graph import run_pipeline
from src.security import (
    Role,
    authenticate,
    filter_pipeline_result_for_role,
    redact_pii,
)
MODEL_PATH = ROOT / "models" / "risk_model.joblib"

HOME_OPTS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
INTENT_OPTS = [
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
]
GRADE_OPTS = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_OPTS = ["N", "Y"]


def _format_underwriting_markdown(filtered: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"**Decision:** `{filtered.get('decision')}`",
        f"**Risk tier:** {filtered.get('risk_tier')}",
        f"**Default probability:** {filtered.get('risk_score')}",
        f"**Binary prediction (1=default):** {filtered.get('prediction')}",
        f"**Confidence:** {filtered.get('confidence')}",
        f"**Policy passed:** {filtered.get('policy_passed')}",
    ]
    vc = filtered.get("policy_violation_count")
    if vc is not None:
        lines.append(f"**Policy violation count:** {vc}")
    pviol = filtered.get("policy_violations") or []
    if pviol:
        lines.append("\n**Policy violations:**")
        for v in pviol:
            lines.append(f"- {v}")
    exp = (filtered.get("explanation_report") or "").strip()
    if exp:
        lines.append("\n**Explanation report:**\n\n" + exp)
    shap = filtered.get("shap_explanation") or {}
    factors = shap.get("top_risk_factors") or []
    if factors:
        lines.append("\n**Top model factors (SHAP):**")
        for f in factors[:10]:
            lines.append(
                f"- **{f.get('feature')}** — impact {f.get('shap_impact')}, {f.get('direction', '')}"
            )
    jv = filtered.get("llm_judge_verdict")
    if jv:
        lines.append("\n**LLM judge:**")
        lines.append(f"- Verdict: `{jv}`")
        jr = filtered.get("llm_judge_rationale") or ""
        if jr:
            lines.append(f"- Rationale: {jr}")
        concerns = filtered.get("llm_judge_concerns") or []
        if concerns:
            lines.append("- Concerns:")
            for c in concerns:
                lines.append(f"  - {c}")
        cn = filtered.get("llm_judge_compliance_notes") or ""
        if cn:
            lines.append(f"- Compliance notes: {cn}")
    trace = filtered.get("agent_trace") or []
    if trace:
        lines.append("\n**Agent trace:**")
        for step in trace:
            lines.append(
                f"- [{step.get('agent')}] {step.get('status')} @ {str(step.get('timestamp', ''))[:19]}"
            )
    return "\n".join(lines)


def _applicant_from_form() -> Dict[str, Any]:
    return {
        "person_age": int(st.session_state.get("f_age", 35)),
        "person_income": float(st.session_state.get("f_income", 60000)),
        "person_home_ownership": st.session_state.get("f_home", "RENT"),
        "person_emp_length": float(st.session_state.get("f_emp", 5.0)),
        "loan_intent": st.session_state.get("f_intent", "PERSONAL"),
        "loan_grade": st.session_state.get("f_grade", "C"),
        "loan_amnt": float(st.session_state.get("f_amnt", 10000)),
        "loan_int_rate": float(st.session_state.get("f_rate", 12.0)),
        "loan_percent_income": float(st.session_state.get("f_pti", 0.2)),
        "cb_person_default_on_file": st.session_state.get("f_def", "N"),
        "cb_person_cred_hist_length": int(st.session_state.get("f_hist", 5)),
    }


def _default_welcome_messages() -> List[Dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Welcome. Use **Applicant profile** in the sidebar, then **Run underwriting**. "
                "Results appear here according to your role. "
                "Free-text below is redacted for common PII patterns before display."
            ),
        }
    ]


def _init_session() -> None:
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "messages" not in st.session_state:
        st.session_state.messages = _default_welcome_messages()


def _login_ui() -> None:
    st.title("Credit risk — underwriting chat")
    db_raw = os.getenv("APP_RBAC_USERS", "").strip()
    if not db_raw:
        st.warning(
            "`APP_RBAC_USERS` is not set — **development mode**. "
            "Pick a role below; configure JSON user map in `.env` for real login."
        )
        role_key = st.selectbox(
            "Role (demo)",
            options=["viewer", "underwriter", "auditor", "admin"],
            index=1,
        )
        if st.button("Continue"):
            st.session_state.auth_role = Role(role_key)
            st.session_state.auth_user = f"demo_{role_key}"
            st.rerun()
        return

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Sign in"):
        role = authenticate(u, p)
        if role is None:
            st.error("Invalid username or password.")
            return
        st.session_state.auth_role = role
        st.session_state.auth_user = u.strip()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Credit underwriting", layout="wide")
    _init_session()
    if st.session_state.auth_role is None:
        _login_ui()
        return

    role: Role = st.session_state.auth_role

    with st.sidebar:
        st.caption(f"Signed in as **{st.session_state.auth_user}** ({role.value})")
        if st.button("Sign out"):
            st.session_state.auth_role = None
            st.session_state.auth_user = None
            st.session_state.messages = _default_welcome_messages()
            st.rerun()

        st.divider()
        st.subheader("Applicant profile")
        st.number_input("Age", min_value=18, max_value=100, value=35, key="f_age")
        st.number_input("Annual income", min_value=0.0, value=60000.0, step=1000.0, key="f_income")
        st.selectbox("Home ownership", HOME_OPTS, key="f_home")
        st.number_input("Employment length (years)", min_value=0.0, value=5.0, step=0.5, key="f_emp")
        st.selectbox("Loan intent", INTENT_OPTS, key="f_intent")
        st.selectbox("Loan grade", GRADE_OPTS, index=2, key="f_grade")
        st.number_input("Loan amount", min_value=0.0, value=10000.0, step=500.0, key="f_amnt")
        st.number_input("Interest rate (%)", min_value=0.0, value=12.0, step=0.1, key="f_rate")
        st.number_input("Loan as fraction of income", min_value=0.0, max_value=2.0, value=0.2, step=0.01, key="f_pti")
        st.selectbox("Prior default on file", DEFAULT_OPTS, key="f_def")
        st.number_input("Credit history length (years)", min_value=0, value=5, key="f_hist")

        missing_model = not MODEL_PATH.exists()
        if missing_model:
            st.error("Train the model first: `python -m src.train`")
        run_btn = st.button("Run underwriting", type="primary", disabled=missing_model)

    st.title("Underwriting chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if run_btn:
        applicant = _applicant_from_form()
        summary = (
            f"Run underwriting for profile: age {applicant['person_age']}, "
            f"income {applicant['person_income']}, loan {applicant['loan_amnt']}, grade {applicant['loan_grade']}."
        )
        st.session_state.messages.append({"role": "user", "content": summary})
        with st.spinner("Running pipeline…"):
            raw = run_pipeline(applicant)
            filtered = filter_pipeline_result_for_role(raw, role)
            md = _format_underwriting_markdown(filtered)
        st.session_state.messages.append({"role": "assistant", "content": md})
        st.rerun()

    chat_in = st.chat_input("Notes (PII will be redacted in the log below)")
    if chat_in:
        redacted, n = redact_pii(chat_in)
        note = redacted
        if n:
            note += f"\n\n_(Redacted {n} PII-like pattern(s).)_"
        st.session_state.messages.append({"role": "user", "content": note})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "This panel records notes only. To score an applicant, use **Run underwriting** "
                    "in the sidebar with the structured form (no raw PII in free text)."
                ),
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()

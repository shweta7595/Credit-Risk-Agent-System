"""Streamlit chat UI for credit underwriting (RBAC + pipeline + PII redaction on chat).

Run from repository root:
  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)


def _apply_streamlit_secrets_to_environ() -> None:
    """Copy Streamlit (Cloud) secrets into os.environ — agents use os.getenv only."""
    try:
        root = dict(st.secrets)
    except Exception:
        return

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, val in node.items():
            sk = str(key)
            if sk.startswith("_"):
                continue
            if isinstance(val, str) and val.strip():
                os.environ[sk] = val.strip()
            elif isinstance(val, dict):
                if sk == "APP_RBAC_USERS":
                    os.environ["APP_RBAC_USERS"] = json.dumps(val)
                else:
                    walk(val)

    walk(root)


_apply_streamlit_secrets_to_environ()

from src.agents.graph import run_pipeline
from src.agents.llm_provider import llm_credentials_ok  # noqa: F401 – kept for pipeline
from src.risk_model import load_model_metrics
from src.security import Role, authenticate, filter_pipeline_result_for_role, redact_pii

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH    = ROOT / "models" / "risk_model.joblib"
PR_CURVE_PATH = ROOT / "data" / "pr_curve_calibrated.png"
LOG_PATH      = ROOT / "data" / "session_log.jsonl"

# ── cross-page navigation registry (populated in main() before pg.run()) ───────
_PAGES: Dict[str, Any] = {}

# ── form options ───────────────────────────────────────────────────────────────
HOME_OPTS    = ["RENT", "MORTGAGE", "OWN", "OTHER"]
INTENT_OPTS  = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
GRADE_OPTS   = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_OPTS = ["N", "Y"]


# ── cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _model_path_exists() -> bool:
    return MODEL_PATH.exists()


@st.cache_data(ttl=300)
def _cached_model_metrics() -> "Dict[str, Any] | None":
    return load_model_metrics()


# ── persistent log (JSONL — survives restarts, accumulates across all users) ───
def _append_log(entry: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_log(n: int = 2000) -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows[-n:]


# ── formatting helpers ─────────────────────────────────────────────────────────
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
    return "\n".join(lines)


def _applicant_from_form() -> Dict[str, Any]:
    return {
        "person_age":                int(st.session_state.get("f_age", 35)),
        "person_income":             float(st.session_state.get("f_income", 60000)),
        "person_home_ownership":     st.session_state.get("f_home", "RENT"),
        "person_emp_length":         float(st.session_state.get("f_emp", 5.0)),
        "loan_intent":               st.session_state.get("f_intent", "PERSONAL"),
        "loan_grade":                st.session_state.get("f_grade", "C"),
        "loan_amnt":                 float(st.session_state.get("f_amnt", 10000)),
        "loan_int_rate":             float(st.session_state.get("f_rate", 12.0)),
        "loan_percent_income":       float(st.session_state.get("f_pti", 0.2)),
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
    st.session_state.setdefault("auth_role", None)
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("run_count", 0)
    if "messages" not in st.session_state:
        st.session_state.messages = _default_welcome_messages()


# ── login ──────────────────────────────────────────────────────────────────────
def _login_ui() -> None:
    st.title("CreditGenie Risk Assistant")
    st.caption("Role-based support for underwriting and credit analysis.")
    st.info("Select a role to enter the workspace.")

    db_raw = os.getenv("APP_RBAC_USERS", "").strip()
    if not db_raw:
        role_key = st.selectbox(
            "Select your role",
            options=["viewer", "underwriter", "auditor", "admin"],
            index=1,
        )
        if st.button("Launch CreditGenie", type="primary"):
            st.session_state.auth_role = Role(role_key)
            st.session_state.auth_user = f"demo_{role_key}"
            st.rerun()
        return

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    role_key = st.selectbox(
        "Select your role",
        options=["viewer", "underwriter", "auditor", "admin"],
        index=1,
    )
    if st.button("Launch CreditGenie", type="primary"):
        role = authenticate(u, p)
        if role is None:
            st.error("Invalid username or password.")
            return
        st.session_state.auth_role = role
        st.session_state.auth_user = u.strip()
        st.rerun()


# ── shared sidebar sign-out ────────────────────────────────────────────────────
def _sidebar_signout() -> None:
    if st.button("Sign out", key="signout"):
        st.session_state.auth_role = None
        st.session_state.auth_user = None
        st.session_state.messages  = _default_welcome_messages()
        st.session_state.run_count = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Underwriting Workspace
# ══════════════════════════════════════════════════════════════════════════════
def _workspace_page() -> None:
    role: Role = st.session_state.auth_role
    user: str  = st.session_state.auth_user or "—"

    # ── sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        _sidebar_signout()
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

        missing_model = not _model_path_exists()
        if missing_model:
            st.error("Train the model first: `python -m src.train`")
        run_btn = st.button("Run underwriting", type="primary", disabled=missing_model)

    # ── admin monitoring button — top-right, above the title ───────────────────
    if role == Role.ADMIN:
        _, btn_col = st.columns([5, 1])
        with btn_col:
            if st.button(
                "📊 Risk Model Monitoring",
                type="primary",
                use_container_width=True,
                key="btn_goto_monitoring",
            ):
                st.switch_page(_PAGES["monitoring"])

    st.title("CreditGenie — Underwriting Workspace")

    # ── chat messages ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            trace = msg.get("trace")
            if trace:
                with st.expander("View agent trace"):
                    for step in trace:
                        st.caption(
                            f"[{step.get('agent')}]  {step.get('status')}  "
                            f"@ {str(step.get('timestamp', ''))[:19]}"
                        )

    # ── pipeline run ───────────────────────────────────────────────────────────
    if run_btn:
        applicant = _applicant_from_form()
        st.session_state.run_count += 1
        run_num = st.session_state.run_count
        summary = (
            f"**Run #{run_num}** — age {applicant['person_age']}, "
            f"income ${applicant['person_income']:,.0f}, "
            f"loan ${applicant['loan_amnt']:,.0f}, grade {applicant['loan_grade']}."
        )
        st.session_state.messages.append({"role": "user", "content": summary})
        with st.spinner("Running pipeline…"):
            raw      = run_pipeline(applicant)
            filtered = filter_pipeline_result_for_role(raw, role)
            md       = _format_underwriting_markdown(filtered)
        trace = filtered.get("agent_trace") or []
        st.session_state.messages.append({
            "role":    "assistant",
            "content": md,
            "trace":   trace if trace else None,
        })
        _append_log({
            "timestamp": datetime.now().isoformat(),
            "date":      datetime.now().strftime("%Y-%m-%d"),
            "time":      datetime.now().strftime("%H:%M:%S"),
            "user":      user,
            "role":      role.value,
            "run":       run_num,
            "age":       applicant["person_age"],
            "income":    applicant["person_income"],
            "loan":      applicant["loan_amnt"],
            "grade":     applicant["loan_grade"],
            "intent":    applicant["loan_intent"],
            "decision":  filtered.get("decision", "—"),
            "score":     filtered.get("risk_score"),
            "tier":      filtered.get("risk_tier", "—"),
            "policy":    "Pass" if filtered.get("policy_passed") else "Fail",
        })
        st.rerun()

    # ── free-text notes ────────────────────────────────────────────────────────
    chat_in = st.chat_input("Notes (PII will be redacted in the log below)")
    if chat_in:
        redacted, n = redact_pii(chat_in)
        note = redacted + (f"\n\n_(Redacted {n} PII-like pattern(s).)_" if n else "")
        st.session_state.messages.append({"role": "user", "content": note})
        st.session_state.messages.append({
            "role":    "assistant",
            "content": (
                "This panel records notes only. To score an applicant, use **Run underwriting** "
                "in the sidebar with the structured form (no raw PII in free text)."
            ),
        })
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Admin Monitoring
# ══════════════════════════════════════════════════════════════════════════════
def _monitoring_page() -> None:
    if st.session_state.auth_role != Role.ADMIN:
        st.error("Admin access required.")
        st.stop()

    # ── sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        if st.button("← Back to Workspace", key="back_to_ws"):
            st.switch_page(_PAGES["workspace"])
        st.divider()
        _sidebar_signout()

    # ── page header ────────────────────────────────────────────────────────────
    st.title("📊 Risk Model Performance Monitoring")
    st.caption(
        "Calibrated Gradient Boosting · production model metrics · underwriting activity log"
    )
    st.divider()

    tab_perf, tab_log = st.tabs(["📈  Model Performance", "📋  Session Log"])

    # ══ Tab 1: Model Performance ═══════════════════════════════════════════════
    with tab_perf:
        metrics = _cached_model_metrics()
        if metrics is None:
            st.warning("No saved metrics. Run `python -m src.train` to generate them.")
        else:
            report  = metrics.get("classification_report", {})
            no_def  = report.get("0", {})
            default = report.get("1", {})

            # row of 5 key metric cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("AUC-ROC",       f"{metrics.get('auc', 0):.4f}")
            c2.metric("Avg Precision", f"{metrics.get('avg_precision', 0):.4f}")
            c3.metric("Accuracy",      f"{metrics.get('accuracy', 0):.4f}")
            c4.metric("Threshold",     f"{metrics.get('threshold', 0):.4f}")
            c5.metric(
                "Default Recall",
                f"{default.get('recall', 0):.4f}",
                help="Share of actual defaults captured at the operating threshold.",
            )

            st.divider()
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.subheader("Per-class metrics")
                cls_rows = []
                for label, cls in [("No Default", no_def), ("Default", default)]:
                    if cls:
                        cls_rows.append({
                            "Class":     label,
                            "Precision": round(cls.get("precision", 0), 4),
                            "Recall":    round(cls.get("recall", 0), 4),
                            "F1-score":  round(cls.get("f1-score", 0), 4),
                            "Support":   int(cls.get("support", 0)),
                        })
                st.dataframe(pd.DataFrame(cls_rows), hide_index=True, use_container_width=True)

                st.subheader("Confusion matrix")
                cm = metrics.get("confusion_matrix")
                if cm:
                    cm_df = pd.DataFrame(
                        cm,
                        index   =["Actual: No Default", "Actual: Default"],
                        columns =["Pred: No Default",   "Pred: Default"],
                    )
                    st.dataframe(cm_df, use_container_width=True)

            with col_right:
                st.subheader("PR / ROC Curves")
                if PR_CURVE_PATH.exists():
                    st.image(str(PR_CURVE_PATH), use_container_width=True)
                else:
                    st.info(
                        "PR curve image not found. "
                        "Run `python -m src.train` to generate it."
                    )

    # ══ Tab 2: Session Log ═════════════════════════════════════════════════════
    with tab_log:
        rows = _load_log()
        if not rows:
            st.info("No underwriting runs recorded yet.")
        else:
            df = pd.DataFrame(rows)

            # filter bar
            fc1, fc2, fc3, fc4 = st.columns(4)
            users     = ["All"] + sorted(df["user"].unique().tolist())
            decisions = ["All"] + sorted(df["decision"].dropna().unique().tolist())
            tiers     = ["All"] + sorted(df["tier"].dropna().unique().tolist())
            dates     = ["All"] + sorted(df["date"].unique().tolist(), reverse=True)

            sel_user     = fc1.selectbox("User",     users,     key="flt_user")
            sel_decision = fc2.selectbox("Decision", decisions, key="flt_decision")
            sel_tier     = fc3.selectbox("Tier",     tiers,     key="flt_tier")
            sel_date     = fc4.selectbox("Date",     dates,     key="flt_date")

            mask = pd.Series([True] * len(df), index=df.index)
            if sel_user     != "All": mask &= df["user"]     == sel_user
            if sel_decision != "All": mask &= df["decision"] == sel_decision
            if sel_tier     != "All": mask &= df["tier"]     == sel_tier
            if sel_date     != "All": mask &= df["date"]     == sel_date

            df_view = df[mask].copy()

            col_rename = {
                "date": "Date", "time": "Time", "user": "User", "role": "Role",
                "age": "Age", "income": "Income ($)", "loan": "Loan ($)",
                "grade": "Grade", "intent": "Intent",
                "decision": "Decision", "score": "Score",
                "tier": "Tier", "policy": "Policy",
            }
            present = [c for c in col_rename if c in df_view.columns]
            df_display = df_view[present].rename(columns=col_rename)

            st.markdown(f"**{len(df_display)}** records shown &nbsp;·&nbsp; {len(df)} total")
            st.dataframe(df_display, hide_index=True, use_container_width=True, height=440)

            csv = df_display.to_csv(index=False)
            st.download_button(
                "⬇ Download as CSV",
                data=csv,
                file_name=f"session_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(page_title="CreditGenie Risk Assistant", layout="wide")
    _init_session()

    if st.session_state.auth_role is None:
        _login_ui()
        return

    role: Role = st.session_state.auth_role

    workspace = st.Page(_workspace_page, title="Underwriting Workspace", icon="🏦", default=True)
    _PAGES["workspace"] = workspace
    pages = [workspace]

    if role == Role.ADMIN:
        monitoring = st.Page(_monitoring_page, title="Risk Model Monitoring", icon="📊")
        _PAGES["monitoring"] = monitoring
        pages.append(monitoring)

    pg = st.navigation(pages, position="hidden")
    pg.run()


if __name__ == "__main__":
    main()

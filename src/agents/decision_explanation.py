"""Decision Explanation Agent – generates human-readable underwriting reports."""

import logging
from typing import Dict, Any
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.security.llm_guardrails import combine_system_message

logger = logging.getLogger(__name__)


EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        combine_system_message(
            """You are an expert credit risk underwriter generating clear, compliant
explanation reports for loan decisions. Your reports must:
- Be understandable by non-technical applicants
- Comply with adverse action notice requirements (ECOA / Reg B)
- Include the top risk factors with plain-language explanations
- State the decision, confidence, and risk tier clearly
- If declined, list the specific reasons per regulatory requirements"""
        ),
    ),
    (
        "human",
        """Generate an underwriting decision report for this application:

## Applicant Profile
- Age: {person_age}
- Annual Income: ${person_income:,}
- Home Ownership: {person_home_ownership}
- Employment Length: {person_emp_length} years
- Loan Intent: {loan_intent}
- Loan Grade: {loan_grade}
- Loan Amount: ${loan_amnt:,}
- Interest Rate: {loan_int_rate}%
- Debt-to-Income Ratio: {loan_percent_income}
- Prior Default on File: {cb_person_default_on_file}
- Credit History Length: {cb_person_cred_hist_length} years

## Risk Assessment
- Default Probability: {risk_score}
- Risk Tier: {risk_tier}
- Model Confidence: {confidence}

## Top Risk Factors (SHAP)
{shap_factors}

## Policy Check Results
- Policy Passed: {policy_passed}
- Violations: {policy_violations}

## Decision: {decision}

Please generate a comprehensive, compliant explanation report.""",
    ),
])


def _format_shap_factors(explanation: Dict[str, Any]) -> str:
    factors = explanation.get("top_risk_factors", [])
    if not factors:
        return "No SHAP factors available."
    lines = []
    for f in factors:
        lines.append(
            f"- {f['feature']}: value={f['value']}, "
            f"SHAP impact={f['shap_impact']}, {f['direction']}"
        )
    return "\n".join(lines)


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Produce the final explainability report using the LLM."""
    trace_entry = {
        "agent": "DecisionExplanationAgent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    applicant = state["applicant"]
    raw = state["raw_data"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500)

    shap_text = _format_shap_factors(state.get("shap_explanation", {}))

    cb_default_display = "Yes" if raw.get("cb_person_default_on_file", 0) == 1 else "No"

    chain = EXPLANATION_PROMPT | llm

    response = chain.invoke({
        "person_age": applicant["person_age"],
        "person_income": applicant["person_income"],
        "person_home_ownership": applicant["person_home_ownership"],
        "person_emp_length": applicant["person_emp_length"],
        "loan_intent": applicant["loan_intent"],
        "loan_grade": applicant["loan_grade"],
        "loan_amnt": applicant["loan_amnt"],
        "loan_int_rate": applicant["loan_int_rate"],
        "loan_percent_income": applicant["loan_percent_income"],
        "cb_person_default_on_file": cb_default_display,
        "cb_person_cred_hist_length": applicant["cb_person_cred_hist_length"],
        "risk_score": state.get("risk_score", "N/A"),
        "risk_tier": state.get("risk_tier", "N/A"),
        "confidence": state.get("confidence", "N/A"),
        "shap_factors": shap_text,
        "policy_passed": state.get("policy_passed", "N/A"),
        "policy_violations": "\n".join(state.get("policy_violations", [])) or "None",
        "decision": state.get("decision", "N/A"),
    })

    report = response.content

    trace_entry["status"] = "completed"
    trace_entry["report_length"] = len(report)

    logger.info("DecisionExplanationAgent: generated %d-char report", len(report))

    return {
        "explanation_report": report,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }

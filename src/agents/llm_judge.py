"""LLM Judge Agent – independent review of the automated underwriting outcome.

Acts as a second-line check: consistency between model score, policy rules,
and narrative explanation; fairness / regulatory red flags; whether a human
review is advisable. Does not override the system decision by default.
"""

import logging
import os
from typing import Any, Dict, List
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.agents.llm_provider import llm_credentials_ok, make_chat_llm
from src.security.llm_guardrails import combine_system_message

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    """Structured output from the judge LLM."""

    verdict: str = Field(
        description=(
            "CONCUR: pipeline outcome is coherent and defensible; "
            "FLAG_FOR_REVIEW: recommend human underwriter review despite outcome; "
            "CHALLENGE: material inconsistency or serious concern (explain in rationale)"
        )
    )
    rationale: str = Field(description="2–5 sentences explaining the judgment")
    concerns: List[str] = Field(
        default_factory=list,
        description="Specific issues, if any (empty if none)",
    )
    compliance_notes: str = Field(
        default="",
        description="Brief note on ECOA/fair-lending or adverse-action alignment, if relevant",
    )


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            combine_system_message(
                """You are an independent senior credit risk auditor conducting a second-line review.
You receive the output of an automated underwriting pipeline and assess its integrity.

Your review covers:
1. **Decision coherence** — does the decision align with the default probability, risk tier, and policy flags?
2. **Model signal quality** — do the top SHAP factors logically support the score?
3. **Policy adherence** — are hard stops and policy violations correctly reflected?
4. **Risk flags** — identify any concentrations, anomalies, or edge cases the model may have missed.

You do NOT re-score the applicant. You assess the quality and defensibility of the pipeline output.

Return structured fields:
- verdict: CONCUR | FLAG_FOR_REVIEW | CHALLENGE
- rationale: 2–4 sentences of internal analyst commentary
- concerns: specific issues flagged (empty list if none)
- compliance_notes: any fair-lending or policy alignment notes (empty if none)"""
            ),
        ),
        (
            "human",
            """## Second-Line Review — Automated Underwriting Output

**Decision**: {decision} | **Risk Tier**: {risk_tier} | **Default Probability**: {risk_score}
**Confidence**: {confidence} | **Binary Prediction**: {prediction} | **Policy Passed**: {policy_passed}

### Policy Violations
{policy_violations}

### Top SHAP Factors
{shap_summary}

### Analyst Risk Memo (excerpt)
{explanation_excerpt}

Provide your internal audit judgment.""",
        ),
    ]
)


def _shap_summary(state: Dict[str, Any]) -> str:
    exp = state.get("shap_explanation") or {}
    factors = exp.get("top_risk_factors") or []
    if not factors:
        return "None listed."
    lines = []
    for f in factors[:8]:
        lines.append(
            f"- {f.get('feature')}: impact={f.get('shap_impact')}, {f.get('direction', '')}"
        )
    return "\n".join(lines)


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the judge LLM and attach verdict + trace."""
    trace_entry = {
        "agent": "LLMJudgeAgent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    if not llm_credentials_ok():
        trace_entry["status"] = "skipped"
        trace_entry["reason"] = "no LLM API key for configured provider"
        logger.warning("LLMJudgeAgent: skipped — no API key")
        return {
            "llm_judge_verdict": "SKIPPED",
            "llm_judge_rationale": (
                "Judge requires GROQ_API_KEY or LLM_PROVIDER=ollama with Ollama running."
            ),
            "llm_judge_concerns": [],
            "llm_judge_compliance_notes": "",
            "agent_trace": state.get("agent_trace", []) + [trace_entry],
        }

    report = state.get("explanation_report") or ""
    excerpt = report[:6000] if len(report) > 6000 else report
    if len(report) > 6000:
        excerpt += "\n\n[... truncated for judge context ...]"

    try:
        llm = make_chat_llm(temperature=0.1, max_tokens=1200)
    except RuntimeError as e:
        trace_entry["status"] = "skipped"
        trace_entry["reason"] = str(e)
        return {
            "llm_judge_verdict": "SKIPPED",
            "llm_judge_rationale": str(e),
            "llm_judge_concerns": [],
            "llm_judge_compliance_notes": "",
            "agent_trace": state.get("agent_trace", []) + [trace_entry],
        }

    structured = llm.with_structured_output(JudgeVerdict)

    chain = JUDGE_PROMPT | structured

    try:
        out: JudgeVerdict = chain.invoke(
            {
                "decision": state.get("decision", "N/A"),
                "risk_score": state.get("risk_score", "N/A"),
                "risk_tier": state.get("risk_tier", "N/A"),
                "prediction": state.get("prediction", "N/A"),
                "confidence": state.get("confidence", "N/A"),
                "policy_passed": state.get("policy_passed", "N/A"),
                "policy_violations": "\n".join(state.get("policy_violations") or [])
                or "None",
                "shap_summary": _shap_summary(state),
                "explanation_excerpt": excerpt or "(empty report)",
            }
        )
    except Exception as e:
        logger.exception("LLMJudgeAgent failed: %s", e)
        trace_entry["status"] = "error"
        trace_entry["error"] = str(e)
        return {
            "llm_judge_verdict": "ERROR",
            "llm_judge_rationale": f"Judge invocation failed: {e}",
            "llm_judge_concerns": [],
            "llm_judge_compliance_notes": "",
            "agent_trace": state.get("agent_trace", []) + [trace_entry],
        }

    trace_entry["status"] = "completed"
    trace_entry["verdict"] = out.verdict

    logger.info("LLMJudgeAgent: verdict=%s", out.verdict)

    return {
        "llm_judge_verdict": out.verdict,
        "llm_judge_rationale": out.rationale,
        "llm_judge_concerns": out.concerns,
        "llm_judge_compliance_notes": out.compliance_notes,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }

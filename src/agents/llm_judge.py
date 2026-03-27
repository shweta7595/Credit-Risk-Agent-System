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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
                """You are an independent senior credit risk auditor and LLM judge.
Your job is to review outputs from an automated underwriting pipeline (model + rules + explanation).

Evaluate:
1. **Consistency**: Does the stated decision (APPROVE / DECLINE / MANUAL_REVIEW) align with the default probability, risk tier, and policy violations?
2. **Policy**: Are hard stops and warnings reflected appropriately in the narrative?
3. **Fairness / proxies**: Any sign the explanation inappropriately relies on protected-class proxies or vague language?
4. **Regulatory tone**: Is the explanation suitable for adverse-action or manual-review context (ECOA / Reg B spirit)?

You do NOT re-score the applicant. You only judge quality and coherence of the automated output.

Return structured fields exactly as specified:
- verdict: exactly one of CONCUR, FLAG_FOR_REVIEW, CHALLENGE
- rationale: concise professional assessment
- concerns: list of strings (empty if none)
- compliance_notes: short string (empty if nothing to add)"""
            ),
        ),
        (
            "human",
            """## Pipeline outputs to review

### Automated decision
- **Decision**: {decision}
- **Default probability (risk score)**: {risk_score}
- **Risk tier**: {risk_tier}
- **Binary prediction (1=default)**: {prediction}
- **Model confidence**: {confidence}
- **Policy passed**: {policy_passed}

### Policy violations (if any)
{policy_violations}

### Top model factors (SHAP)
{shap_summary}

### Generated explanation report (may be truncated)
---
{explanation_excerpt}
---

Provide your structured judgment.""",
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

    if not os.getenv("OPENAI_API_KEY"):
        trace_entry["status"] = "skipped"
        trace_entry["reason"] = "OPENAI_API_KEY not set"
        logger.warning("LLMJudgeAgent: skipped — no API key")
        return {
            "llm_judge_verdict": "SKIPPED",
            "llm_judge_rationale": "Judge requires OPENAI_API_KEY.",
            "llm_judge_concerns": [],
            "llm_judge_compliance_notes": "",
            "agent_trace": state.get("agent_trace", []) + [trace_entry],
        }

    report = state.get("explanation_report") or ""
    excerpt = report[:6000] if len(report) > 6000 else report
    if len(report) > 6000:
        excerpt += "\n\n[... truncated for judge context ...]"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1200)
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

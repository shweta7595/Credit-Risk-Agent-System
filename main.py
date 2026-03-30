"""Main entry point for the Credit Risk Underwriting Agent System.

Run: python main.py
Requires: trained model (run `python -m src.train` first) and LLM config in .env
  (OPENAI / GOOGLE / GROQ keys by LLM_PROVIDER, or ollama for local Llama)
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LangSmith tracing — activated automatically when these env vars are set
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "credit-risk-agent-system")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

from src.agents.graph import run_pipeline
from src.agents.llm_provider import llm_credentials_ok, resolve_llm_provider


SAMPLE_APPLICANTS = [
    {
        "name": "Low Risk – Strong Profile",
        "data": {
            "person_age": 35,
            "person_income": 95000,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 10.0,
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_grade": "A",
            "loan_amnt": 8000,
            "loan_int_rate": 6.5,
            "loan_percent_income": 0.08,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 12,
        },
    },
    {
        "name": "High Risk – Prior Default",
        "data": {
            "person_age": 23,
            "person_income": 28000,
            "person_home_ownership": "RENT",
            "person_emp_length": 1.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "D",
            "loan_amnt": 15000,
            "loan_int_rate": 16.5,
            "loan_percent_income": 0.54,
            "cb_person_default_on_file": "Y",
            "cb_person_cred_hist_length": 2,
        },
    },
    {
        "name": "Medium Risk – Moderate Profile",
        "data": {
            "person_age": 29,
            "person_income": 55000,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "EDUCATION",
            "loan_grade": "C",
            "loan_amnt": 12000,
            "loan_int_rate": 12.0,
            "loan_percent_income": 0.22,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5,
        },
    },
]


def run_single_applicant(applicant: dict) -> dict:
    """Process a single applicant through the full agent pipeline."""
    print(f"\n{'='*70}")
    print(f"  PROCESSING: {applicant['name']}")
    print(f"{'='*70}")

    result = run_pipeline(applicant["data"])

    print(f"\n  Decision:     {result.get('decision', 'N/A')}")
    print(f"  Risk Score:   {result.get('risk_score', 'N/A')}")
    print(f"  Risk Tier:    {result.get('risk_tier', 'N/A')}")
    print(f"  Confidence:   {result.get('confidence', 'N/A')}")
    print(f"  Policy Pass:  {result.get('policy_passed', 'N/A')}")

    if result.get("policy_violations"):
        print(f"\n  Policy Violations:")
        for v in result["policy_violations"]:
            print(f"    - {v}")

    if result.get("explanation_report"):
        print(f"\n  --- Explanation Report ---")
        print(result["explanation_report"])

    if result.get("llm_judge_verdict"):
        print(f"\n  --- LLM Judge ---")
        print(f"  Verdict:      {result.get('llm_judge_verdict')}")
        print(f"  Rationale:    {result.get('llm_judge_rationale', '')}")
        if result.get("llm_judge_concerns"):
            print(f"  Concerns:")
            for c in result["llm_judge_concerns"]:
                print(f"    - {c}")
        if result.get("llm_judge_compliance_notes"):
            print(f"  Compliance:   {result['llm_judge_compliance_notes']}")

    print(f"\n  Agent Trace ({len(result.get('agent_trace', []))} steps):")
    for step in result.get("agent_trace", []):
        print(f"    [{step.get('agent')}] {step.get('status')} @ {step.get('timestamp', '')[:19]}")

    return result


def main():
    model_path = Path("models/risk_model.joblib")
    if not model_path.exists():
        print("ERROR: No trained model found. Run training first:")
        print("  python -m src.train")
        return

    if not llm_credentials_ok():
        print("WARNING: No LLM API key for the configured provider.")
        print(
            f"  LLM_PROVIDER={resolve_llm_provider()} — set the matching key in .env "
            "(OPENAI / GOOGLE / GROQ) or use ollama with the server running."
        )

    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        print("LangSmith tracing ENABLED → project: credit-risk-agent-system")
    else:
        print("LangSmith tracing DISABLED (set LANGCHAIN_API_KEY in .env to enable)")

    print("\n" + "#" * 70)
    print("#  CREDIT RISK UNDERWRITING AGENT SYSTEM")
    print("#  Powered by LangGraph + Gradient Boosting + SHAP + LangSmith")
    print("#" * 70)

    results = []
    for applicant in SAMPLE_APPLICANTS:
        result = run_single_applicant(applicant)
        results.append({
            "applicant": applicant["name"],
            "decision": result.get("decision"),
            "risk_score": result.get("risk_score"),
            "risk_tier": result.get("risk_tier"),
            "confidence": result.get("confidence"),
            "policy_passed": result.get("policy_passed"),
            "violations_count": len(result.get("policy_violations", [])),
            "judge_verdict": result.get("llm_judge_verdict"),
        })

    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        jv = r.get("judge_verdict") or "—"
        print(
            f"  {r['applicant']:36s} → {r['decision']:15s} "
            f"(score={r['risk_score']}, judge={jv})"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()

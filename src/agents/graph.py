"""LangGraph orchestration of the Credit Risk Underwriting pipeline.

Six agents: data → features → risk model → policy → explanation → LLM judge.
"""

import logging
import operator
from typing import Any, Dict, List, Annotated, TypedDict, Optional

from langgraph.graph import StateGraph, END

from src.agents import (
    data_aggregation,
    feature_engineering_agent,
    risk_modeling,
    policy_validation,
    decision_explanation,
    llm_judge,
)

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State flowing through the LangGraph pipeline."""

    applicant: Dict[str, Any]
    raw_data: Dict[str, Any]
    engineered_features: Dict[str, Any]
    feature_vector: Dict[str, float]
    risk_score: Optional[float]
    risk_tier: Optional[str]
    prediction: Optional[int]
    confidence: Optional[float]
    shap_explanation: Dict[str, Any]
    policy_violations: List[str]
    policy_passed: Optional[bool]
    decision: Optional[str]
    explanation_report: str
    llm_judge_verdict: Optional[str]
    llm_judge_rationale: str
    llm_judge_concerns: List[str]
    llm_judge_compliance_notes: str
    agent_trace: Annotated[List[Dict[str, Any]], operator.add]


def _data_aggregation_node(state: GraphState) -> Dict[str, Any]:
    return data_aggregation.run(state)


def _feature_engineering_node(state: GraphState) -> Dict[str, Any]:
    return feature_engineering_agent.run(state)


def _risk_modeling_node(state: GraphState) -> Dict[str, Any]:
    return risk_modeling.run(state)


def _policy_validation_node(state: GraphState) -> Dict[str, Any]:
    return policy_validation.run(state)


def _decision_explanation_node(state: GraphState) -> Dict[str, Any]:
    return decision_explanation.run(state)


def _llm_judge_node(state: GraphState) -> Dict[str, Any]:
    return llm_judge.run(state)


def _should_explain(state: GraphState) -> str:
    """Route: always go to explanation for full transparency."""
    return "explain"


def build_graph() -> StateGraph:
    """Construct and compile the credit risk agent graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("data_aggregation", _data_aggregation_node)
    workflow.add_node("feature_engineering", _feature_engineering_node)
    workflow.add_node("risk_modeling", _risk_modeling_node)
    workflow.add_node("policy_validation", _policy_validation_node)
    workflow.add_node("decision_explanation", _decision_explanation_node)
    workflow.add_node("llm_judge", _llm_judge_node)

    workflow.set_entry_point("data_aggregation")
    workflow.add_edge("data_aggregation", "feature_engineering")
    workflow.add_edge("feature_engineering", "risk_modeling")
    workflow.add_edge("risk_modeling", "policy_validation")
    workflow.add_conditional_edges(
        "policy_validation",
        _should_explain,
        {"explain": "decision_explanation"},
    )
    workflow.add_edge("decision_explanation", "llm_judge")
    workflow.add_edge("llm_judge", END)

    return workflow.compile()


def run_pipeline(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full credit risk pipeline for one applicant."""
    graph = build_graph()

    initial_state: GraphState = {
        "applicant": applicant_data,
        "raw_data": {},
        "engineered_features": {},
        "feature_vector": {},
        "risk_score": None,
        "risk_tier": None,
        "prediction": None,
        "confidence": None,
        "shap_explanation": {},
        "policy_violations": [],
        "policy_passed": None,
        "decision": None,
        "explanation_report": "",
        "llm_judge_verdict": None,
        "llm_judge_rationale": "",
        "llm_judge_concerns": [],
        "llm_judge_compliance_notes": "",
        "agent_trace": [],
    }

    result = graph.invoke(initial_state)
    return result

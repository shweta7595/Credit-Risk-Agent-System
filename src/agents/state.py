"""Shared state definition for the Credit Risk Agent graph."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    """Raw applicant data submitted for underwriting."""

    person_age: int
    person_income: int
    person_home_ownership: str  # RENT, OWN, MORTGAGE, OTHER
    person_emp_length: float
    loan_intent: str  # PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION
    loan_grade: str  # A–G
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str  # Y or N
    cb_person_cred_hist_length: int


class CreditRiskState(BaseModel):
    """Full pipeline state that flows through all agents in the graph."""

    applicant: ApplicantInput
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    engineered_features: Dict[str, Any] = Field(default_factory=dict)
    feature_vector: Dict[str, float] = Field(default_factory=dict)
    risk_score: Optional[float] = None
    risk_tier: Optional[str] = None
    prediction: Optional[int] = None
    confidence: Optional[float] = None
    shap_explanation: Dict[str, Any] = Field(default_factory=dict)
    policy_violations: List[str] = Field(default_factory=list)
    policy_passed: Optional[bool] = None
    decision: Optional[str] = None  # APPROVE, DECLINE, MANUAL_REVIEW
    explanation_report: str = ""
    llm_judge_verdict: Optional[str] = None  # CONCUR, FLAG_FOR_REVIEW, CHALLENGE, SKIPPED
    llm_judge_rationale: str = ""
    llm_judge_concerns: List[str] = Field(default_factory=list)
    llm_judge_compliance_notes: str = ""
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)

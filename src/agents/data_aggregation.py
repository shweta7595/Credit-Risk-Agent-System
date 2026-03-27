"""Data Aggregation Agent – validates and normalizes raw applicant data."""

import logging
from typing import Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate raw applicant inputs, flag anomalies, and normalize values."""
    applicant = state["applicant"]
    trace_entry = {
        "agent": "DataAggregationAgent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    raw_data = {
        "person_age": applicant["person_age"],
        "person_income": applicant["person_income"],
        "person_home_ownership": applicant["person_home_ownership"].upper(),
        "person_emp_length": applicant["person_emp_length"],
        "loan_intent": applicant["loan_intent"].upper(),
        "loan_grade": applicant["loan_grade"].upper(),
        "loan_amnt": applicant["loan_amnt"],
        "loan_int_rate": applicant["loan_int_rate"],
        "loan_percent_income": applicant["loan_percent_income"],
        "cb_person_default_on_file": 1 if applicant["cb_person_default_on_file"].upper() == "Y" else 0,
        "cb_person_cred_hist_length": applicant["cb_person_cred_hist_length"],
    }

    anomalies = []
    if raw_data["person_age"] < 18 or raw_data["person_age"] > 100:
        anomalies.append(f"Unusual age: {raw_data['person_age']}")
    if raw_data["person_income"] < 0:
        anomalies.append(f"Negative income: {raw_data['person_income']}")
    if raw_data["person_emp_length"] < 0 or raw_data["person_emp_length"] > 60:
        anomalies.append(f"Unusual employment length: {raw_data['person_emp_length']}")
    if raw_data["loan_amnt"] < 500 or raw_data["loan_amnt"] > 35000:
        anomalies.append(f"Loan amount outside range: {raw_data['loan_amnt']}")

    raw_data["anomalies"] = anomalies

    trace_entry["status"] = "completed"
    trace_entry["anomalies_found"] = len(anomalies)

    logger.info("DataAggregationAgent: validated applicant data, %d anomalies", len(anomalies))

    return {
        "raw_data": raw_data,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }

"""LangSmith evaluators for the credit risk pipeline.

Per-example evaluators  → called once per run/example pair
Summary evaluators      → called once per experiment with ALL runs (for aggregate metrics)

All metrics are deterministic (no LLM calls) — free on any LangSmith tier.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from langsmith.evaluation import EvaluationResult, EvaluationResults


# ── helpers ──────────────────────────────────────────────────────────────────

def _actual(run) -> Dict[str, Any]:
    return run.outputs or {}


def _expected(example) -> Dict[str, Any]:
    return example.outputs or {}


# ── per-example evaluators ────────────────────────────────────────────────────

def decision_match(run, example) -> Dict[str, Any]:
    """Exact match on APPROVE / DECLINE / MANUAL_REVIEW."""
    actual = _actual(run).get("decision")
    exp = _expected(example).get("expected_decision")
    if exp is None:
        return {"key": "decision_match", "score": None}
    return {
        "key": "decision_match",
        "score": 1 if actual == exp else 0,
        "comment": f"got={actual} expected={exp}",
    }


def risk_tier_match(run, example) -> Dict[str, Any]:
    """Exact match on LOW / MEDIUM / HIGH."""
    actual = _actual(run).get("risk_tier")
    exp = _expected(example).get("expected_risk_tier")
    if exp is None:
        return {"key": "risk_tier_match", "score": None}
    return {
        "key": "risk_tier_match",
        "score": 1 if actual == exp else 0,
        "comment": f"got={actual} expected={exp}",
    }


def policy_correctness(run, example) -> Dict[str, Any]:
    """policy_passed matches expected boolean."""
    actual = _actual(run).get("policy_passed")
    exp = _expected(example).get("expected_policy_passed")
    if exp is None:
        return {"key": "policy_correctness", "score": None}
    return {
        "key": "policy_correctness",
        "score": 1 if actual == exp else 0,
        "comment": f"got={actual} expected={exp}",
    }


def risk_score_in_range(run, example) -> Dict[str, Any]:
    """risk_score falls within [expected_score_min, expected_score_max]."""
    score_val = _actual(run).get("risk_score")
    exp = _expected(example)
    lo = exp.get("expected_score_min", 0.0)
    hi = exp.get("expected_score_max", 1.0)
    if score_val is None:
        return {"key": "risk_score_in_range", "score": 0, "comment": "risk_score missing"}
    in_range = lo <= float(score_val) <= hi
    return {
        "key": "risk_score_in_range",
        "score": 1 if in_range else 0,
        "comment": f"score={score_val:.4f} range=[{lo}, {hi}]",
    }


def judge_ran(run, example) -> Dict[str, Any]:
    """LLM judge produced a real verdict (not SKIPPED / ERROR)."""
    verdict = _actual(run).get("llm_judge_verdict", "")
    passed = verdict not in ("SKIPPED", "ERROR", None, "")
    return {
        "key": "judge_ran",
        "score": 1 if passed else 0,
        "comment": f"verdict={verdict}",
    }


def judge_coherence(run, example) -> Dict[str, Any]:
    """For clear APPROVE/DECLINE cases, judge should CONCUR (not CHALLENGE)."""
    exp_decision = _expected(example).get("expected_decision")
    verdict = _actual(run).get("llm_judge_verdict", "")
    if exp_decision not in ("APPROVE", "DECLINE"):
        return {"key": "judge_coherence", "score": None, "comment": "not a clear-cut case"}
    return {
        "key": "judge_coherence",
        "score": 1 if verdict == "CONCUR" else 0,
        "comment": f"verdict={verdict} for expected={exp_decision}",
    }


# ── summary evaluators (aggregate, across all examples) ──────────────────────

def decision_accuracy(runs, examples) -> Dict[str, Any]:
    """Overall decision accuracy across the full eval set."""
    correct = total = 0
    for run, ex in zip(runs, examples):
        exp = (ex.outputs or {}).get("expected_decision")
        actual = (run.outputs or {}).get("decision")
        if exp is not None:
            total += 1
            if actual == exp:
                correct += 1
    return {
        "key": "decision_accuracy",
        "score": correct / total if total else 0.0,
        "comment": f"{correct}/{total} correct",
    }


def decision_precision_recall_f1(runs, examples) -> EvaluationResults:
    """Per-class precision, recall, F1 for decision labels."""
    classes = ["APPROVE", "DECLINE", "MANUAL_REVIEW"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for run, ex in zip(runs, examples):
        exp = (ex.outputs or {}).get("expected_decision")
        actual = (run.outputs or {}).get("decision")
        if exp is None or actual is None:
            continue
        for cls in classes:
            if actual == cls and exp == cls:
                tp[cls] += 1
            elif actual == cls and exp != cls:
                fp[cls] += 1
            elif actual != cls and exp == cls:
                fn[cls] += 1

    results = []
    macro_p = macro_r = macro_f1 = 0.0
    n_classes = 0

    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        label = cls.lower()
        results += [
            EvaluationResult(key=f"precision_{label}", score=round(p, 4)),
            EvaluationResult(key=f"recall_{label}",    score=round(r, 4)),
            EvaluationResult(key=f"f1_{label}",        score=round(f1, 4)),
        ]
        macro_p += p
        macro_r += r
        macro_f1 += f1
        n_classes += 1

    if n_classes:
        results += [
            EvaluationResult(key="macro_precision", score=round(macro_p / n_classes, 4)),
            EvaluationResult(key="macro_recall",    score=round(macro_r / n_classes, 4)),
            EvaluationResult(key="macro_f1",        score=round(macro_f1 / n_classes, 4)),
        ]

    return EvaluationResults(results=results)


def brier_score(runs, examples) -> Dict[str, Any]:
    """Brier score on default probability vs binary expected outcome.

    Maps expected_decision → binary label: DECLINE=1, APPROVE=0, MANUAL_REVIEW=0.5
    Lower score = better calibrated probabilities (0 = perfect, 1 = worst).
    """
    label_map = {"DECLINE": 1.0, "APPROVE": 0.0, "MANUAL_REVIEW": 0.5}
    total = sq_err = 0
    for run, ex in zip(runs, examples):
        exp = (ex.outputs or {}).get("expected_decision")
        score_val = (run.outputs or {}).get("risk_score")
        if exp is None or score_val is None:
            continue
        y = label_map.get(exp, 0.5)
        sq_err += (float(score_val) - y) ** 2
        total += 1
    bs = sq_err / total if total else None
    return {
        "key": "brier_score",
        "score": round(bs, 4) if bs is not None else None,
        "comment": f"avg squared error over {total} examples (lower=better)",
    }


def policy_recall(runs, examples) -> Dict[str, Any]:
    """Recall for detecting policy failures (policy_passed=False).

    A false negative here means a risky applicant was not flagged — most critical.
    """
    tp = fn = 0
    for run, ex in zip(runs, examples):
        exp = (ex.outputs or {}).get("expected_policy_passed")
        actual = (run.outputs or {}).get("policy_passed")
        if exp is False:           # ground truth: should have flagged
            if actual is False:
                tp += 1            # correctly caught
            else:
                fn += 1            # missed — dangerous
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return {
        "key": "policy_violation_recall",
        "score": round(recall, 4) if recall is not None else None,
        "comment": f"caught {tp}/{tp+fn} true policy failures",
    }


# ── exports ───────────────────────────────────────────────────────────────────

ALL_EVALUATORS = [
    decision_match,
    risk_tier_match,
    policy_correctness,
    risk_score_in_range,
    judge_ran,
    judge_coherence,
]

ALL_SUMMARY_EVALUATORS = [
    decision_accuracy,
    decision_precision_recall_f1,
    brier_score,
    policy_recall,
]

"""ML model monitoring report — pulls recent LangSmith runs and computes:

  - Score distribution (mean, std, percentiles)
  - Decision & risk-tier distribution
  - Model confidence distribution
  - PSI (Population Stability Index) vs reference baseline
  - Policy violation rate
  - Judge verdict distribution

Usage:
    python -m eval.monitor            # last 50 runs
    python -m eval.monitor --n 200
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from langsmith import Client

# ── reference baseline (from training set statistics) ────────────────────────
# Bucket boundaries for risk_score PSI: 10 equal-width bins [0, 0.1), ..., [0.9, 1.0]
# Expected proportions per bucket from training data distribution
REFERENCE_SCORE_BUCKETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
REFERENCE_SCORE_DIST = [0.18, 0.22, 0.14, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04]


def _bucket(score: float, edges: List[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= score < edges[i + 1]:
            return i
    return len(edges) - 2


def psi(actual_dist: List[float], expected_dist: List[float]) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.2 minor drift, >0.2 major drift."""
    total = 0.0
    for a, e in zip(actual_dist, expected_dist):
        a = max(a, 1e-6)
        e = max(e, 1e-6)
        total += (a - e) * math.log(a / e)
    return round(total, 4)


def _stats(values: List[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "n":    n,
        "mean": round(mean, 4),
        "std":  round(variance ** 0.5, 4),
        "p10":  round(s[int(n * 0.10)], 4),
        "p25":  round(s[int(n * 0.25)], 4),
        "p50":  round(s[n // 2], 4),
        "p75":  round(s[int(n * 0.75)], 4),
        "p90":  round(s[int(n * 0.90)], 4),
        "p95":  round(s[int(n * 0.95)], 4),
    }


def _pct(counter: Counter, total: int) -> dict:
    return {k: f"{v} ({100*v/total:.1f}%)" for k, v in counter.most_common()}


def run_monitoring_report(n: int = 50) -> None:
    client = Client()

    print(f"\nFetching last {n} 'credit-risk-pipeline' runs from LangSmith...")
    # Fetch top-level pipeline runs only (no parent_run_id = root runs)
    all_runs = list(client.list_runs(
        project_name="credit-risk-agent-system",
        run_type="chain",
        limit=n * 5,  # fetch extra to account for non-pipeline runs
    ))
    # Keep root-level runs that carry full pipeline outputs
    runs = [
        r for r in all_runs
        if r.parent_run_id is None
        and (r.outputs or {}).get("risk_score") is not None
    ][:n]

    if not runs:
        print("No runs found. Run the pipeline at least once via the UI or main.py.")
        return

    print(f"Found {len(runs)} runs.\n")

    risk_scores:    List[float] = []
    confidences:    List[float] = []
    decisions:      List[str]   = []
    risk_tiers:     List[str]   = []
    judge_verdicts: List[str]   = []
    violation_counts: List[int] = []

    for run in runs:
        out = run.outputs or {}
        if out.get("risk_score") is not None:
            risk_scores.append(float(out["risk_score"]))
        if out.get("confidence") is not None:
            confidences.append(float(out["confidence"]))
        if out.get("decision"):
            decisions.append(out["decision"])
        if out.get("risk_tier"):
            risk_tiers.append(out["risk_tier"])
        if out.get("llm_judge_verdict"):
            judge_verdicts.append(out["llm_judge_verdict"])
        violations = out.get("policy_violations") or []
        violation_counts.append(len(violations))

    total = len(runs)

    # ── risk score distribution ───────────────────────────────────────────────
    print("=" * 60)
    print("RISK SCORE DISTRIBUTION")
    print("=" * 60)
    for k, v in _stats(risk_scores).items():
        print(f"  {k:<6}: {v}")

    # ── PSI vs training baseline ──────────────────────────────────────────────
    if risk_scores:
        bucket_counts = Counter(_bucket(s, REFERENCE_SCORE_BUCKETS) for s in risk_scores)
        actual_dist = [bucket_counts.get(i, 0) / len(risk_scores)
                       for i in range(len(REFERENCE_SCORE_DIST))]
        score_psi = psi(actual_dist, REFERENCE_SCORE_DIST)
        drift = "STABLE" if score_psi < 0.1 else ("MINOR DRIFT" if score_psi < 0.2 else "MAJOR DRIFT ⚠")
        print(f"\n  PSI vs training baseline: {score_psi} — {drift}")

    # ── decision distribution ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DECISION DISTRIBUTION")
    print("=" * 60)
    for k, v in _pct(Counter(decisions), total).items():
        print(f"  {k:<15}: {v}")

    # ── risk tier distribution ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RISK TIER DISTRIBUTION")
    print("=" * 60)
    for k, v in _pct(Counter(risk_tiers), total).items():
        print(f"  {k:<10}: {v}")

    # ── model confidence ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL CONFIDENCE DISTRIBUTION")
    print("=" * 60)
    for k, v in _stats(confidences).items():
        print(f"  {k:<6}: {v}")

    # ── policy violations ─────────────────────────────────────────────────────
    flagged = sum(1 for v in violation_counts if v > 0)
    print("\n" + "=" * 60)
    print("POLICY VIOLATIONS")
    print("=" * 60)
    print(f"  Runs with violations: {flagged}/{total} ({100*flagged/total:.1f}%)")
    if violation_counts:
        print(f"  Avg violations/run:   {sum(violation_counts)/len(violation_counts):.2f}")

    # ── judge verdict distribution ────────────────────────────────────────────
    if judge_verdicts:
        print("\n" + "=" * 60)
        print("LLM JUDGE VERDICTS")
        print("=" * 60)
        for k, v in _pct(Counter(judge_verdicts), len(judge_verdicts)).items():
            print(f"  {k:<20}: {v}")

    print("\n" + "=" * 60)
    print(f"Report complete. View traces at: https://smith.langchain.com")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of recent runs to analyse")
    args = parser.parse_args()
    run_monitoring_report(args.n)

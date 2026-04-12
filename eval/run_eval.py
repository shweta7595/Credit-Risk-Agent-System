"""Run LangSmith evaluation against the credit risk pipeline.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --experiment my-run-name
    python -m eval.run_eval --dataset credit-risk-golden-set
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from langsmith import Client
from langsmith.evaluation import evaluate

from src.agents.graph import run_pipeline
from eval.dataset import create_or_get_dataset, CHALLENGING_EXAMPLES
from eval.evaluators import ALL_EVALUATORS, ALL_SUMMARY_EVALUATORS


def pipeline_target(inputs: dict) -> dict:
    """Adapter: LangSmith passes example inputs dict; extract applicant and run."""
    return run_pipeline(inputs["applicant"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangSmith evaluation")
    parser.add_argument("--experiment", default="baseline", help="Experiment prefix")
    parser.add_argument("--dataset", default="credit-risk-eval-v2", help="Dataset name")
    args = parser.parse_args()

    client = Client()

    print(f"Pushing dataset '{args.dataset}' to LangSmith...")
    dataset = create_or_get_dataset(client, args.dataset, CHALLENGING_EXAMPLES)
    print(f"Dataset ready: {dataset.name} (id={dataset.id})")

    print(f"\nRunning evaluation — experiment prefix: '{args.experiment}'")
    results = evaluate(
        pipeline_target,
        data=args.dataset,
        evaluators=ALL_EVALUATORS,
        summary_evaluators=ALL_SUMMARY_EVALUATORS,
        experiment_prefix=args.experiment,
        max_concurrency=1,
        metadata={"pipeline_version": "1.0"},
    )

    print("\n=== Evaluation Results ===")
    for r in results:
        print(r)

    print(f"\nView results at: https://smith.langchain.com")


if __name__ == "__main__":
    main()

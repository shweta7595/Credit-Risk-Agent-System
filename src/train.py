"""Train the risk model and persist artifacts.

Production: Gradient Boosting + PR threshold + SHAP.
Benchmark: Random Forest (same split, reported only — not deployed).

Run: python -m src.train
"""

import logging

from src.data_ingestion import ingest
from src.feature_engineering import create_features, prepare_model_data, get_feature_summary
from src.risk_model import (
    train_model,
    train_benchmark_random_forest,
    save_model,
    evaluate_model,
    find_optimal_threshold,
    plot_precision_recall_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Step 1: Data Ingestion (with outlier capping) ===")
    df, clean_path = ingest()

    logger.info("=== Step 2: Feature Engineering (log transforms + buckets) ===")
    df = create_features(df)
    summary = get_feature_summary(df)
    logger.info("Feature summary: %s", summary)

    logger.info("=== Step 3: Prepare Train/Test Split ===")
    X_train, X_test, y_train, y_test, feature_names, scaler, scaled_columns = (
        prepare_model_data(df)
    )

    logger.info("=== Step 4: Train production Gradient Boosting + calibration ===")
    model, calibrated, metrics_default = train_model(X_train, y_train, X_test, y_test)

    logger.info("=== Step 4b: Train Random Forest (benchmark only) ===")
    rf_metrics = train_benchmark_random_forest(X_train, y_train, X_test, y_test)

    logger.info("=== Step 5: Precision-Recall Threshold (calibrated model) ===")
    threshold_analysis = find_optimal_threshold(calibrated, X_test, y_test, min_recall=0.85)
    optimal_threshold = threshold_analysis["optimal_threshold"]
    metrics_optimal = evaluate_model(calibrated, X_test, y_test, threshold=optimal_threshold)

    logger.info("=== Step 6: Generate PR / ROC Curves (calibrated model) ===")
    curve_path = plot_precision_recall_curve(calibrated, X_test, y_test, optimal_threshold)

    logger.info("=== Step 7: Save production artifacts ===")
    save_model(model, scaler, feature_names, scaled_columns,
               threshold=optimal_threshold, calibrated_model=calibrated,
               metrics=metrics_optimal)

    print("\n" + "=" * 70)
    print("  MODEL TRAINING COMPLETE")
    print("  Production: Gradient Boosting  |  Benchmark: Random Forest (not deployed)")
    print("=" * 70)
    print(f"  Features: {len(feature_names)}")
    print(f"  GB: 300 trees, max_depth=5, balanced sample_weight")
    if summary.get("log_transformed"):
        print(f"  Log-transformed: {summary['log_transformed']}")
    if summary.get("information_values"):
        print(f"  IV Scores: {summary['information_values']}")

    print(f"\n  {'MODEL':<28} {'AUC':>8} {'PREC':>8} {'RECALL':>8} {'F1':>8}  (@ t=0.50)")
    print(f"  {'-'*70}")
    print(
        f"  {'Gradient Boosting (production)':<28} "
        f"{metrics_default['auc']:>8.4f} {metrics_default['precision']:>8.4f} "
        f"{metrics_default['recall']:>8.4f} {metrics_default['f1']:>8.4f}"
    )
    print(
        f"  {'Random Forest (benchmark)':<28} "
        f"{rf_metrics['auc']:>8.4f} {rf_metrics['precision']:>8.4f} "
        f"{rf_metrics['recall']:>8.4f} {rf_metrics['f1']:>8.4f}"
    )

    print(f"\n  Production GB — threshold tuning (PR curve, min_recall=0.85)")
    print(f"  {'METRIC':<20} {'@ t=0.50':>12} {'@ t=' + str(round(optimal_threshold, 2)):>12} {'CHANGE':>10}")
    print(f"  {'-'*54}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        v1, v2 = metrics_default[key], metrics_optimal[key]
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        print(f"  {key.upper():<20} {v1:>12.4f} {v2:>12.4f} {sign}{delta:.4f}{'':>4}")
    print(f"  {'AUC-ROC':<20} {metrics_default['auc']:>12.4f} {metrics_optimal['auc']:>12.4f} {'(same)':>10}")

    print(f"\n  Optimal threshold (saved): {optimal_threshold}")
    print(f"  Strategy: {threshold_analysis['strategy']}")

    cm_def = metrics_default["confusion_matrix"]
    cm_opt = metrics_optimal["confusion_matrix"]
    print(f"\n  GB Confusion Matrix @ t=0.50: {cm_def}")
    print(f"  GB Confusion Matrix @ t={optimal_threshold:.2f}: {cm_opt}")
    fn_before, fn_after = cm_def[1][0], cm_opt[1][0]
    print(f"  Missed defaults: {fn_before} → {fn_after} ({fn_before - fn_after} more caught)")

    print(f"\n  {'THRESHOLD':>10} {'PRECISION':>10} {'RECALL':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for row in threshold_analysis["threshold_sweep"]:
        marker = " ◀" if abs(row["threshold"] - round(optimal_threshold, 2)) < 0.01 else ""
        print(f"  {row['threshold']:>10.2f} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}{marker}")

    print(f"\n  PR / ROC curves (GB): {curve_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

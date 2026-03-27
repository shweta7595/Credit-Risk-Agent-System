# Risk Scoring Thresholds

## Risk Tiers

| Tier     | Default Probability | Action          | Interest Adjustment |
|----------|--------------------:|-----------------|--------------------:|
| LOW      |            < 0.30   | Auto-Approve    |              +0.0%  |
| MEDIUM   |       0.30 – 0.60   | Manual Review   |              +1.5%  |
| HIGH     |            > 0.60   | Decline / Refer |              +3.0%  |

## Confidence Scoring

| Confidence Level | Range         | Interpretation                      |
|------------------|---------------|-------------------------------------|
| HIGH             | ≥ 0.85        | Strong signal; auto-action allowed  |
| MODERATE         | 0.60 – 0.85   | Reasonable signal; review suggested |
| LOW              | < 0.60        | Weak signal; escalate to human      |

## Model Performance Requirements

The production risk model must meet these minimum thresholds on a held-out
test set before deployment:

| Metric    | Minimum |
|-----------|--------:|
| AUC-ROC   |   0.85  |
| Precision |   0.70  |
| Recall    |   0.65  |
| F1 Score  |   0.67  |

## Feature Importance Monitoring

Top features must be tracked monthly for drift:

1. `loan_percent_income` (DTI proxy)
2. `loan_grade_num`
3. `loan_int_rate`
4. `person_income`
5. `cb_person_default_on_file`

A feature importance shift of > 20% relative to baseline triggers model
retraining review.

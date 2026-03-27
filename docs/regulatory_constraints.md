# Regulatory Constraints

## Fair Lending Compliance (ECOA / Regulation B)

The system must NOT use the following as decision factors:

- Race or ethnicity
- Gender or sex
- Marital status
- Religion
- National origin

### Proxy Variable Monitoring

Even though the dataset does not contain protected attributes directly,
the following features must be monitored for proxy discrimination:

| Feature                 | Potential Proxy For | Monitoring Action               |
|-------------------------|---------------------|----------------------------------|
| `person_home_ownership` | Race / Income class | Disparate impact analysis        |
| `person_age`            | Protected class     | Age must not be sole factor      |
| `loan_intent`           | Cultural patterns   | Review for disparate treatment   |

## Explainable AI Requirements (SR 11-7 / OCC Guidance)

1. Every decline must include a human-readable explanation
2. Top contributing factors must be disclosed to the applicant
3. SHAP-based feature attributions must accompany each decision
4. Model documentation must include:
   - Training data description
   - Feature definitions
   - Performance metrics on protected subgroups

## Adverse Action Notices

When a loan is declined, the system must generate an adverse action
notice containing:

1. The specific reasons for denial (top 4 factors)
2. The applicant's right to request a copy of any reports used
3. Contact information for the creditor
4. Notice of the right to a free credit report within 60 days

## Data Retention

- Application data must be retained for 25 months minimum
- Model decisions and explanations must be retained for 5 years
- Audit logs must be immutable and timestamped

## Model Governance

- Model must be retrained at least annually
- Bias testing must be conducted quarterly
- An independent model validation must occur before deployment
- All model changes must go through a change management process

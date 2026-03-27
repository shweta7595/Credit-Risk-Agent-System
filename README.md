# Credit Risk Underwriting Agent System

An AI-powered multi-agent system for automated credit risk underwriting, built with **LangGraph** for agent orchestration, **scikit-learn Gradient Boosting** (production) with **Random Forest** as a training-time benchmark, **SHAP** on the production model, and **LangSmith** for monitoring.

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ Data Aggregation │───►│ Feature Engineering  │───►│  Risk Modeling  │
│     Agent        │    │       Agent          │    │     Agent       │
└─────────────────┘    └──────────────────────┘    └────────┬────────┘
                                                            │
                       ┌──────────────────────┐    ┌────────▼────────┐
                       │ Decision Explanation  │◄───│ Policy Validation│
                       │       Agent (LLM)     │    │      Agent       │
                       └──────────┬───────────┘    └─────────────────┘
                                  │
                       ┌──────────▼───────────┐
                       │    LLM Judge Agent   │
                       │ (audit / 2nd line)   │
                       └──────────────────────┘
```

### Agents

| Agent | Responsibility |
|-------|---------------|
| **Data Aggregation** | Validates and normalizes raw applicant data, flags anomalies |
| **Feature Engineering** | WOE/IV analysis, income ratios, risk buckets, employment quantiles, one-hot encoding |
| **Risk Modeling** | Runs production **Gradient Boosting**; PR-tuned threshold; **SHAP** (TreeExplainer on GB) |
| **Policy Validation** | Checks against credit policy, risk thresholds, and regulatory constraints |
| **Decision Explanation** | Generates human-readable, compliant underwriting reports via LLM |
| **LLM Judge** | Independent review: consistency of decision vs score/policy, fairness red flags, **CONCUR** / **FLAG_FOR_REVIEW** / **CHALLENGE** (does not auto-override the system decision) |

## Dataset

Uses the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle:
- **32,581** loan applications
- **12 raw features** → **41 model features** after engineering
- **Target**: `loan_status` (0 = no default, 1 = default)

## Feature Engineering

| Category | Features | Description |
|----------|----------|-------------|
| **Income Ratios** | DTI, PTI, monthly_payment_est, loan_to_income | Debt-to-income, payment-to-income, credit utilisation |
| **Risk Buckets** | loan_grade_bucket, age_bucket, cred_hist_bucket | Prime/subprime, age groups, credit history bins |
| **Employment** | emp_length_quantile, emp_length_missing | Quantile-based bucketing with missingness flag |
| **WOE/IV Analysis** | Computed for all categoricals | loan_grade IV=0.88, home_ownership IV=0.38 |
| **One-Hot Encoding** | 29 dummy variables | All categoricals encoded with drop_first=True |

## Model Performance

**Production:** Gradient Boosting (300 estimators, `max_depth=5`, balanced `sample_weight`). **Threshold** is chosen from the **precision–recall curve** on the test set (default strategy: max precision subject to recall ≥ 0.85) and saved to `models/threshold.joblib`.

**Benchmark:** Random Forest (500 trees, `max_depth=16`, `class_weight=balanced_subsample`) is trained on the same split and printed at `t=0.50` only — **not** deployed or saved for inference.

Run `python -m src.train` for current numbers. Typical production GB on this dataset is around **AUC ≈ 0.95** at `t=0.50` (RF benchmark is usually lower).

### Information Values (IV)

| Feature | IV | Predictive Power |
|---------|---:|-----------------|
| loan_grade | 0.8817 | Very Strong |
| person_home_ownership | 0.3770 | Strong |
| cb_person_default_on_file | 0.1640 | Medium |
| loan_intent | 0.0958 | Weak |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=your-key
#   LANGCHAIN_API_KEY=your-langsmith-key (optional, for monitoring)
```

### 3. Download Dataset & Train Model

```bash
python -m src.train
```

This will:
- Download the dataset from Kaggle (via kagglehub)
- Clean data (outlier capping, missing value imputation with missingness flags)
- Engineer 41 features (income ratios, risk buckets, WOE/IV, one-hot encoding)
- Train **Gradient Boosting** (production) and **Random Forest** (benchmark metrics only)
- Tune and save the production **threshold** from the GB precision–recall curve
- Save model artifacts to `models/`

### 4. Run the Agent System

```bash
python main.py
```

Processes 3 sample applicants (low/medium/high risk) through the full 6-agent LangGraph pipeline (including LLM judge) and outputs decisions with explainability and audit verdicts.

### 5. Underwriting chat UI (Streamlit)

After training, run the browser UI from the repository root:

```bash
streamlit run ui/app.py
```

Sign in with users defined in **`APP_RBAC_USERS`** (see `.env.example`), or leave it unset to use **demo mode** (choose a role without a password). Enter applicant fields in the sidebar and click **Run underwriting**; results appear as chat messages, **filtered by role** (see below). Free-text notes in the chat are passed through **regex PII redaction** before being stored in the session (SSN-like, card-like, email, US phone).

### Security: LLM guardrails, PII, and RBAC

- **Bias and PII system prompts**: The Decision Explanation and LLM Judge agents prepend a shared system block (`src/security/llm_guardrails.py`) that instructs fair-lending–aware, neutral language and forbids inferring protected attributes, inventing identifiers, or echoing sensitive data.
- **PII redaction**: `src/security/redact.py` provides `redact_pii()` for free-text channels (used in the UI).
- **RBAC**: `src/security/rbac.py` maps users from **`APP_RBAC_USERS`** JSON to roles **`viewer`**, **`underwriter`**, **`auditor`**, **`admin`**. The UI calls `filter_pipeline_result_for_role()` so viewers see coarse outcomes only (no full explanation, SHAP, judge, or trace); underwriters see operational detail but truncated judge rationale and no judge compliance notes; auditors and admins see the full pipeline output including trace and judge.

For production, replace plaintext passwords in `APP_RBAC_USERS` with an identity provider or hashed secrets.

## Output

For each applicant, the system produces:

- **Decision**: `APPROVE`, `DECLINE`, or `MANUAL_REVIEW`
- **Risk Score**: Default probability (0.0 – 1.0)
- **Risk Tier**: `LOW`, `MEDIUM`, or `HIGH`
- **Confidence Score**: Model confidence in the prediction
- **Explainability Report**: Human-readable report with top risk factors (SHAP-based)
- **Policy Violations**: List of any credit policy or regulatory violations
- **LLM Judge**: Verdict (`CONCUR`, `FLAG_FOR_REVIEW`, `CHALLENGE`, or `SKIPPED`), rationale, concerns list, compliance notes
- **Agent Trace**: Timestamped execution log of all 6 agents

## LangSmith Monitoring

When `LANGCHAIN_API_KEY` is set in `.env`, all agent executions are automatically traced to [LangSmith](https://smith.langchain.com):

- View the full agent graph execution
- Monitor LLM calls (Decision Explanation + LLM Judge agents)
- Track latency, token usage, and costs
- Debug individual agent steps

Project name: `credit-risk-agent-system`

## Policy Documents

| Document | Purpose |
|----------|---------|
| [`docs/credit_policy.md`](docs/credit_policy.md) | Eligibility criteria, loan parameters, auto-approve/decline rules |
| [`docs/risk_thresholds.md`](docs/risk_thresholds.md) | Risk tiers, confidence levels, model performance requirements |
| [`docs/regulatory_constraints.md`](docs/regulatory_constraints.md) | Fair lending (ECOA), explainable AI, adverse action notices |

## Project Structure

```
Credit-Risk-Agent-System/
├── main.py                          # Entry point – runs 3 sample applicants
├── ui/
│   └── app.py                       # Streamlit chat UI (RBAC + underwriting runs)
├── requirements.txt
├── .env.example
├── data/
│   ├── credit_risk_dataset.csv      # Raw Kaggle dataset
│   └── credit_risk_clean.csv        # Cleaned dataset
├── models/
│   ├── risk_model.joblib            # Trained Gradient Boosting (production)
│   ├── scaler.joblib                # StandardScaler for continuous features
│   ├── feature_names.joblib         # Ordered feature name list (41)
│   ├── scaled_columns.joblib        # Columns the scaler was fit on (11)
│   └── emp_quantile_edges.joblib    # Employment length quantile bin edges
├── docs/
│   ├── credit_policy.md
│   ├── risk_thresholds.md
│   └── regulatory_constraints.md
└── src/
    ├── security/
    │   ├── llm_guardrails.py        # Bias + PII instructions prepended to LLM system prompts
    │   ├── redact.py                # PII redaction for free text
    │   └── rbac.py                  # Roles, login helper, pipeline result filtering
    ├── data_ingestion.py            # Download, validate, clean, missingness flags
    ├── feature_engineering.py       # WOE/IV, income ratios, buckets, one-hot
    ├── risk_model.py                # Model training, evaluation, SHAP
    ├── train.py                     # Training pipeline script
    └── agents/
        ├── state.py                 # Pydantic state definitions
        ├── graph.py                 # LangGraph pipeline orchestration
        ├── data_aggregation.py      # Agent 1: Data validation
        ├── feature_engineering_agent.py  # Agent 2: 41-feature computation
        ├── risk_modeling.py         # Agent 3: Model scoring + SHAP
        ├── policy_validation.py     # Agent 4: Policy checks
        ├── decision_explanation.py  # Agent 5: LLM explanation report
        └── llm_judge.py             # Agent 6: LLM audit / judge
```

## Compliance

- **Bias Checks**: Proxy variable monitoring for protected attributes; LLM agents use explicit fair-lending and non-discrimination instructions in system prompts
- **Explainable AI**: SHAP-based feature attributions for every decision (withheld at `viewer` role in the UI)
- **Regulatory Adherence**: ECOA/Reg B compliant adverse action notices
- **Audit Trail**: Full agent trace with timestamps for every decision (visible to `auditor` / `admin` in the UI)

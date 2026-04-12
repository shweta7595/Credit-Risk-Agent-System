# CreditGenie — AI-Powered Credit Risk Underwriting

CreditGenie is a production-grade multi-agent system for automated credit risk underwriting. It combines a **scikit-learn Gradient Boosting** model (calibrated, PR-threshold tuned) with a **LangGraph** agent pipeline, **SHAP** explainability, a **Groq/Llama 3** LLM judge for independent audit, and a role-gated **Streamlit** UI with a built-in admin monitoring dashboard.

---

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ Data Aggregation │───►│ Feature Engineering  │───►│  Risk Modeling  │
│     Agent        │    │       Agent          │    │     Agent       │
└─────────────────┘    └──────────────────────┘    └────────┬────────┘
                                                            │
                       ┌──────────────────────┐    ┌────────▼────────┐
                       │ Decision Explanation  │◄───│ Policy Validation│
                       │     Agent  (LLM)      │    │      Agent       │
                       └──────────┬───────────┘    └─────────────────┘
                                  │
                       ┌──────────▼───────────┐
                       │    LLM Judge Agent   │
                       │  (2nd-line audit)    │
                       └──────────────────────┘
```

### Agents

| Agent | Responsibility |
|---|---|
| **Data Aggregation** | Validates and normalises raw applicant data, flags anomalies |
| **Feature Engineering** | WOE/IV analysis, income ratios, risk buckets, employment quantiles, one-hot encoding |
| **Risk Modeling** | Calibrated Gradient Boosting; PR-tuned threshold; SHAP TreeExplainer |
| **Policy Validation** | Checks eligibility criteria, risk thresholds, and regulatory constraints |
| **Decision Explanation** | Internal analyst memo via LLM (Groq/Llama 3) — risk summary, factor analysis, recommendation |
| **LLM Judge** | Independent second-line auditor: **CONCUR / FLAG_FOR_REVIEW / CHALLENGE** (does not auto-override) |

---

## Quick Start

```bash
git clone <repo-url>
cd Credit-Risk-Agent-System
./setup.sh
```

`setup.sh` will:
- Check Python ≥ 3.10
- Create and activate a virtual environment (`.venv/`)
- Install all dependencies from `requirements.txt`
- Copy `.env.example` → `.env` if no `.env` exists
- Optionally download the dataset and train the model

---

## Manual Setup

### 1. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | Free key at [console.groq.com](https://console.groq.com) |
| `LANGCHAIN_API_KEY` | Optional | LangSmith tracing — [smith.langchain.com](https://smith.langchain.com) |
| `APP_RBAC_USERS` | Optional | JSON user directory for login (see `.env.example`) |

### 3. Train the model

```bash
python -m src.train
```

Downloads the Kaggle credit risk dataset, engineers 41 features, trains a calibrated Gradient Boosting model, tunes the decision threshold on the PR curve, and saves all artifacts to `models/`.

### 4. Start CreditGenie

```bash
streamlit run ui/app.py
```

---

## CreditGenie UI

The Streamlit app runs on **http://localhost:8501** and has two pages.

### Underwriting Workspace

Sign in using the role selector (dev mode, no password required) or with credentials from `APP_RBAC_USERS`. Enter applicant details in the sidebar and click **Run underwriting** — results appear as chat messages filtered by role.

- **Admin users** see a highlighted **"📊 Risk Model Monitoring"** button in the top-right corner of the workspace, which navigates to the monitoring page.
- **Auditor / Admin** results include a collapsed **"View agent trace"** expander beneath each decision.
- Free-text notes in the chat input are PII-redacted before display.

### Admin Monitoring Page (admin only)

Accessible via the **📊 Risk Model Monitoring** button. Contains two tabs:

**📈 Model Performance**
- Key metrics row: AUC-ROC, Average Precision, Accuracy, Decision Threshold, Default Recall
- Per-class precision / recall / F1 / support table (No Default vs Default)
- Confusion matrix
- PR curve + ROC curve (generated at training time)

**📋 Session Log**
- Persistent log of every underwriting run across all sessions and all users
- Filter by: user · decision · risk tier · date
- Download as CSV

---

## Role-Based Access Control (RBAC)

| Role | Decision | Risk Score | SHAP | Explanation | Policy Violations | LLM Judge | Agent Trace |
|---|---|---|---|---|---|---|---|
| **viewer** | ✓ | ✓ | — | Withheld | Count only | — | — |
| **underwriter** | ✓ | ✓ | ✓ | ✓ | ✓ | Partial | — |
| **auditor** | ✓ | ✓ | ✓ | ✓ | ✓ | Full | Expander |
| **admin** | ✓ | ✓ | ✓ | ✓ | ✓ | Full | Expander |

Demo mode (no `APP_RBAC_USERS` set): select any role at login, no password required.

---

## Dataset

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — downloaded automatically via `kagglehub`.

- **32,581** loan applications
- **12 raw features** → **41 model features** after engineering
- **Target**: `loan_status` (0 = no default, 1 = default)

---

## Feature Engineering

| Category | Features | Description |
|---|---|---|
| **Income Ratios** | DTI, PTI, monthly_payment_est, loan_to_income | Debt-to-income, payment-to-income, credit utilisation |
| **Risk Buckets** | loan_grade_bucket, age_bucket, cred_hist_bucket | Prime/subprime, age groups, credit history bins |
| **Employment** | emp_length_quantile, emp_length_missing | Quantile-based bucketing with missingness flag |
| **WOE/IV** | Computed for all categoricals | loan_grade IV=0.88, home_ownership IV=0.38 |
| **One-Hot** | 29 dummy variables | All categoricals encoded with drop_first=True |

---

## Model Performance

**Production model:** Gradient Boosting (300 estimators, `max_depth=5`, balanced `sample_weight`), followed by isotonic calibration to correct probability inflation. Threshold tuned on the PR curve (strategy: max precision with recall ≥ 0.85).

**Benchmark:** Random Forest (500 trees, `max_depth=16`, `class_weight=balanced_subsample`) — trained for comparison only, not deployed.

Typical results on the test set (calibrated model, tuned threshold):

| Metric | Value |
|---|---|
| AUC-ROC | ~0.949 |
| Default Recall | ~0.855 |
| Default Precision | ~0.689 |
| Default F1 | ~0.763 |
| Accuracy | ~0.884 |

Full metrics and PR/ROC curves are displayed in the **Admin Monitoring** page and regenerated on each `python -m src.train` run.

### Information Values (IV)

| Feature | IV | Predictive Power |
|---|---|---|
| loan_grade | 0.8817 | Very Strong |
| person_home_ownership | 0.3770 | Strong |
| cb_person_default_on_file | 0.1640 | Medium |
| loan_intent | 0.0958 | Weak |

---

## LangSmith — Tracing & Evaluation

Set `LANGCHAIN_API_KEY` in `.env` to enable automatic tracing of every pipeline run to [LangSmith](https://smith.langchain.com) under project `credit-risk-agent-system`.

### Evaluation suite

```bash
python -m eval.run_eval           # runs eval against credit-risk-eval-v2 dataset
python -m eval.monitor --n 100    # prints monitoring report from recent LangSmith runs
```

**Per-example evaluators:** decision match, risk tier match, policy correctness, score range, judge ran, judge coherence.

**Summary evaluators:** decision accuracy, precision/recall/F1, Brier score, policy recall.

---

## Security

- **LLM guardrails** (`src/security/llm_guardrails.py`): system prompt block prepended to all LLM calls — fair-lending aware, forbids inferring protected attributes, echoing PII, or inventing identifiers.
- **PII redaction** (`src/security/redact.py`): regex redaction for SSN-like, card-like, email, and US phone patterns — applied to all free-text chat input.
- **RBAC** (`src/security/rbac.py`): user directory in `APP_RBAC_USERS` JSON; `filter_pipeline_result_for_role()` enforces field-level access per role.

For production: replace plaintext passwords in `APP_RBAC_USERS` with bcrypt hashes or delegate to an identity provider.

---

## Policy Documents

| Document | Purpose |
|---|---|
| [`docs/credit_policy.md`](docs/credit_policy.md) | Eligibility criteria, loan parameters, auto-approve/decline rules |
| [`docs/risk_thresholds.md`](docs/risk_thresholds.md) | Risk tiers, confidence levels, model performance requirements |
| [`docs/regulatory_constraints.md`](docs/regulatory_constraints.md) | Fair lending (ECOA), explainable AI, adverse action notices |

---

## Project Structure

```
Credit-Risk-Agent-System/
├── setup.sh                          # One-command local setup
├── main.py                           # CLI — runs 3 sample applicants
├── requirements.txt
├── .env.example                      # Environment variable template
├── ui/
│   └── app.py                        # CreditGenie Streamlit app (2 pages)
├── eval/
│   ├── dataset.py                    # LangSmith golden + challenging eval examples
│   ├── evaluators.py                 # Per-example and summary evaluators
│   ├── run_eval.py                   # Run LangSmith evaluation suite
│   └── monitor.py                    # Pull recent runs, compute PSI + distributions
├── data/
│   ├── credit_risk_dataset.csv       # Raw Kaggle dataset (auto-downloaded)
│   ├── credit_risk_clean.csv         # Cleaned dataset
│   ├── pr_curve_calibrated.png       # PR / ROC curves (generated by train)
│   └── session_log.jsonl             # Persistent underwriting session log
├── models/
│   ├── risk_model.joblib             # Calibrated GB model (scoring)
│   ├── risk_model_raw.joblib         # Raw GB model (SHAP TreeExplainer)
│   ├── scaler.joblib                 # StandardScaler
│   ├── feature_names.joblib          # Ordered feature list (41)
│   ├── scaled_columns.joblib         # Columns the scaler was fit on
│   ├── threshold.joblib              # Optimal decision threshold
│   └── model_metrics.joblib          # Saved evaluation metrics (for monitoring UI)
├── docs/
│   ├── credit_policy.md
│   ├── risk_thresholds.md
│   └── regulatory_constraints.md
└── src/
    ├── security/
    │   ├── llm_guardrails.py         # Bias + PII system prompt block
    │   ├── redact.py                 # PII redaction
    │   └── rbac.py                   # Roles, auth, result filtering
    ├── data_ingestion.py             # Download, validate, clean
    ├── feature_engineering.py        # WOE/IV, ratios, buckets, one-hot
    ├── risk_model.py                 # Training, calibration, SHAP, threshold tuning
    ├── train.py                      # Full training pipeline script
    └── agents/
        ├── state.py                  # LangGraph state schema
        ├── graph.py                  # Pipeline orchestration + LangSmith tracing
        ├── llm_provider.py           # Groq/Llama 3 factory
        ├── data_aggregation.py       # Agent 1
        ├── feature_engineering_agent.py  # Agent 2
        ├── risk_modeling.py          # Agent 3
        ├── policy_validation.py      # Agent 4
        ├── decision_explanation.py   # Agent 5 — internal analyst memo
        └── llm_judge.py              # Agent 6 — independent audit
```

---

## Output

For each applicant the pipeline produces:

| Field | Description |
|---|---|
| **Decision** | `APPROVE` / `DECLINE` / `MANUAL_REVIEW` |
| **Risk Score** | Default probability 0.0 – 1.0 (calibrated) |
| **Risk Tier** | `LOW` / `MEDIUM` / `HIGH` |
| **Confidence** | Model confidence in the prediction |
| **Explanation Report** | Internal analyst memo with top SHAP factors |
| **Policy Violations** | List of breached credit policy rules |
| **LLM Judge Verdict** | `CONCUR` / `FLAG_FOR_REVIEW` / `CHALLENGE` / `SKIPPED` |
| **Agent Trace** | Timestamped execution log of all 6 agents |

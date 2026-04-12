#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CreditGenie — Local Setup Script
# Usage: ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR=".venv"
PYTHON_MIN="3.10"

# ── helpers ───────────────────────────────────────────────────────────────────
info()    { echo "  [INFO]  $*"; }
success() { echo "  [OK]    $*"; }
warn()    { echo "  [WARN]  $*"; }
error()   { echo "  [ERROR] $*" >&2; exit 1; }
header()  { echo; echo "──────────────────────────────────────────"; echo "  $*"; echo "──────────────────────────────────────────"; }

# ── 1. Python version check ───────────────────────────────────────────────────
header "Checking Python version"

PYTHON_BIN=$(command -v python3 || command -v python || error "python3 not found — install Python $PYTHON_MIN or newer.")

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
MIN_MINOR=$(echo "$PYTHON_MIN" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt "$MIN_MINOR" ]; }; then
  error "Python $PYTHON_MIN+ required, found $PY_VERSION."
fi
success "Python $PY_VERSION"

# ── 2. Virtual environment ─────────────────────────────────────────────────────
header "Setting up virtual environment"

if [ -d "$VENV_DIR" ]; then
  info "Existing venv found at $VENV_DIR — skipping creation."
else
  info "Creating venv at $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  success "Virtual environment created."
fi

# Activate
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Activated $VENV_DIR"

# ── 3. Upgrade pip & install dependencies ─────────────────────────────────────
header "Installing dependencies"

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
success "All packages installed from requirements.txt"

# ── 4. Environment file ────────────────────────────────────────────────────────
header "Configuring environment"

if [ -f ".env" ]; then
  info ".env already exists — skipping copy."
else
  cp .env.example .env
  warn ".env created from .env.example"
  warn "You MUST edit .env and set GROQ_API_KEY before running the app."
  warn "  Get a free key at: https://console.groq.com"
fi

# Remind about LangSmith (optional)
if grep -q "^LANGCHAIN_API_KEY=$" .env 2>/dev/null; then
  warn "LANGCHAIN_API_KEY is empty — LangSmith tracing is disabled (optional)."
  warn "  Get a free key at: https://smith.langchain.com"
fi

# ── 5. Download dataset & train model ─────────────────────────────────────────
header "Training the risk model"

info "This downloads the Credit Risk Dataset from Kaggle via kagglehub"
info "and trains the Gradient Boosting model (~2-5 min on first run)."
echo

read -r -p "  Train the model now? [Y/n]: " TRAIN_ANSWER
TRAIN_ANSWER="${TRAIN_ANSWER:-Y}"

if [[ "$TRAIN_ANSWER" =~ ^[Yy]$ ]]; then
  python -m src.train
  success "Model artifacts saved to models/"
else
  warn "Skipped training. Run 'python -m src.train' before starting the app."
fi

# ── 6. Done ────────────────────────────────────────────────────────────────────
header "Setup complete"

echo "  Next steps:"
echo
echo "  1. Edit .env and fill in your GROQ_API_KEY:"
echo "       nano .env"
echo
echo "  2. Activate the virtual environment (if not already active):"
echo "       source $VENV_DIR/bin/activate"
echo
echo "  3. Start CreditGenie:"
echo "       streamlit run ui/app.py"
echo
echo "  4. (Optional) Run LangSmith evaluation:"
echo "       python -m eval.run_eval"
echo
echo "  5. (Optional) Run the pipeline CLI on sample applicants:"
echo "       python main.py"
echo

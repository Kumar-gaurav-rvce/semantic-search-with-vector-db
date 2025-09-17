#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Script: run_demo.sh
# Purpose: Start the Streamlit app with safe environment settings
#          (especially important on macOS / Apple Silicon where BLAS/MKL
#           libraries can cause segfaults if threads are unmanaged).
#
# Features:
#   - Ensures script runs from repo root
#   - Exports safe environment variables
#   - Activates local virtual environment if available
#   - Launches Streamlit app at http://localhost:8501
#
# Usage:
#   bash scripts/run_demo.sh
# ---------------------------------------------------------------------------

set -euo pipefail  # strict mode: fail early on errors/undefined vars/pipelines

# ---------------------------------------------------------------------------
# Ensure we are running from the project root
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Limit native library threading (prevents segfaults on macOS M1/M2)
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Optional extra safety
export OMP_WAIT_POLICY=PASSIVE
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# ---------------------------------------------------------------------------
# Activate virtual environment if available
# ---------------------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  source .venv/bin/activate
  echo "Activated virtual environment: .venv"
else
  echo "No .venv found — using system Python"
fi

# ---------------------------------------------------------------------------
# Launch Streamlit
# ---------------------------------------------------------------------------
echo "Starting Streamlit with safe environment vars..."
streamlit run app/streamlit_app.py

#!/usr/bin/env bash
# run.sh - reproduce every artifact from raw data with one command.
#
# Usage:
#   ./run.sh            # clean + train (regenerates model/ and output/)
#   ./run.sh install    # pip install -r requirements.txt
#   ./run.sh train      # train only (skip clean)
#   ./run.sh test       # run the pytest suite (fast subset; skips @slow)
#   ./run.sh clean      # remove generated model/output artifacts
#   ./run.sh external   # score the trained model against external UCI cohorts
#   ./run.sh benchmark  # supplementary: RF vs. Logistic Regression vs. Decision Tree (out of thesis scope, see docs/appendix_baseline_comparison.md)
#   ./run.sh tuning     # RQ3: default-hyperparameter RF vs. tuned RF (same algorithm, in scope)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Prefer `python` over `python3`: on Windows in particular, `python3` can
# resolve to the App Execution Alias stub (a distinct, partially-populated
# environment under WindowsApps/) rather than your real installation. Override
# with `PYTHON=python3 ./run.sh` if your system is the other way around.
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

install() {
    "$PYTHON" -m pip install -r requirements.txt
}

clean() {
    rm -f model/*.joblib model/*.json output/*.png output/*.json output/*.txt
}

train() {
    "$PYTHON" -m src.training.train_model
}

external() {
    "$PYTHON" -m src.analysis.external_validation
}

benchmark() {
    "$PYTHON" -m src.analysis.benchmark
}

tuning() {
    "$PYTHON" -m src.analysis.tuning_effect
}

run_tests() {
    "$PYTHON" -m pip show pytest >/dev/null 2>&1 || "$PYTHON" -m pip install -r requirements-dev.txt
    "$PYTHON" -m pytest -m "not slow"
}

case "${1:-all}" in
    install)   install ;;
    clean)     clean ;;
    train)     train ;;
    test)      run_tests ;;
    external)  external ;;
    benchmark) benchmark ;;
    tuning)    tuning ;;
    all)       clean; train ;;
    *) echo "Usage: $0 [install|clean|train|test|external|benchmark|tuning|all]" >&2; exit 1 ;;
esac

#!/bin/bash
# Run full convergence comparison: CUDA (requires uninstall), then Vulkan+CPU, then plot.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CUDA_RESULTS="$SCRIPT_DIR/cuda_results.json"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "======================================================================"
echo "PHASE 1: CUDA Training (torch_vulkan temporarily uninstalled)"
echo "======================================================================"

pip uninstall torch_vulkan -y -q 2>/dev/null || true

python3 "$SCRIPT_DIR/cuda_batch_runner.py" "$CUDA_RESULTS" 2>/dev/null

echo ""
echo "--- Reinstalling torch_vulkan ---"
pip install -e . -q 2>/dev/null

echo ""
echo "======================================================================"
echo "PHASE 2: Vulkan + CPU Training + Plotting"
echo "======================================================================"

python3 "$SCRIPT_DIR/plot_convergence.py" 2>/dev/null

echo ""
echo "Done! Check tests/convergence_plots/ for output."

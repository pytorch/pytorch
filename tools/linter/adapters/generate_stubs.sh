#!/usr/bin/env bash
# Script to generate PyTorch .pyi stub files
# This script should be run from the PyTorch repository root

set -euo pipefail

# Find repository root (try git, then hg, then use pwd)
if REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null); then
    :
elif REPO_ROOT=$(hg root 2>/dev/null); then
    :
else
    REPO_ROOT=$(pwd)
fi

cd "$REPO_ROOT"

# File to store the commit hash when stubs were last generated
STUB_COMMIT_FILE="$REPO_ROOT/.pyrefly_stub_commit"

echo "Generating .pyi stub files..."

# Step 1: Generate torch version
echo "Generating torch version..."
python3 -m tools.generate_torch_version --is_debug=false

# Step 2: Generate main stub files
echo "Generating main stub files..."
python3 -m tools.pyi.gen_pyi \
    --native-functions-path aten/src/ATen/native/native_functions.yaml \
    --tags-path aten/src/ATen/native/tags.yaml \
    --deprecated-functions-path tools/autograd/deprecated.yaml

# Step 3: Generate DataPipe stub files
echo "Generating DataPipe stub files..."
python3 torch/utils/data/datapipes/gen_pyi.py

# Save the current commit hash to track when stubs were last generated
# Try git first, then hg, then mark as unknown
if CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null); then
    :
elif CURRENT_COMMIT=$(hg id -i 2>/dev/null); then
    :
else
    CURRENT_COMMIT="unknown"
fi

echo "$CURRENT_COMMIT" > "$STUB_COMMIT_FILE"

echo "All stub files generated successfully"
echo "Saved commit hash: $CURRENT_COMMIT"

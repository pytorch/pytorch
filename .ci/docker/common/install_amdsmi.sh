#!/bin/bash

set -ex

# Detect ROCM_PATH dynamically
if command -v rocm-sdk &> /dev/null && python3 -m rocm_sdk path --root &> /dev/null; then
    # theRock installation
    ROCM_PATH="$(python3 -m rocm_sdk path --root)"
else
    # Traditional installation
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
fi

echo "Installing amdsmi from: ${ROCM_PATH}/share/amd_smi"
cd ${ROCM_PATH}/share/amd_smi && pip install .

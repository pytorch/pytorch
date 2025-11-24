#!/bin/bash

set -ex

# Source common script to detect ROCM_PATH
source "$(dirname "${BASH_SOURCE[0]}")/detect_rocm_path.sh"

echo "Installing amdsmi from: ${ROCM_PATH}/share/amd_smi"
cd ${ROCM_PATH}/share/amd_smi && pip install .

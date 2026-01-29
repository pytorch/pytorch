#!/bin/bash

set -ex

source /etc/rocm_env.sh

# For theRock nightly, amd_smi may already be installed or in a different location
if [[ -d "${ROCM_PATH}/share/amd_smi" ]]; then
  echo "Installing amdsmi from: ${ROCM_PATH}/share/amd_smi"
  cd ${ROCM_PATH}/share/amd_smi && pip install .
else
  echo "AMD SMI not found at ${ROCM_PATH}/share/amd_smi - skipping (may already be installed via pip)"
fi

#!/bin/bash

set -ex

source /etc/rocm_env.sh

# Install the amdsmi Python module. There are several possible locations
# depending on how ROCm was installed (packages vs theRock tarballs).
if [ -d "${ROCM_PATH}/share/amd_smi" ]; then
  echo "Installing amdsmi from: ${ROCM_PATH}/share/amd_smi"
  cd "${ROCM_PATH}/share/amd_smi" && python3 -m pip install .
elif [ -d "${ROCM_PATH}/lib/amd_smi" ]; then
  echo "Installing amdsmi from: ${ROCM_PATH}/lib/amd_smi"
  cd "${ROCM_PATH}/lib/amd_smi" && python3 -m pip install .
else
  echo "AMD SMI source not found - checking if already importable..."
  if python3 -c "import amdsmi" 2>/dev/null; then
    echo "amdsmi is already importable"
  else
    echo "WARNING: amdsmi Python module not found. GPU monitoring via amdsmi will be unavailable."
    echo "To install manually: pip install amdsmi, or set PYTHONPATH to include the amd_smi directory."
  fi
fi

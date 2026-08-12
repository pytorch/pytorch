#!/bin/bash

set -ex

source /etc/rocm_env.sh

# Install the amdsmi Python module. Older ROCm packages include project metadata,
# while ROCm 10.1 wheels provide only the importable package directory.
AMDSMI_PATH="${ROCM_PATH}/share/amd_smi"
if [ -f "${AMDSMI_PATH}/setup.py" ] || [ -f "${AMDSMI_PATH}/pyproject.toml" ]; then
  echo "Installing amdsmi from: ${AMDSMI_PATH}"
  python3 -m pip install "${AMDSMI_PATH}"
elif [ -d "${AMDSMI_PATH}/amdsmi" ]; then
  echo "Exposing amdsmi from: ${AMDSMI_PATH}"
  AMDSMI_PATH="${AMDSMI_PATH}" python3 - <<'PY'
import os
import sysconfig
from pathlib import Path


purelib = Path(sysconfig.get_path("purelib"))
(purelib / "rocm_amdsmi.pth").write_text(os.environ["AMDSMI_PATH"] + "\n")
PY
  python3 -c "import amdsmi"
elif python3 -c "import amdsmi" 2>/dev/null; then
  echo "amdsmi is already importable"
else
  echo "WARNING: amdsmi Python module not found. GPU monitoring via amdsmi will be unavailable."
  echo "To install manually: pip install amdsmi, or set PYTHONPATH to include the amd_smi directory."
fi

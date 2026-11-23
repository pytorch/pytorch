#!/bin/bash

set -ex

source /etc/rocm_env.sh

# AMD SMI is staged under the ROCm root and is intended to be imported by
# adding its share directory to Python's search path.
AMDSMI_SHARE_DIR="${ROCM_PATH}/share/amd_smi"
if [ -d "${AMDSMI_SHARE_DIR}/amdsmi" ]; then
  echo "Exposing amdsmi from: ${AMDSMI_SHARE_DIR}"
  SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
  printf '%s\n' "${AMDSMI_SHARE_DIR}" > "${SITE_PACKAGES}/amdsmi.pth"
  python3 -c "import amdsmi"
elif python3 -c "import amdsmi" 2>/dev/null; then
  echo "amdsmi is already importable"
else
  echo "WARNING: amdsmi Python module not found. GPU monitoring via amdsmi will be unavailable."
  echo "Set PYTHONPATH to include ${AMDSMI_SHARE_DIR}."
fi

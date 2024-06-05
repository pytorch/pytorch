#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# Cache the test models at ~/.cache/torch/hub/
IMPORT_SCRIPT_FILENAME="/tmp/torchvision_import_script.py"
as_jenkins echo 'import torchvision; torchvision.models.mobilenet_v2(pretrained=True); torchvision.models.mobilenet_v3_large(pretrained=True);' > "${IMPORT_SCRIPT_FILENAME}"

pip_install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Very weird quoting behavior here https://github.com/conda/conda/issues/10972,
# so echo the command to a file and run the file instead
conda_run python "${IMPORT_SCRIPT_FILENAME}"

# Cleaning up
conda_run pip uninstall -y torch torchvision
rm "${IMPORT_SCRIPT_FILENAME}" || true

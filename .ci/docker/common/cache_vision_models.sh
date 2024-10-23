#!/bin/bash

set -ex

# Skip pytorch-nightly installation in docker images
# Installation of pytorch-nightly is needed to prefetch mobilenet_v2 avd v3 models for some tests.
# Came from https://github.com/ROCm/pytorch/commit/85bd6bc0105162293fa0bbfb7b661f85ec67f85a
# Models are downloaded on first use to the folder /root/.cache/torch/hub
# But pytorch-nightly installation also overrides .ci/docker/requirements-ci.txt settings
# and upgrades some of python packages (sympy from 1.12.0 to 1.13.0)
# which causes several 'dynamic_shapes' tests to fail
# Skip prefetching models affects these tests without any errors:
#   python test/mobile/model_test/gen_test_model.py mobilenet_v2
#   python test/quantization/eager/test_numeric_suite_eager.py -k test_mobilenet_v3
# Issue https://github.com/ROCm/frameworks-internal/issues/8772
echo "Skip torch-nightly installation"
exit 0

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

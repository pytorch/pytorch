#!/bin/bash

set -ex

# 1. Guard Clause
# This ensures the script exits harmlessly on non-TPU builds
if [ -z "${TORCH_TPU}" ]; then
  echo "TORCH_TPU is not set. Skipping TorchTPU installation..."
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# 2. Configuration
TORCH_TPU_REPO="${TORCH_TPU_REPO:-https://github.com/google-ml-infra/torch_tpu.git}"
TORCH_TPU_BRANCH="${TORCH_TPU_BRANCH:-main}"

# Pin File Configuration
TORCH_TPU_TEXT_FILE="${TORCH_TPU_TEXT_FILE:-/var/lib/jenkins/workspace/.github/ci_commit_pins/torch_tpu.txt}"
if [ -f "${TORCH_TPU_TEXT_FILE}" ]; then
    TORCH_TPU_PINNED_COMMIT=$(cat "${TORCH_TPU_TEXT_FILE}")
fi

# 3. Install Bazel (Root Step)
# We install to /usr/local/bin so it is available to all users (root & jenkins)
if ! command -v bazel &> /dev/null; then
    echo "Bazel not found. Installing Bazelisk..."
    curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
    chmod +x /usr/local/bin/bazel
else
    echo "Bazel is already installed."
fi
bazel --version

# 4. Preparation
mkdir -p /var/lib/jenkins/torch_tpu
chown -R jenkins /var/lib/jenkins/torch_tpu
pushd /var/lib/jenkins/

# 5. Clone
as_jenkins git clone --recursive "${TORCH_TPU_REPO}" torch_tpu
cd torch_tpu

# 6. Checkout
if [ -n "${TORCH_TPU_PINNED_COMMIT}" ]; then
    echo "Checking out pinned commit: ${TORCH_TPU_PINNED_COMMIT}"
    as_jenkins git checkout "${TORCH_TPU_PINNED_COMMIT}"
else
    echo "No pinned commit found at ${TORCH_TPU_TEXT_FILE}. Checking out branch: ${TORCH_TPU_BRANCH}"
    as_jenkins git checkout "${TORCH_TPU_BRANCH}"
fi

as_jenkins git submodule update --init --recursive

# 7. JAX/LibTPU Dependencies (Runtime)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "${SCRIPT_DIR}/requirements_tpu.txt" ]; then
    pip_install -r "${SCRIPT_DIR}/requirements_tpu.txt"
else
    # Fallback to repo root (standard PyTorch CI location)
    PYTORCH_ROOT="${PYTORCH_ROOT:-/opt/pytorch/pytorch}"
    if [ -f "${PYTORCH_ROOT}/requirements_tpu.txt" ]; then
        pip_install -r "${PYTORCH_ROOT}/requirements_tpu.txt"
    else
        echo "ERROR: requirements_tpu.txt not found!"
        echo "Checked locations:"
        echo "  1. ${SCRIPT_DIR}/requirements_tpu.txt"
        echo "  2. ${PYTORCH_ROOT}/requirements_tpu.txt"
        exit 1
    fi
fi

# 8. Build Dependencies
# Using the confirmed path: requirements/requirements.txt
pip_install -r requirements/requirements.txt

# 9. Build
echo "Building TorchTPU Wheel..."
export TORCH_SOURCE=$(python -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))")

as_jenkins env TORCH_SOURCE="${TORCH_SOURCE}" bazel build //ci/wheel:torch_tpu_wheel --config=local --define WHEEL_VERSION=0.1.0 --define TORCH_SOURCE=local

# 10. Install
pip_install bazel-bin/ci/wheel/*.whl

# 11. Cleanup
popd # Back to /var/lib/jenkins

echo "Cleaning up build artifacts..."
rm -rf torch_tpu

# 12. Verification
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TORCH_LIB_PATH}"
echo "Updated LD_LIBRARY_PATH to include: ${TORCH_LIB_PATH}"

# Verify installation
python -c "import torch; from torch_tpu import api; print(f'Success! Device: {api.tpu_device()}')"

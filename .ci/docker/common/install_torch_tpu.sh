#!/bin/bash
set -ex

# 1. Guard Clause
# This ensures the script exits harmlessly on non-TPU builds
if [ -z "${TORCH_TPU}" ]; then
  echo "TORCH_TPU is not set. Skipping TorchTPU installation..."
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# 2. Tokens
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
START_PATH=$(pwd)

# 2. Add functions to pull TorchTPU prior to being fully OSS
# Cleanup function to ensure SSH key is removed
cleanup() {
    if [ -f "temp_ssh_key" ]; then
        echo "Cleaning up temporary SSH key..."
        rm -f "temp_ssh_key"
    fi
}

install_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        echo "gcloud CLI not found. Installing..."

        # Ensure curl and apt-transport-https are present
        sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates gnupg curl ssh

        # Import Google Cloud public key
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

        # Add the Cloud SDK distribution URI as a package source
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list #@lint-ignore

        # Update and install
        sudo apt-get update && sudo apt-get install -y google-cloud-cli
    else
        echo "gcloud CLI is already installed."
    fi
    if ! command -v ssh &> /dev/null; then
        echo "ssh is needed for private pulling of repo, installing"
        sudo apt-get update
        sudo apt-get install -y ssh
    else
        echo "ssh is installed"
    fi
}

fetch_secret() {
    echo "Fetching SSH key from Secret Manager..."

    # Check if xtrace (set -x) is enabled
    local xtrace_enabled=0
    if [[ "$-" == *x* ]]; then
        xtrace_enabled=1
        set +x
    fi

    if ! gcloud secrets versions access latest --secret="torchtpu-readonly-key" --project="ml-velocity-actions-testing" > "temp_ssh_key"; then
        echo "Error: Failed to fetch secret. Ensure you are authenticated with gcloud."

        # Restore xtrace if it was enabled, before exiting
        if [ $xtrace_enabled -eq 1 ]; then
            set -x
        fi
        exit 1
    fi

    # Restore xtrace if it was enabled
    if [ $xtrace_enabled -eq 1 ]; then
        set -x
    fi
}

clone_repo() {
    echo "Cloning torch tpu repository..."
    chmod 600 "temp_ssh_key"

    # Use GIT_SSH_COMMAND to specify the key and disable strict host key checking for automation
    export GIT_SSH_COMMAND="ssh -i temp_ssh_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
    if git clone --recursive "git@github.com:google-ml-infra/torch_tpu.git"; then
        echo "Repository cloned successfully."
    else
        echo "Error: Failed to clone repository."
        exit 1
    fi
}

pull_torch_tpu() {
    trap cleanup EXIT

    echo "Attempting to clone repository publicly..."
    if GIT_TERMINAL_PROMPT=0 git clone "${TORCH_TPU_REPO}" torch_tpu; then
         echo "Public clone successful."
         return 0
    fi

    echo "Public clone failed. Falling back to authenticated clone..."
    echo "Starting setup_repo.sh..."
    install_gcloud
    fetch_secret
    clone_repo
    echo "Done."
}

# sleep 28800 # Debug sleep to connect to runner to streamline debugging, do not submit

# 3. Configuration
TORCH_TPU_REPO="${TORCH_TPU_REPO:-https://github.com/google-ml-infra/torch_tpu.git}"
TORCH_TPU_BRANCH="${TORCH_TPU_BRANCH:-main}"

# Pin File Configuration
TORCH_TPU_TEXT_FILE="${TORCH_TPU_TEXT_FILE:-/var/lib/jenkins/workspace/.github/ci_commit_pins/torch_tpu.txt}"
if [ -f "${TORCH_TPU_TEXT_FILE}" ]; then
    TORCH_TPU_PINNED_COMMIT=$(cat "${TORCH_TPU_TEXT_FILE}")
fi

# 4. Install Bazel (Root Step)
# We install to /usr/local/bin so it is available to all users (root & jenkins)
if ! command -v bazel &> /dev/null; then
    echo "Bazel not found. Installing Bazelisk..."
    temp_dir=$(mktemp -d)
    curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 -o "${temp_dir}/bazel"
    sudo mv "${temp_dir}/bazel" /usr/local/bin/bazel
    sudo chmod +x /usr/local/bin/bazel
    rm -rf "${temp_dir}"
else
    echo "Bazel is already installed."
fi
bazel --version

# 5. Preparation
mkdir -p /var/lib/jenkins/
pushd /var/lib/jenkins/

# 6. Clone
pull_torch_tpu
chown -R jenkins /var/lib/jenkins/torch_tpu
pushd torch_tpu

# 7. Checkout
if [ -n "${TORCH_TPU_PINNED_COMMIT}" ]; then
    echo "Checking out pinned commit: ${TORCH_TPU_PINNED_COMMIT}"
    as_jenkins git checkout "${TORCH_TPU_PINNED_COMMIT}"
else
    echo "No pinned commit found at ${TORCH_TPU_TEXT_FILE}. Checking out branch: ${TORCH_TPU_BRANCH}"
    as_jenkins git checkout "${TORCH_TPU_BRANCH}"
fi

as_jenkins git submodule update --init --recursive

# 8. JAX/LibTPU Dependencies (Runtime)
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

# 9. Build Dependencies
# Using the confirmed path: requirements/requirements.txt
# Filter out torch pins to prevent downgrading the CI build
grep -vE "torch|torchvision|torchaudio" requirements/requirements.txt > requirements_no_torch.txt
pip_install -r requirements_no_torch.txt
rm requirements_no_torch.txt

# 10. Build
echo "Building TorchTPU Wheel..."
export TORCH_SOURCE=$(python -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))")

as_jenkins env TORCH_SOURCE="${TORCH_SOURCE}" bazel build //ci/wheel:torch_tpu_wheel --config=local --define WHEEL_VERSION=0.1.0 --define TORCH_SOURCE=local

# 11. Install
pip_install bazel-bin/ci/wheel/*.whl

# 12. Cleanup
popd # Back to /var/lib/jenkins/workspace
echo "Cleaning up build artifacts..."
rm -rf torch_tpu

# 13. Verification
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TORCH_LIB_PATH}"
echo "Updated LD_LIBRARY_PATH to include: ${TORCH_LIB_PATH}"

# Verify installation
python -c "import torch; from torch_tpu import api; print(f'Success! Device: {api.tpu_device()}')"

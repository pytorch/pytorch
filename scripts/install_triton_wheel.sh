#!/bin/bash
# Updates Triton to the pinned version for this copy of PyTorch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ -z "${USE_XPU}" ]]; then
    # Default install from PyTorch source

    TRITON_VERSION="pytorch-triton==$(cat .ci/docker/triton_version.txt)"
    DOWNLOAD_PYTORCH_ORG="https://download.pytorch.org/whl"
    if [[ "$BRANCH" =~ .*release.* ]]; then
        pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/test/ $TRITON_VERSION
    else
        pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/nightly/ $TRITON_VERSION+$(head -c 10 .ci/docker/ci_commit_pins/triton.txt)
    fi
else
    # Always install Triton for XPU from source

    TRITON_XPU_REPO="https://github.com/intel/intel-xpu-backend-for-triton"
    TRITON_XPU_COMMIT_ID="$(cat .ci/docker/ci_commit_pins/triton-xpu.txt)"

    # force-reinstall to ensure the pinned version is installed
    pip install --force-reinstall "git+${TRITON_XPU_REPO}@${TRITON_XPU_COMMIT_ID}#subdirectory=python"
fi

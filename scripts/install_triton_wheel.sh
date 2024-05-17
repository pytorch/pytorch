#!/bin/bash
# Updates Triton to the pinned version for this copy of PyTorch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

TRITON_VERSION="pytorch-triton==$(cat .ci/docker/triton_version.txt)"
DOWNLOAD_PYTORCH_ORG="https://download.pytorch.org/whl"

if [[ -z "${USE_XPU}" ]]; then
    # Default install from PyTorch source
    if [[ "$BRANCH" =~ .*release.* ]]; then
        pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/test/ $TRITON_VERSION
    else
        pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/nightly/ $TRITON_VERSION+$(head -c 10 .ci/docker/ci_commit_pins/triton.txt)
    fi
else
    # Install Triton for XPU
    if [[ "$BRANCH" =~ .*release.* ]]; then
        # Option 1: Stable version. Use the pre-built wheel from pypi.
        TRITON_XPU_VERSION="triton-xpu==$(cat .ci/docker/triton_xpu_version.txt)"
        pip install ${TRITON_XPU_VERSION}
    else
        # Option 2: Nightly version. Always build triton-xpu from source.
        TRITON_XPU_REPO="https://github.com/intel/intel-xpu-backend-for-triton"
        TRITON_XPU_COMMIT_ID="$(cat .ci/docker/ci_commit_pins/triton-xpu.txt)"

        pip install --force-reinstall "git+${TRITON_XPU_REPO}@${TRITON_XPU_COMMIT_ID}#subdirectory=python"
    fi
fi

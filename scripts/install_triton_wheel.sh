#!/bin/bash
# Updates Triton to the pinned version for this copy of PyTorch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DOWNLOAD_PYTORCH_ORG="https://download.pytorch.org/whl"

if [[ -z "${USE_XPU}" ]]; then
    # Default install from PyTorch source

    TRITON_VERSION="pytorch-triton==$(cat .ci/docker/triton_version.txt)"
    if [[ "$BRANCH" =~ .*release.* ]]; then
        pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/test/ $TRITON_VERSION
    else
        pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/nightly/ $TRITON_VERSION+git$(head -c 8 .ci/docker/ci_commit_pins/triton.txt)
    fi
else
    # The Triton xpu logic is as follows:
    # 1. By default, install pre-built whls.
    # 2. [Not exposed to user] If the user set `TRITON_XPU_BUILD_FROM_SOURCE=1` flag,
    #    it will install Triton from the source.

    TRITON_VERSION="pytorch-triton-xpu==$(cat .ci/docker/triton_version.txt)"
    TRITON_XPU_COMMIT_ID="$(head -c 8 .ci/docker/ci_commit_pins/triton-xpu.txt)"
    if [[ -z "${TRITON_XPU_BUILD_FROM_SOURCE}" ]]; then
        pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/nightly/ ${TRITON_VERSION}+git${TRITON_XPU_COMMIT_ID}
    else
        TRITON_XPU_REPO="https://github.com/intel/intel-xpu-backend-for-triton"

        # force-reinstall to ensure the pinned version is installed
        pip install --force-reinstall "git+${TRITON_XPU_REPO}@${TRITON_XPU_COMMIT_ID}#subdirectory=python"
    fi
fi

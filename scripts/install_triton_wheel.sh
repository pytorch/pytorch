#!/bin/bash
# Updates Triton to the pinned version for this copy of PyTorch
PYTHON="python3"
PIP="$PYTHON -m pip"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DOWNLOAD_PYTORCH_ORG="https://download.pytorch.org/whl"

if [[ -n "${USE_XPU}" ]]; then
    TRITON_VERSION="triton-xpu==$(cat .ci/docker/triton_xpu_version.txt)"
    TRITON_COMMIT_ID="$(head -c 8 .ci/docker/ci_commit_pins/triton-xpu.txt)"
elif [[ "${USE_ROCM:-0}" == "1" ]]; then
    TRITON_VERSION="triton-rocm==$(cat .ci/docker/triton_version.txt)"
    TRITON_COMMIT_ID="$(head -c 8 .ci/docker/ci_commit_pins/triton-rocm.txt)"
else
    # Default CUDA install from PyTorch source
    TRITON_VERSION="triton==$(cat .ci/docker/triton_version.txt)"
    TRITON_COMMIT_ID="$(head -c 8 .ci/docker/ci_commit_pins/triton.txt)"
fi

if [[ "$BRANCH" =~ .*release.* ]]; then
    ${PIP} install --index-url ${DOWNLOAD_PYTORCH_ORG}/test/ $TRITON_VERSION
else
    ${PIP} install --index-url ${DOWNLOAD_PYTORCH_ORG}/nightly/ $TRITON_VERSION+git${TRITON_COMMIT_ID}
fi

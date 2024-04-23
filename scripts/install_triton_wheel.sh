#!/bin/bash
# Updates Triton to the pinned version for this copy of PyTorch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
TRITON_VERSION="pytorch-triton==$(cat .ci/docker/triton_version.txt)"
DOWNLOAD_PYTORCH_ORG="https://download.pytorch.org/whl"

if [[ "$BRANCH" =~ .*release.* ]]; then
    pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/test/ $TRITON_VERSION
else
    pip install --index-url ${DOWNLOAD_PYTORCH_ORG}/nightly/ $TRITON_VERSION+$(head -c 10 .ci/docker/ci_commit_pins/triton.txt)
fi

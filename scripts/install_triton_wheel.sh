#!/bin/bash
# Updates Triton to the pinned version for this copy of PyTorch
PYTHON="python3"
PIP="$PYTHON -m pip"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DOWNLOAD_PYTORCH_ORG="https://download.pytorch.org/whl"

if [[ -z "${USE_XPU}" ]]; then
    # Default install from PyTorch source
    # triton-xpu and the legacy pytorch-triton provide the same `triton` module
    ${PIP} uninstall -y triton-xpu pytorch-triton
    TRITON_PACKAGE="triton"
    TRITON_VERSION="$(cat .ci/docker/triton_version.txt)"
    TRITON_COMMIT_ID="$(head -c 8 .ci/docker/ci_commit_pins/triton.txt)"
else
    # triton-xpu provides the same `triton` module, so it can't coexist with triton
    ${PIP} uninstall -y triton
    TRITON_PACKAGE="triton-xpu"
    TRITON_VERSION="$(cat .ci/docker/triton_xpu_version.txt)"
    TRITON_COMMIT_ID="$(head -c 8 .ci/docker/ci_commit_pins/triton-xpu.txt)"
fi

if [[ "$BRANCH" =~ .*release.* ]]; then
    # Release branches take any patch release of the pinned minor, matching the
    # `~=` requirement baked into release wheels by binary_populate_env.sh. A
    # nightly (+git) build of the same minor also satisfies it, so replace it.
    if ${PYTHON} -c "import importlib.metadata as m, sys; sys.exit('+git' not in m.version('${TRITON_PACKAGE}'))" 2>/dev/null; then
        ${PIP} uninstall -y "${TRITON_PACKAGE}"
    fi
    ${PIP} install --index-url ${DOWNLOAD_PYTORCH_ORG}/test/ "${TRITON_PACKAGE}~=${TRITON_VERSION}"
else
    # Nightly stays on the exact commit-pinned build
    ${PIP} install --index-url ${DOWNLOAD_PYTORCH_ORG}/nightly/ "${TRITON_PACKAGE}==${TRITON_VERSION}+git${TRITON_COMMIT_ID}"
fi

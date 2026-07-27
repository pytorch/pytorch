#!/bin/bash
# shellcheck disable=SC1090
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

if [[ "$OS" != "windows-arm64" ]]; then
    export CUDA_VERSION="${DESIRED_CUDA/cu/}"
    export USE_SCCACHE=1
    export SCCACHE_BUCKET=ossci-compiler-cache
    export SCCACHE_IGNORE_SERVER_IO_ERROR=1
    export VC_YEAR=2022
fi

if [[ "$DESIRED_CUDA" == 'xpu' ]]; then
    export VC_YEAR=2022
    export USE_SCCACHE=0
    export XPU_VERSION=2026.0
fi

echo "Free space on filesystem before build:"
df -h

pushd "$PYTORCH_ROOT/.ci/pytorch/"
export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"

if [[ "$OS" == "windows-arm64" ]]; then
    if [[ "$PACKAGE_TYPE" == 'libtorch' ]]; then
        ./windows/arm64/build_libtorch.bat
    elif [[ "$PACKAGE_TYPE" == 'wheel' ]]; then
        ./windows/arm64/build_pytorch.bat
    fi
else
    # New Python pipeline: install the requested Python, then chain
    # build_env_setup.py -> build_install_deps.py -> build_wheel.py.
    # Mirrors the Linux split landed in gh-182409.
    case "$DESIRED_CUDA" in
        cpu)  export GPU_ARCH_TYPE=cpu  ;;
        cu*)  export GPU_ARCH_TYPE=cuda ;;
        xpu)  export GPU_ARCH_TYPE=xpu  ;;
        *)    echo "Unsupported DESIRED_CUDA=$DESIRED_CUDA" >&2; exit 1 ;;
    esac

    # Ensure VS2022 BuildTools are present before vcvarsall lookup. The
    # helper bat invokes vs2022_install.ps1 if missing and resolves
    # VS15INSTALLDIR via vswhere.
    cmd /c "windows\\internal\\vc_install_helper.bat"

    # shellcheck source=./windows/set_desired_python.sh
    source ./windows/set_desired_python.sh

    ENV_FILE="$(mktemp)"
    trap 'rm -f "$ENV_FILE"' EXIT

    # Use the interpreter set_desired_python.sh selected, not a bare `python`
    # (which resolves to the regular exe even for a free-threaded build), so
    # the pip deps land in it and the wheel's ABI tag matches DESIRED_PYTHON.
    "$DESIRED_PYTHON_EXE" ./windows/build_env_setup.py --env-out "$ENV_FILE"
    # shellcheck source=/dev/null
    source "$ENV_FILE"

    "$DESIRED_PYTHON_EXE" ./windows/build_install_deps.py --env-out "$ENV_FILE"
    # shellcheck source=/dev/null
    source "$ENV_FILE"

    cd "$PYTORCH_ROOT"
    "$DESIRED_PYTHON_EXE" "$PYTORCH_ROOT/.ci/pytorch/windows/build_wheel.py" "$PYTORCH_FINAL_PACKAGE_DIR"
fi

echo "Free space on filesystem after build:"
df -h

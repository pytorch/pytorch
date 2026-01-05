#!/usr/bin/env bash
# build FA3 wheels for multiple CUDA versions

set -ex -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PYTORCH_ROOT="${PYTORCH_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

CUDA_VERSIONS=("12.6" "13.0")

ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    export WHEEL_PLAT="x86_64"
elif [[ "$ARCH" == "aarch64" ]]; then
    export WHEEL_PLAT="aarch64"
else
    echo "warning: unknown architecture $ARCH, defaulting to x86_64"
    export WHEEL_PLAT="x86_64"
fi

get_arch_list() {
    local cuda_version=$1
    case ${cuda_version} in
        12.6)
            echo "8.0;8.6;9.0"
            ;;
        13.0)
            echo "8.0;8.6;9.0;10.0+PTX" # check this
            ;;
        *)
            echo "8.0;8.6;9.0"
            ;;
    esac
}

for CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
    echo "build for CUDA ${CUDA_VERSION}"
    CUDA_SHORT_VERSION=$(echo "$CUDA_VERSION" | tr -d '.')

    export TORCH_CUDA_ARCH_LIST=$(get_arch_list "$CUDA_VERSION")

    export FA_FINAL_PACKAGE_DIR="${FA_FINAL_PACKAGE_DIR:-${PYTORCH_ROOT}/third_party/flash-attention/hopper/dist}/cu${CUDA_SHORT_VERSION}"
    mkdir -p "$FA_FINAL_PACKAGE_DIR"

    # install pytorch
    echo "installing PyTorch for CUDA ${CUDA_SHORT_VERSION}..."
    pip install torch \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT_VERSION}/"
    echo "running build.sh..."
    bash "${SCRIPT_DIR}/build.sh"
done

echo "all builds complete"

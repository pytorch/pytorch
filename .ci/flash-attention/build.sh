#!/bin/bash

set -ex -o pipefail

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PYTORCH_ROOT="${PYTORCH_ROOT:-$(cd "$SOURCE_DIR/../.." && pwd)}"

source "${PYTORCH_ROOT}/.ci/pytorch/common_utils.sh"
FLASH_ATTENTION_DIR="${PYTORCH_ROOT}/third_party/flash-attention"
FLASH_ATTENTION_HOPPER_DIR="${FLASH_ATTENTION_DIR}/hopper"

if [[ -z "$FA_FINAL_PACKAGE_DIR" ]]; then
    fatal "FA_FINAL_PACKAGE_DIR must be set"
fi

if [[ -z "$WHEEL_PLAT" ]]; then
    fatal "WHEEL_PLAT must be set"
fi

if [[ ! -d "$FLASH_ATTENTION_HOPPER_DIR" ]]; then
    fatal "flash attn directory not found $FLASH_ATTENTION_HOPPER_DIR"
fi

PYTHON="${PYTHON_EXECUTABLE:-python}"

if [[ ! -d "/usr/local/cuda" ]]; then
    echo "CUDA not found, installing CUDA toolkit for Flash Attention build"

    if command -v dnf &> /dev/null; then
        dnf install -y \
            wget \
            perl \
            make \
            xz \
            bzip2 \
            gcc-toolset-13-gcc \
            gcc-toolset-13-gcc-c++ \
            || true
        if [[ -d "/opt/rh/gcc-toolset-13" ]]; then
            export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH
            export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:/opt/rh/gcc-toolset-13/root/usr/lib:${LD_LIBRARY_PATH:-}
        fi
    fi

    if [[ "$(uname -m)" == "aarch64" && "${CUDA_VERSION:-12.6}" == "13.0" ]]; then
        echo "installing CUDA 13.0.0 for ARM"
        source "${PYTORCH_ROOT}/.ci/docker/common/install_cuda.sh"
        install_cuda 13.0.0 cuda_13.0.0_580.65.06_linux
    else
        CUDA_INSTALL_DIR=$(mktemp -d)
        cp "${PYTORCH_ROOT}/.ci/docker/common/install_cuda.sh" "${CUDA_INSTALL_DIR}/"
        cp "${PYTORCH_ROOT}/.ci/docker/common/install_nccl.sh" "${CUDA_INSTALL_DIR}/"
        cp "${PYTORCH_ROOT}/.ci/docker/common/install_cusparselt.sh" "${CUDA_INSTALL_DIR}/"
        mkdir -p "${CUDA_INSTALL_DIR}/ci_commit_pins"
        cp "${PYTORCH_ROOT}/.ci/docker/ci_commit_pins"/nccl* "${CUDA_INSTALL_DIR}/ci_commit_pins/" 2>/dev/null || true

        pushd "${CUDA_INSTALL_DIR}"
        bash ./install_cuda.sh "${CUDA_VERSION:-12.6}"
        popd
        rm -rf "${CUDA_INSTALL_DIR}"
    fi

    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH

    echo "=== CUDA installation check ==="
    ls -la /usr/local/cuda/include/cuda_runtime.h || echo "WARNING: cuda_runtime.h not found"
    ls -la /usr/local/cuda/include/cuda/std/utility || echo "WARNING: libcu++ headers not found"

    echo "Installed CUDA version:"
    nvcc --version
fi

echo "installing dependencies"
"$PYTHON" -m pip install einops packaging ninja numpy wheel setuptools

export PATH="$(dirname "$PYTHON"):$PATH"
echo "ninja location: $(which ninja)"

export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"

export FLASH_ATTENTION_DISABLE_SPLIT="${FLASH_ATTENTION_DISABLE_SPLIT:-FALSE}"
export FLASH_ATTENTION_DISABLE_PAGEDKV="${FLASH_ATTENTION_DISABLE_PAGEDKV:-FALSE}"
export FLASH_ATTENTION_DISABLE_APPENDKV="${FLASH_ATTENTION_DISABLE_APPENDKV:-FALSE}"
export FLASH_ATTENTION_DISABLE_LOCAL="${FLASH_ATTENTION_DISABLE_LOCAL:-FALSE}"
export FLASH_ATTENTION_DISABLE_SOFTCAP="${FLASH_ATTENTION_DISABLE_SOFTCAP:-FALSE}"
export FLASH_ATTENTION_DISABLE_PACKGQA="${FLASH_ATTENTION_DISABLE_PACKGQA:-FALSE}"
export FLASH_ATTENTION_DISABLE_FP16="${FLASH_ATTENTION_DISABLE_FP16:-FALSE}"
export FLASH_ATTENTION_DISABLE_FP8="${FLASH_ATTENTION_DISABLE_FP8:-FALSE}"
export FLASH_ATTENTION_DISABLE_VARLEN="${FLASH_ATTENTION_DISABLE_VARLEN:-FALSE}"
export FLASH_ATTENTION_DISABLE_CLUSTER="${FLASH_ATTENTION_DISABLE_CLUSTER:-FALSE}"
export FLASH_ATTENTION_DISABLE_HDIM64="${FLASH_ATTENTION_DISABLE_HDIM64:-FALSE}"
export FLASH_ATTENTION_DISABLE_HDIM96="${FLASH_ATTENTION_DISABLE_HDIM96:-FALSE}"
export FLASH_ATTENTION_DISABLE_HDIM128="${FLASH_ATTENTION_DISABLE_HDIM128:-FALSE}"
export FLASH_ATTENTION_DISABLE_HDIM192="${FLASH_ATTENTION_DISABLE_HDIM192:-FALSE}"
export FLASH_ATTENTION_DISABLE_HDIM256="${FLASH_ATTENTION_DISABLE_HDIM256:-FALSE}"
export FLASH_ATTENTION_DISABLE_SM80="${FLASH_ATTENTION_DISABLE_SM80:-FALSE}"
export FLASH_ATTENTION_ENABLE_VCOLMAJOR="${FLASH_ATTENTION_ENABLE_VCOLMAJOR:-FALSE}"
export FLASH_ATTENTION_DISABLE_HDIMDIFF64="${FLASH_ATTENTION_DISABLE_HDIMDIFF64:-FALSE}"
export FLASH_ATTENTION_DISABLE_HDIMDIFF192="${FLASH_ATTENTION_DISABLE_HDIMDIFF192:-FALSE}"

export NVCC_THREADS="${NVCC_THREADS:-8}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

echo "NVCC_THREADS=${NVCC_THREADS}"
echo "MAX_JOBS=${MAX_JOBS}"

pushd "$FLASH_ATTENTION_HOPPER_DIR"

git config --global --add safe.directory '*'
git submodule update --init ../csrc/cutlass

if [[ "$(uname -m)" != "aarch64" ]]; then
    sed -i 's/bare_metal_version != Version("12.8")/bare_metal_version < Version("12.8")/' \
        "$FLASH_ATTENTION_HOPPER_DIR/setup.py"
fi

"$PYTHON" setup.py bdist_wheel \
    -d "$FA_FINAL_PACKAGE_DIR" \
    -k \
    --plat-name "manylinux_2_28_${WHEEL_PLAT}"

echo "wheel built: "
find "$FA_FINAL_PACKAGE_DIR" -name '*.whl' -exec ls -la {} \;

popd

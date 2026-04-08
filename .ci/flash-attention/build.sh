#!/bin/bash

set -ex -o pipefail

PYTORCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${PYTORCH_ROOT}/.ci/pytorch/common_utils.sh"
FLASH_ATTENTION_DIR="${PYTORCH_ROOT}/third_party/flash-attention"
FLASH_ATTENTION_HOPPER_DIR="${FLASH_ATTENTION_DIR}/hopper"

[[ -z "$FA_FINAL_PACKAGE_DIR" ]] && fatal "FA_FINAL_PACKAGE_DIR must be set"
[[ -z "$MANYLINUX_PLAT" ]] && fatal "MANYLINUX_PLAT must be set"
[[ -z "$CUDA_VERSION" ]] && fatal "CUDA_VERSION must be set"
[[ -z "$CUDA_SHORT" ]] && fatal "CUDA_SHORT must be set"
[[ -z "$PYTHON_VERSION" ]] && fatal "PYTHON_VERSION must be set"
[[ ! -d "$FLASH_ATTENTION_HOPPER_DIR" ]] && fatal "flash attn directory not found $FLASH_ATTENTION_HOPPER_DIR"

TORCH_MIN_VERSION="2.9.0"
PYTHON="${PYTHON_EXECUTABLE:-python}"

# for ARM builds we need GLIBC 2.29+ so we use upstream linux image
# need to install dependencies
if [[ "$(uname -m)" == "aarch64" ]]; then
    if command -v dnf &> /dev/null; then
        dnf install -y \
            wget \
            perl \
            make \
            xz \
            bzip2 \
            gcc-toolset-13-gcc \
            gcc-toolset-13-gcc-c++
        export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH
        export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:/opt/rh/gcc-toolset-13/root/usr/lib:${LD_LIBRARY_PATH:-}
    fi

    source "${PYTORCH_ROOT}/.ci/docker/common/install_cuda.sh"
    [[ -z "$CUDA_INSTALLER_NAME" ]] && fatal "CUDA_INSTALLER_NAME must be set for aarch64 builds"
    install_cuda "$CUDA_VERSION" "$CUDA_INSTALLER_NAME"

    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH

    echo "installed CUDA version:"
    nvcc --version
fi

echo "installing dependencies"
"$PYTHON" -m pip install einops packaging ninja numpy wheel setuptools

export PATH="$(dirname "$PYTHON"):$PATH"

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

if [[ "${CUDA_VERSION}" == 13.* ]]; then
    CCCL_INCLUDE="/usr/local/cuda/include/cccl"
    [[ ! -d "${CCCL_INCLUDE}" ]] && fatal "CCCL include directory not found at ${CCCL_INCLUDE}"
    echo "Adding CCCL include path: ${CCCL_INCLUDE}"
    export CPLUS_INCLUDE_PATH="${CCCL_INCLUDE}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="${CCCL_INCLUDE}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi

if [[ "${FA_TEST_BUILD}" == "true" ]]; then
    BUILD_DATE=$(date +%Y%m%d)
    export FLASH_ATTN_LOCAL_VERSION="${BUILD_DATE}.cu${CUDA_SHORT}"
fi

# stable ABI wheel requires torch>=2.9.0
# since Python 3.9 support was dropped in torch 2.9.0, we need to use Python 3.10+
sed -i "s/python_requires=\">=3.8\"/python_requires=\">=${PYTHON_VERSION}\"/" setup.py
sed -i "s/\"torch\",/\"torch>=${TORCH_MIN_VERSION}\",/" setup.py

"$PYTHON" setup.py bdist_wheel \
    -d "$FA_FINAL_PACKAGE_DIR" \
    -k \
    --plat-name "${MANYLINUX_PLAT}"

echo "wheel built: "
find "$FA_FINAL_PACKAGE_DIR" -name '*.whl' -exec ls -la {} \;

popd

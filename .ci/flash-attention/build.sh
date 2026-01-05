#!/bin/bash
# Build FA3 wheel for Linux
#
#
# Optional environment variables:
#   TORCH_CUDA_ARCH_LIST  - GPU architectures to compile for (default: "8.0+PTX 8.6 9.0a")
#   FA_FINAL_PACKAGE_DIR  - Output directory for wheels (default: hopper/dist)
#   FLASH_ATTENTION_DISABLE_*  - Feature flags to disable features
#   NVCC_THREADS          - Parallel NVCC threads (default: 4)
#   MAX_JOBS              - Max parallel jobs (default: nproc)

set -ex -o pipefail

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PYTORCH_ROOT="${PYTORCH_ROOT:-$(cd "$SOURCE_DIR/../.." && pwd)}"


source "${PYTORCH_ROOT}/.ci/pytorch/common_utils.sh"
FLASH_ATTENTION_DIR="${PYTORCH_ROOT}/third_party/flash-attention"
FLASH_ATTENTION_HOPPER_DIR="${FLASH_ATTENTION_DIR}/hopper"

if [[ -z "$FA_FINAL_PACKAGE_DIR" ]]; then
    FA_FINAL_PACKAGE_DIR="${FLASH_ATTENTION_HOPPER_DIR}/dist"
fi
mkdir -p "$FA_FINAL_PACKAGE_DIR" || true


ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    PIP_PLATFORM="x86_64"
elif [[ "$ARCH" == "aarch64" ]]; then
    PIP_PLATFORM="aarch64"
else
    echo "Warning: Unknown architecture $ARCH, defaulting to x86_64"
    PIP_PLATFORM="x86_64"
fi

if [[ ! -d "$FLASH_ATTENTION_HOPPER_DIR" ]]; then
    fatal "Flash Attention hopper directory not found: $FLASH_ATTENTION_HOPPER_DIR"
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0+PTX 8.6 9.0a}"

echo "Installing build dependencies..."
pip_install einops packaging ninja numpy

export TORCH_CUDA_ARCH_LIST
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"

export FLASH_ATTENTION_DISABLE_BACKWARD="${FLASH_ATTENTION_DISABLE_BACKWARD:-FALSE}"
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
export FLASH_ATTENTION_DISABLE_SM80="${FLASH_ATTENTION_DISABLE_SM80:-FALSE}"

export NVCC_THREADS="${NVCC_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"


pushd "$FLASH_ATTENTION_HOPPER_DIR"

git submodule update --init ../csrc/cutlass

# build wheel
python setup.py bdist_wheel \
    -d "$FA_FINAL_PACKAGE_DIR" \
    -k \
    --plat-name "manylinux_2_28_${PIP_PLATFORM}"

echo "Wheel(s) built:"
find "$FA_FINAL_PACKAGE_DIR" -name '*.whl' -exec ls -la {} \;

popd

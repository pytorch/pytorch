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

    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH

    echo "=== CUDA installation check ==="
    ls -la /usr/local/cuda/include/cuda_runtime.h || echo "WARNING: cuda_runtime.h not found"
    ls -la /usr/local/cuda/include/cuda/std/utility || echo "WARNING: libcu++ headers not found"

    if [[ ! -f "/usr/local/cuda/include/cuda/std/utility" ]]; then
        echo "libcu++ headers missing, downloading cuda_cccl package..."

        if [[ "$(uname -m)" == "aarch64" ]]; then
            REDIST_ARCH="linux-sbsa"
        else
            REDIST_ARCH="linux-x86_64"
        fi

        CUDA_MAJOR_MINOR="${CUDA_VERSION:-12.6}"
        case "${CUDA_MAJOR_MINOR}" in
            13.0|13.0.*)
                CCCL_VERSION="13.0.85"
                ;;
            12.6|12.6.*)
                CCCL_VERSION="12.6.77"
                ;;
            *)
                echo "Unknown CUDA version for CCCL: ${CUDA_MAJOR_MINOR}"
                CCCL_VERSION=""
                ;;
        esac

        if [[ -n "${CCCL_VERSION}" ]]; then
            CCCL_PKG="cuda_cccl-${REDIST_ARCH}-${CCCL_VERSION}-archive"
            CCCL_URL="https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/${REDIST_ARCH}/${CCCL_PKG}.tar.xz"
            echo "Downloading CCCL from: ${CCCL_URL}"

            CCCL_TMP=$(mktemp -d)
            pushd "${CCCL_TMP}"
            wget -q "${CCCL_URL}" -O cccl.tar.xz
            tar xf cccl.tar.xz

            echo "=== CCCL package contents ==="
            find "${CCCL_PKG}" -type d -name "cuda" | head -20
            find "${CCCL_PKG}" -name "utility" | head -10

            # Copy all include directories
            if [[ -d "${CCCL_PKG}/include" ]]; then
                cp -a "${CCCL_PKG}"/include/* /usr/local/cuda/include/
            fi

            # CCCL 2.x may have libcudacxx headers under lib/cmake or different location
            # Check for cuda/std in various locations and copy if found
            for search_dir in "${CCCL_PKG}/lib" "${CCCL_PKG}"; do
                if [[ -d "${search_dir}" ]]; then
                    cuda_std_dir=$(find "${search_dir}" -type d -path "*/cuda/std" 2>/dev/null | head -1)
                    if [[ -n "${cuda_std_dir}" ]]; then
                        # Found cuda/std, copy the parent cuda directory
                        cuda_dir=$(dirname "${cuda_std_dir}")
                        echo "Found libcu++ at: ${cuda_dir}"
                        mkdir -p /usr/local/cuda/include/cuda
                        cp -a "${cuda_dir}"/* /usr/local/cuda/include/cuda/
                        break
                    fi
                fi
            done

            popd
            rm -rf "${CCCL_TMP}"

            echo "CCCL installed, verifying..."
            ls -la /usr/local/cuda/include/cuda/std/utility || echo "WARNING: libcu++ still not found after CCCL install"
        fi
    fi

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

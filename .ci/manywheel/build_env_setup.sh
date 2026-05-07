#!/usr/bin/env bash
# GPU environment setup (runs once before any wheel is built).
#
# When running on a standard manylinux container (no pre-installed GPU
# toolkits), this script installs CUDA, cuDNN, NCCL, MAGMA, cuSPARSELt,
# etc. from the existing .ci/docker/common/ install scripts.
#
# Environment variables expected:
#   GPU_ARCH_TYPE    - cpu, cuda, rocm, xpu
#   DESIRED_CUDA     - cpu, cu126, cu128, rocm7.1, xpu, etc.
#   GPU_ARCH_VERSION - 12.6, 12.8, 7.1, etc.

set -ex

ARCH=$(uname -m)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${GITHUB_WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
echo "Architecture: $ARCH, GPU_ARCH_TYPE: ${GPU_ARCH_TYPE:-unset}, REPO_ROOT: $REPO_ROOT"

# Install OS packages
OS_NAME=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
if [[ "$OS_NAME" == *"AlmaLinux"* ]] || [[ "$OS_NAME" == *"CentOS"* ]] || [[ "$OS_NAME" == *"Red Hat"* ]]; then
    yum install -q -y zip openssl openssl-devel sudo wget curl perl util-linux xz bzip2 git patch which zlib-devel
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    sed -i 's/.*nvidia.*/# &/' $(find /etc/apt/ -type f -name "*.list") 2>/dev/null || true
    apt-get update -qq
    apt-get -y -qq install zip openssl wget curl git
fi

# Set platform tag
case $ARCH in
    x86_64)  export PLATFORM="manylinux_2_28_x86_64" ;;
    aarch64) export PLATFORM="manylinux_2_28_aarch64" ;;
esac

# Ensure a Python with pip is on PATH (the system /usr/bin/python3 in
# manylinux images has no pip; use any /opt/python/ build).
if ! python3 -mpip --version &>/dev/null; then
    FALLBACK_PYTHON=$(ls -d /opt/python/cp3*/bin 2>/dev/null | sort -V | tail -1)
    if [[ -n "$FALLBACK_PYTHON" ]]; then
        export PATH="$FALLBACK_PYTHON:$PATH"
        echo "Added $FALLBACK_PYTHON to PATH for pip access"
    fi
fi

# Install MKL (BLAS) if not already present.
# MKL is x86_64 only; aarch64 uses OpenBLAS/ACL from the builder image.
if [[ "$ARCH" == "x86_64" && ! -d /opt/intel/lib ]]; then
    echo "MKL not found, installing..."
    bash "$REPO_ROOT/.ci/docker/common/install_mkl.sh"
fi

install_cuda_toolkit() {
    # Install CUDA toolkit, cuDNN, NCCL, cuSPARSELt, nvSHMEM using the
    # existing Docker build scripts. These scripts expect a flat directory
    # layout with install_nccl.sh, install_cusparselt.sh, and ci_commit_pins/
    # as siblings — matching the Docker build context from Dockerfile_2_28.
    local cuda_version="$1"
    local stage_dir
    stage_dir=$(mktemp -d)

    cp "$REPO_ROOT/.ci/docker/common/install_cuda.sh" "$stage_dir/"
    cp "$REPO_ROOT/.ci/docker/common/install_nccl.sh" "$stage_dir/"
    cp "$REPO_ROOT/.ci/docker/common/install_cusparselt.sh" "$stage_dir/"
    mkdir -p "$stage_dir/ci_commit_pins"
    cp "$REPO_ROOT/.ci/docker/ci_commit_pins"/nccl* "$stage_dir/ci_commit_pins/"

    pushd "$stage_dir"
    bash install_cuda.sh "$cuda_version"
    popd
    rm -rf "$stage_dir"

    # Install MAGMA (downloads pre-built tarball from S3)
    bash "$REPO_ROOT/.ci/docker/common/install_magma.sh" "$cuda_version"

    echo "CUDA $cuda_version toolkit installation complete"
}

# GPU-specific setup
case "${GPU_ARCH_TYPE:-cpu}" in
    cuda|cuda-aarch64)
        # Determine CUDA version from DESIRED_CUDA or GPU_ARCH_VERSION
        if [[ -n "${GPU_ARCH_VERSION:-}" ]]; then
            CUDA_VERSION="${GPU_ARCH_VERSION}"
        elif [[ "${DESIRED_CUDA}" =~ ^[0-9]+\.[0-9]+$ ]]; then
            CUDA_VERSION="${DESIRED_CUDA}"
        elif [[ ${#DESIRED_CUDA} -eq 5 ]]; then
            CUDA_VERSION="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4:1}"
        fi

        # Install CUDA if not already present (standard manylinux image)
        if [[ ! -d "/usr/local/cuda-${CUDA_VERSION}" ]]; then
            echo "CUDA ${CUDA_VERSION} not found, installing from scratch..."
            install_cuda_toolkit "${CUDA_VERSION}"
        else
            echo "CUDA ${CUDA_VERSION} already installed, switching symlinks..."
            rm -rf /usr/local/cuda || true
            ln -s "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda
        fi

        # Install MAGMA if not already present (test-infra images don't include it)
        if [[ "$ARCH" != "aarch64" && ! -d "/usr/local/cuda-${CUDA_VERSION}/magma" ]]; then
            echo "MAGMA not found, installing..."
            bash "$REPO_ROOT/.ci/docker/common/install_magma.sh" "${CUDA_VERSION}"
        fi

        # MAGMA symlink (x86_64 only)
        if [[ "$ARCH" != "aarch64" ]]; then
            rm -rf /usr/local/magma || true
            ln -s "/usr/local/cuda-${CUDA_VERSION}/magma" /usr/local/magma
        fi

        # Ensure cuDNN unversioned symlinks exist (some image builders
        # copy individual files and drop the libcudnn.so -> libcudnn.so.9
        # symlink that CMake's find_library needs).
        CUDA_LIB_DIR="/usr/local/cuda/lib64"
        if [[ -d "$CUDA_LIB_DIR" ]]; then
            for sofile in "$CUDA_LIB_DIR"/libcudnn*.so.[0-9]*; do
                # e.g. libcudnn.so.9 -> create libcudnn.so
                base_link="${sofile%%.[0-9]*}.so"
                if [[ ! -e "$base_link" ]]; then
                    echo "Creating missing symlink: $base_link -> $(basename $sofile)"
                    ln -s "$(basename $sofile)" "$base_link"
                fi
            done
        fi

        # Verify cuDNN is discoverable
        echo "cuDNN check:"
        ls -la /usr/local/cuda/lib64/libcudnn*.so* 2>/dev/null || echo "WARNING: cuDNN libraries not found in /usr/local/cuda/lib64/"
        ls /usr/local/cuda/include/cudnn*.h 2>/dev/null || echo "WARNING: cuDNN headers not found in /usr/local/cuda/include/"

        # CUDA-specific environment variables
        export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
        export NCCL_ROOT_DIR=/usr/local/cuda
        export CUDNN_ROOT_DIR=/usr/local/cuda
        export TH_BINARY_BUILD=1
        export USE_STATIC_CUDNN=1
        export USE_STATIC_NCCL=1
        export ATEN_STATIC_CUDA=1
        export USE_CUDA_STATIC_LINK=1
        export USE_CUPTI_SO=0
        export USE_CUSPARSELT=1
        export USE_CUFILE=1
        export USE_SYSTEM_NCCL=1
        export NCCL_INCLUDE_DIR="/usr/local/cuda/include/"
        export NCCL_LIB_DIR="/usr/local/cuda/lib64/"

        echo "CUDA ${CUDA_VERSION} environment configured"
        ;;

    # ROCm and XPU use the legacy Docker-in-Docker workflow, not this script.

    cpu|cpu-aarch64|cpu-s390x|cpu-cxx11-abi)
        export TH_BINARY_BUILD=1
        export USE_CUDA=0
        rm -rf /usr/local/cuda* 2>/dev/null || true
        echo "CPU environment configured"
        ;;
esac

echo "before-all setup complete"

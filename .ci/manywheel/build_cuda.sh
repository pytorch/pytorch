#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P ))"

export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export NCCL_ROOT_DIR=/usr/local/cuda
export TH_BINARY_BUILD=1
export USE_STATIC_CUDNN=1
export USE_STATIC_NCCL=1
export ATEN_STATIC_CUDA=1
export USE_CUDA_STATIC_LINK=1
export INSTALL_TEST=0 # dont install test binaries into site-packages
export USE_CUPTI_SO=0
export USE_CUSPARSELT=${USE_CUSPARSELT:-1} # Enable if not disabled by libtorch build
export USE_CUFILE=${USE_CUFILE:-1}
export USE_SYSTEM_NCCL=1
export NCCL_INCLUDE_DIR="/usr/local/cuda/include/"
export NCCL_LIB_DIR="/usr/local/cuda/lib64/"

# Keep an array of cmake variables to add to
if [[ -z "$CMAKE_ARGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build()
    CMAKE_ARGS=()
fi
if [[ -z "$EXTRA_CAFFE2_CMAKE_FLAGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build_caffe2()
    EXTRA_CAFFE2_CMAKE_FLAGS=()
fi

# Detect architecture
ARCH=$(uname -m)
echo "Building for architecture: $ARCH"

# Determine CUDA version and architectures to build for
#
# NOTE: We should first check `DESIRED_CUDA` when determining `CUDA_VERSION`,
# because in some cases a single Docker image can have multiple CUDA versions
# on it, and `nvcc --version` might not show the CUDA version we want.
if [[ -n "$DESIRED_CUDA" ]]; then
    # If the DESIRED_CUDA already matches the format that we expect
    if [[ ${DESIRED_CUDA} =~ ^[0-9]+\.[0-9]+$ ]]; then
        CUDA_VERSION=${DESIRED_CUDA}
    else
        # cu126, cu128 etc...
        if [[ ${#DESIRED_CUDA} -eq 5 ]]; then
            CUDA_VERSION="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4:1}"
        fi
    fi
    echo "Using CUDA $CUDA_VERSION as determined by DESIRED_CUDA"
else
    CUDA_VERSION=$(nvcc --version|grep release|cut -f5 -d" "|cut -f1 -d",")
    echo "CUDA $CUDA_VERSION Detected"
fi

cuda_version_nodot=$(echo $CUDA_VERSION | tr -d '.')
EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")

# Function to remove architectures from a list
remove_archs() {
    local result="$1"
    shift
    for arch in "$@"; do
        result="${result//${arch};/}"
    done
    echo "$result"
}

# Function to filter CUDA architectures for aarch64
# aarch64 ARM GPUs only support certain compute capabilities
# Keep: 8.0 (A100), 9.0+ (Hopper, Grace Hopper, newer)
# Remove: < 8.0 (no ARM GPUs), 8.6 (x86_64 RTX 3090/A6000 only)
filter_aarch64_archs() {
    local arch_list="$1"
    # Explicitly remove architectures not needed on aarch64
    arch_list=$(remove_archs "$arch_list" "5.0" "6.0" "7.0" "7.5" "8.6")
    echo "$arch_list"
}

# Base: Common architectures across all modern CUDA versions
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

case ${CUDA_VERSION} in
    12.6) TORCH_CUDA_ARCH_LIST="5.0;6.0;${TORCH_CUDA_ARCH_LIST}" ;;  # Only 12.6 includes Legacy Maxwell/Pascal that will be removed in future releases
    12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};10.0;12.0" ;;  # +Hopper/Blackwell support
    12.9) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};10.0;12.0+PTX" # +Hopper/Blackwell support + PTX for forward compatibility
        if [[ "$PACKAGE_TYPE" == "libtorch" ]]; then
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//7.0;/}"  # Remove 7.0 to resolve the ld error
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//8.6;/}"  # Remove 8.6 for libtorch
        fi
        ;;
    13.0)
        TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;$([[ "$ARCH" == "aarch64" ]] && echo "11.0;" || echo "")12.0+PTX"
        export TORCH_NVCC_FLAGS="-compress-mode=size"
        export BUILD_BUNDLE_PTXAS=1
        ;;
    *) echo "unknown cuda version $CUDA_VERSION"; exit 1 ;;
esac

# Filter for aarch64: Remove < 8.0 and 8.6
[[ "$ARCH" == "aarch64" ]] && TORCH_CUDA_ARCH_LIST=$(filter_aarch64_archs "$TORCH_CUDA_ARCH_LIST")

echo "TORCH_CUDA_ARCH_LIST set to: $TORCH_CUDA_ARCH_LIST"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
echo "${TORCH_CUDA_ARCH_LIST}"

# Disable MAGMA for aarch64 as pre-built libraries are x86-64 only
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Disabling MAGMA for aarch64 architecture"
    export USE_MAGMA=0
fi

# Package directories
WHEELHOUSE_DIR="wheelhouse$cuda_version_nodot"
LIBTORCH_HOUSE_DIR="libtorch_house$cuda_version_nodot"
if [[ -z "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    if [[ -z "$BUILD_PYTHONLESS" ]]; then
        PYTORCH_FINAL_PACKAGE_DIR="/remote/wheelhouse$cuda_version_nodot"
    else
        PYTORCH_FINAL_PACKAGE_DIR="/remote/libtorch_house$cuda_version_nodot"
    fi
fi
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true

OS_NAME=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
if [[ "$OS_NAME" == *"AlmaLinux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Red Hat Enterprise Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
else
    echo "Unknown OS: '$OS_NAME'"
    exit 1
fi

DEPS_LIST=(
    "$LIBGOMP_PATH"
)
DEPS_SONAME=(
    "libgomp.so.1"
)


# CUDA_VERSION 12.*, 13.*
if [[ $CUDA_VERSION == 12* || $CUDA_VERSION == 13* ]]; then
    export USE_STATIC_CUDNN=0
    # Try parallelizing nvcc as well
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all --threads 2"
    # Compress the fatbin with -compress-mode=size for CUDA 13
    if [[ $CUDA_VERSION == 13* ]]; then
        export TORCH_NVCC_FLAGS="$TORCH_NVCC_FLAGS -compress-mode=size"
    fi
    if [[ -z "$PYTORCH_EXTRA_INSTALL_REQUIREMENTS" ]]; then
        echo "Bundling with cudnn and cublas."

        DEPS_LIST+=(
            "/usr/local/cuda/lib64/libcudnn_adv.so.9"
            "/usr/local/cuda/lib64/libcudnn_cnn.so.9"
            "/usr/local/cuda/lib64/libcudnn_graph.so.9"
            "/usr/local/cuda/lib64/libcudnn_ops.so.9"
            "/usr/local/cuda/lib64/libcudnn_engines_runtime_compiled.so.9"
            "/usr/local/cuda/lib64/libcudnn_engines_precompiled.so.9"
            "/usr/local/cuda/lib64/libcudnn_heuristic.so.9"
            "/usr/local/cuda/lib64/libcudnn.so.9"
            "/usr/local/cuda/lib64/libcusparseLt.so.0"
            "/usr/local/cuda/lib64/libnvrtc-builtins.so"
            "/usr/local/cuda/lib64/libcufile.so.0"
            "/usr/local/cuda/lib64/libcufile_rdma.so.1"
            "/usr/local/cuda/lib64/libnvshmem_host.so.3"
            "/usr/local/cuda/extras/CUPTI/lib64/libnvperf_host.so"
        )
        DEPS_SONAME+=(
            "libcudnn_adv.so.9"
            "libcudnn_cnn.so.9"
            "libcudnn_graph.so.9"
            "libcudnn_ops.so.9"
            "libcudnn_engines_runtime_compiled.so.9"
            "libcudnn_engines_precompiled.so.9"
            "libcudnn_heuristic.so.9"
            "libcudnn.so.9"
            "libcusparseLt.so.0"
            "libnvrtc-builtins.so"
            "libnvshmem_host.so.3"
            "libcufile.so.0"
            "libcufile_rdma.so.1"
            "libnvperf_host.so"
        )
        # Add libnvToolsExt only if CUDA version is not 12.9
        if [[ $CUDA_VERSION == 13* ]]; then
            DEPS_LIST+=(
                "/usr/local/cuda/lib64/libcublas.so.13"
                "/usr/local/cuda/lib64/libcublasLt.so.13"
                "/usr/local/cuda/lib64/libcudart.so.13"
                "/usr/local/cuda/lib64/libnvrtc.so.13"
                "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.13"
                "/usr/local/cuda/lib64/libibverbs.so.1"
                "/usr/local/cuda/lib64/librdmacm.so.1"
                "/usr/local/cuda/lib64/libmlx5.so.1"
                "/usr/local/cuda/lib64/libnl-3.so.200"
                "/usr/local/cuda/lib64/libnl-route-3.so.200")
            DEPS_SONAME+=(
                "libcublas.so.13"
                "libcublasLt.so.13"
                "libcudart.so.13"
                "libnvrtc.so.13"
                "libcupti.so.13"
                "libibverbs.so.1"
                "librdmacm.so.1"
                "libmlx5.so.1"
                "libnl-3.so.200"
                "libnl-route-3.so.200")
            export USE_CUPTI_SO=1
            export ATEN_STATIC_CUDA=0
            export USE_CUDA_STATIC_LINK=0
            export USE_CUFILE=0
        else
            DEPS_LIST+=(
                "/usr/local/cuda/lib64/libcublas.so.12"
                "/usr/local/cuda/lib64/libcublasLt.so.12"
                "/usr/local/cuda/lib64/libcudart.so.12"
                "/usr/local/cuda/lib64/libnvrtc.so.12"
                "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12")
            DEPS_SONAME+=(
                "libcublas.so.12"
                "libcublasLt.so.12"
                "libcudart.so.12"
                "libnvrtc.so.12"
                "libcupti.so.12")

            if [[ $CUDA_VERSION != 12.9* ]]; then
                DEPS_LIST+=("/usr/local/cuda/lib64/libnvToolsExt.so.1")
                DEPS_SONAME+=("libnvToolsExt.so.1")
            fi
        fi
    else
        echo "Using nvidia libs from pypi."
        CUDA_RPATHS=(
            '$ORIGIN/../../nvidia/cudnn/lib'
            '$ORIGIN/../../nvidia/nvshmem/lib'
            '$ORIGIN/../../nvidia/nccl/lib'
            '$ORIGIN/../../nvidia/cusparselt/lib'
        )
        if [[ $CUDA_VERSION == 13* ]]; then
            CUDA_RPATHS+=('$ORIGIN/../../nvidia/cu13/lib')
        else
            CUDA_RPATHS+=(
                '$ORIGIN/../../nvidia/cublas/lib'
                '$ORIGIN/../../nvidia/cuda_cupti/lib'
                '$ORIGIN/../../nvidia/cuda_nvrtc/lib'
                '$ORIGIN/../../nvidia/cuda_runtime/lib'
                '$ORIGIN/../../nvidia/cufft/lib'
                '$ORIGIN/../../nvidia/curand/lib'
                '$ORIGIN/../../nvidia/cusolver/lib'
                '$ORIGIN/../../nvidia/cusparse/lib'
                '$ORIGIN/../../cusparselt/lib'
                '$ORIGIN/../../nvidia/nvtx/lib'
                '$ORIGIN/../../nvidia/cufile/lib'
            )
        fi

        CUDA_RPATHS=$(IFS=: ; echo "${CUDA_RPATHS[*]}")
        export C_SO_RPATH=$CUDA_RPATHS':$ORIGIN:$ORIGIN/lib'
        export LIB_SO_RPATH=$CUDA_RPATHS':$ORIGIN'
        export FORCE_RPATH="--force-rpath"
        export USE_STATIC_NCCL=0
        export ATEN_STATIC_CUDA=0
        export USE_CUDA_STATIC_LINK=0
        export USE_CUPTI_SO=1
    fi
else
    echo "Unknown cuda version $CUDA_VERSION"
    exit 1
fi

# Add ARM-specific library dependencies
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Adding ARM-specific library dependencies"

    # ARM Compute Library (if available)
    if [[ -d "/acl/build" ]]; then
        echo "Adding ARM Compute Library"
        DEPS_LIST+=(
            "/acl/build/libarm_compute.so"
            "/acl/build/libarm_compute_graph.so"
        )
        DEPS_SONAME+=(
            "libarm_compute.so"
            "libarm_compute_graph.so"
        )
    fi

    # ARM system libraries
    DEPS_LIST+=(
        "/lib64/libgomp.so.1"
        "/usr/lib64/libgfortran.so.5"
    )
    DEPS_SONAME+=(
        "libgomp.so.1"
        "libgfortran.so.5"
    )

    # NVPL libraries (ARM optimized BLAS/LAPACK)
    if [[ -d "/usr/local/lib" && -f "/usr/local/lib/libnvpl_blas_lp64_gomp.so.0" ]]; then
        echo "Adding NVPL libraries for ARM"
        DEPS_LIST+=(
            "/usr/local/lib/libnvpl_lapack_lp64_gomp.so.0"
            "/usr/local/lib/libnvpl_blas_lp64_gomp.so.0"
            "/usr/local/lib/libnvpl_lapack_core.so.0"
            "/usr/local/lib/libnvpl_blas_core.so.0"
        )
        DEPS_SONAME+=(
            "libnvpl_lapack_lp64_gomp.so.0"
            "libnvpl_blas_lp64_gomp.so.0"
            "libnvpl_lapack_core.so.0"
            "libnvpl_blas_core.so.0"
        )
    fi
fi

# run_tests.sh requires DESIRED_CUDA to know what tests to exclude
export DESIRED_CUDA="$cuda_version_nodot"

# Switch `/usr/local/cuda` to the desired CUDA version
rm -rf /usr/local/cuda || true
ln -s "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda

# Switch `/usr/local/magma` to the desired CUDA version (skip for aarch64)
if [[ "$ARCH" != "aarch64" ]]; then
    rm -rf /usr/local/magma || true
    ln -s /usr/local/cuda-${CUDA_VERSION}/magma /usr/local/magma
fi

export CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev) # 10.0.130
export CUDA_VERSION_SHORT=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev | cut -f1,2 -d".") # 10.0
export CUDNN_VERSION=$(ls /usr/local/cuda/lib64/libcudnn.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev)

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
if [[ -z "$BUILD_PYTHONLESS" ]]; then
    BUILD_SCRIPT=build_common.sh
else
    BUILD_SCRIPT=build_libtorch.sh
fi
source $SCRIPTPATH/${BUILD_SCRIPT}

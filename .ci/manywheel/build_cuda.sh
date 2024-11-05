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

# Keep an array of cmake variables to add to
if [[ -z "$CMAKE_ARGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build()
    CMAKE_ARGS=()
fi
if [[ -z "$EXTRA_CAFFE2_CMAKE_FLAGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build_caffe2()
    EXTRA_CAFFE2_CMAKE_FLAGS=()
fi

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
        # cu90, cu92, cu100, cu101
        if [[ ${#DESIRED_CUDA} -eq 4 ]]; then
            CUDA_VERSION="${DESIRED_CUDA:2:1}.${DESIRED_CUDA:3:1}"
        elif [[ ${#DESIRED_CUDA} -eq 5 ]]; then
            CUDA_VERSION="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4:1}"
        fi
    fi
    echo "Using CUDA $CUDA_VERSION as determined by DESIRED_CUDA"

    # There really has to be a better way to do this - eli
    # Possibly limiting builds to specific cuda versions be delimiting images would be a choice
    if [[ "$OS_NAME" == *"Ubuntu"* ]]; then
        echo "Switching to CUDA version ${DESIRED_CUDA}"
        /builder/conda/switch_cuda_version.sh "${DESIRED_CUDA}"
    fi
else
    CUDA_VERSION=$(nvcc --version|grep release|cut -f5 -d" "|cut -f1 -d",")
    echo "CUDA $CUDA_VERSION Detected"
fi

cuda_version_nodot=$(echo $CUDA_VERSION | tr -d '.')

TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6"
case ${CUDA_VERSION} in
    12.4)
        if [[ "$GPU_ARCH_TYPE" = "cuda-aarch64" ]]; then
            TORCH_CUDA_ARCH_LIST="9.0"
        else
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};9.0+PTX"
        fi
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    12.1)
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};9.0"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    11.8)
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};3.7;9.0"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    11.[67])
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};3.7"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    *)
        echo "unknown cuda version $CUDA_VERSION"
        exit 1
        ;;
esac

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
echo "${TORCH_CUDA_ARCH_LIST}"

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
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"AlmaLinux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Red Hat Enterprise Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
fi

DEPS_LIST=(
    "$LIBGOMP_PATH"
)
DEPS_SONAME=(
    "libgomp.so.1"
)

# CUDA 11.8 have to ship the libcusparseLt.so.0 with the binary
# since nvidia-cusparselt-cu11 is not available in PYPI
if [[ $USE_CUSPARSELT == "1" && $CUDA_VERSION == "11.8" ]]; then
        DEPS_SONAME+=(
            "libcusparseLt.so.0"
        )
        DEPS_LIST+=(
            "/usr/local/cuda/lib64/libcusparseLt.so.0"
        )
fi

if [[ $CUDA_VERSION == "12.1" || $CUDA_VERSION == "12.4" ]]; then
    export USE_STATIC_CUDNN=0
    # Try parallelizing nvcc as well
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all --threads 2"

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
            "/usr/local/cuda/lib64/libcublas.so.12"
            "/usr/local/cuda/lib64/libcublasLt.so.12"
            "/usr/local/cuda/lib64/libcusparseLt.so.0"
            "/usr/local/cuda/lib64/libcudart.so.12"
            "/usr/local/cuda/lib64/libnvToolsExt.so.1"
            "/usr/local/cuda/lib64/libnvrtc.so.12"
            "/usr/local/cuda/lib64/libnvrtc-builtins.so"
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
            "libcublas.so.12"
            "libcublasLt.so.12"
            "libcusparseLt.so.0"
            "libcudart.so.12"
            "libnvToolsExt.so.1"
            "libnvrtc.so.12"
            "libnvrtc-builtins.so"
        )
    else
        echo "Using nvidia libs from pypi."
        CUDA_RPATHS=(
            '$ORIGIN/../../nvidia/cublas/lib'
            '$ORIGIN/../../nvidia/cuda_cupti/lib'
            '$ORIGIN/../../nvidia/cuda_nvrtc/lib'
            '$ORIGIN/../../nvidia/cuda_runtime/lib'
            '$ORIGIN/../../nvidia/cudnn/lib'
            '$ORIGIN/../../nvidia/cufft/lib'
            '$ORIGIN/../../nvidia/curand/lib'
            '$ORIGIN/../../nvidia/cusolver/lib'
            '$ORIGIN/../../nvidia/cusparse/lib'
            '$ORIGIN/../../cusparselt/lib'
            '$ORIGIN/../../nvidia/nccl/lib'
            '$ORIGIN/../../nvidia/nvtx/lib'
        )
        CUDA_RPATHS=$(IFS=: ; echo "${CUDA_RPATHS[*]}")
        export C_SO_RPATH=$CUDA_RPATHS':$ORIGIN:$ORIGIN/lib'
        export LIB_SO_RPATH=$CUDA_RPATHS':$ORIGIN'
        export FORCE_RPATH="--force-rpath"
        export USE_STATIC_NCCL=0
        export USE_SYSTEM_NCCL=1
        export ATEN_STATIC_CUDA=0
        export USE_CUDA_STATIC_LINK=0
        export USE_CUPTI_SO=1
        export NCCL_INCLUDE_DIR="/usr/local/cuda/include/"
        export NCCL_LIB_DIR="/usr/local/cuda/lib64/"
    fi
elif [[ $CUDA_VERSION == "11.8" ]]; then
    export USE_STATIC_CUDNN=0
    # Try parallelizing nvcc as well
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all --threads 2"
    # Bundle ptxas into the wheel, see https://github.com/pytorch/pytorch/pull/119750
    export BUILD_BUNDLE_PTXAS=1

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
            "/usr/local/cuda/lib64/libcublas.so.11"
            "/usr/local/cuda/lib64/libcublasLt.so.11"
            "/usr/local/cuda/lib64/libcudart.so.11.0"
            "/usr/local/cuda/lib64/libnvToolsExt.so.1"
            "/usr/local/cuda/lib64/libnvrtc.so.11.2"    # this is not a mistake, it links to more specific cuda version
            "/usr/local/cuda/lib64/libnvrtc-builtins.so.11.8"
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
            "libcublas.so.11"
            "libcublasLt.so.11"
            "libcudart.so.11.0"
            "libnvToolsExt.so.1"
            "libnvrtc.so.11.2"
            "libnvrtc-builtins.so.11.8"
        )
    else
        echo "Using nvidia libs from pypi."
        CUDA_RPATHS=(
            '$ORIGIN/../../nvidia/cublas/lib'
            '$ORIGIN/../../nvidia/cuda_cupti/lib'
            '$ORIGIN/../../nvidia/cuda_nvrtc/lib'
            '$ORIGIN/../../nvidia/cuda_runtime/lib'
            '$ORIGIN/../../nvidia/cudnn/lib'
            '$ORIGIN/../../nvidia/cufft/lib'
            '$ORIGIN/../../nvidia/curand/lib'
            '$ORIGIN/../../nvidia/cusolver/lib'
            '$ORIGIN/../../nvidia/cusparse/lib'
            '$ORIGIN/../../nvidia/nccl/lib'
            '$ORIGIN/../../nvidia/nvtx/lib'
        )
        CUDA_RPATHS=$(IFS=: ; echo "${CUDA_RPATHS[*]}")
        export C_SO_RPATH=$CUDA_RPATHS':$ORIGIN:$ORIGIN/lib'
        export LIB_SO_RPATH=$CUDA_RPATHS':$ORIGIN'
        export FORCE_RPATH="--force-rpath"
        export USE_STATIC_NCCL=0
        export USE_SYSTEM_NCCL=1
        export ATEN_STATIC_CUDA=0
        export USE_CUDA_STATIC_LINK=0
        export USE_CUPTI_SO=1
        export NCCL_INCLUDE_DIR="/usr/local/cuda/include/"
        export NCCL_LIB_DIR="/usr/local/cuda/lib64/"
    fi
else
    echo "Unknown cuda version $CUDA_VERSION"
    exit 1
fi

# builder/test.sh requires DESIRED_CUDA to know what tests to exclude
export DESIRED_CUDA="$cuda_version_nodot"

# Switch `/usr/local/cuda` to the desired CUDA version
rm -rf /usr/local/cuda || true
ln -s "/usr/local/cuda-${CUDA_VERSION}" /usr/local/cuda

# Switch `/usr/local/magma` to the desired CUDA version
rm -rf /usr/local/magma || true
ln -s /usr/local/cuda-${CUDA_VERSION}/magma /usr/local/magma

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

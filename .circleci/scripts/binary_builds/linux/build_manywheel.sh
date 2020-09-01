#!/usr/bin/env bash

set -ex

export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export NCCL_ROOT_DIR=/usr/local/cuda
export TH_BINARY_BUILD=1
export USE_STATIC_CUDNN=1
export USE_STATIC_NCCL=1
export ATEN_STATIC_CUDA=1
export USE_CUDA_STATIC_LINK=1
export INSTALL_TEST=0 # dont install test binaries into site-packages

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
    # cu90, cu92, cu100, cu101
    if [[ ${#DESIRED_CUDA} -eq 4 ]]; then
        CUDA_VERSION="${DESIRED_CUDA:2:1}.${DESIRED_CUDA:3:1}"
    elif [[ ${#DESIRED_CUDA} -eq 5 ]]; then
        CUDA_VERSION="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by DESIRED_CUDA"

    # There really has to be a better way to do this - eli
    # Possibly limiting builds to specific cuda versions be delimiting images would be a choice
    if [[ "$OS_NAME" == *"Ubuntu"* ]]; then
        echo "Switching to CUDA version $desired_cuda"
        /builder/conda/switch_cuda_version.sh "${DESIRED_CUDA}"
    fi
else
    CUDA_VERSION=$(nvcc --version|tail -n1|cut -f5 -d" "|cut -f1 -d",")
    echo "CUDA $CUDA_VERSION Detected"
fi

export TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0"
case ${CUDA_VERSION} in
    11.*)
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;7.5;8.0"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    10.*)
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;7.5"
        EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
        ;;
    9.*)
        export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
        ;;
    *)
        echo "unknown cuda version $CUDA_VERSION"
        exit 1
        ;;
esac
echo $TORCH_CUDA_ARCH_LIST

cuda_version_nodot=$(echo $CUDA_VERSION | tr -d '.')

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

OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
fi

if [[ $CUDA_VERSION == "9.0" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.9.0"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.9.0"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "$LIBGOMP_PATH"
)

DEPS_SONAME=(
    "libcudart.so.9.0"
    "libnvToolsExt.so.1"
    "libnvrtc.so.9.0"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
elif [[ $CUDA_VERSION == "9.2" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.9.2"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.9.2"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "$LIBGOMP_PATH"
)

DEPS_SONAME=(
    "libcudart.so.9.2"
    "libnvToolsExt.so.1"
    "libnvrtc.so.9.2"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
elif [[ $CUDA_VERSION == "10.0" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.10.0"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.10.0"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "$LIBGOMP_PATH"
)

DEPS_SONAME=(
    "libcudart.so.10.0"
    "libnvToolsExt.so.1"
    "libnvrtc.so.10.0"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
elif [[ $CUDA_VERSION == "10.1" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.10.1"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.10.1"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "$LIBGOMP_PATH"
)

DEPS_SONAME=(
    "libcudart.so.10.1"
    "libnvToolsExt.so.1"
    "libnvrtc.so.10.1"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
elif [[ $CUDA_VERSION == "10.2" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.10.2"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.10.2"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "$LIBGOMP_PATH"
)

DEPS_SONAME=(
    "libcudart.so.10.2"
    "libnvToolsExt.so.1"
    "libnvrtc.so.10.2"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
elif [[ $CUDA_VERSION == "11.0" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.11.0"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.11.0"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "$LIBGOMP_PATH"
)

DEPS_SONAME=(
    "libcudart.so.11.0"
    "libnvToolsExt.so.1"
    "libnvrtc.so.11.0"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
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

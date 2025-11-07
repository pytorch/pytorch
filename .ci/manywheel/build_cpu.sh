#!/usr/bin/env bash

set -ex

export TH_BINARY_BUILD=1
export USE_CUDA=0

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
echo "Building CPU wheel for architecture: $ARCH"

# Enable MKLDNN with ARM Compute Library for ARM builds
if [[ "$ARCH" == "aarch64" ]]; then
  export USE_MKLDNN=1
  # Only enable ACL if it's installed
  if [[ -d "/acl" ]]; then
    export USE_MKLDNN_ACL=1
    export ACL_ROOT_DIR=/acl
    echo "ARM Compute Library enabled for MKLDNN: ACL_ROOT_DIR=/acl"
  else
    echo "Warning: ARM Compute Library not found at /acl, building without ACL optimization"
  fi
fi

WHEELHOUSE_DIR="wheelhousecpu"
LIBTORCH_HOUSE_DIR="libtorch_housecpu"
if [[ -z "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    if [[ -z "$BUILD_PYTHONLESS" ]]; then
        PYTORCH_FINAL_PACKAGE_DIR="/remote/wheelhousecpu"
    else
        PYTORCH_FINAL_PACKAGE_DIR="/remote/libtorch_housecpu"
    fi
fi
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true

OS_NAME=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Red Hat Enterprise Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"AlmaLinux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    if [[ "$ARCH" == "s390x" ]]; then
        LIBGOMP_PATH="/usr/lib/s390x-linux-gnu/libgomp.so.1"
    elif [[ "$ARCH" == "aarch64" ]]; then
        LIBGOMP_PATH="/usr/lib/aarch64-linux-gnu/libgomp.so.1"
    else
        LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    fi
fi

DEPS_LIST=(
    "$LIBGOMP_PATH"
)

DEPS_SONAME=(
    "libgomp.so.1"
)

# Add ARM-specific library dependencies for CPU builds
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Adding ARM-specific CPU library dependencies"

    # ARM Compute Library (if available)
    if [[ -d "/acl/build" ]]; then
        echo "Adding ARM Compute Library for CPU"
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
        "/usr/lib64/libgfortran.so.5"
    )
    DEPS_SONAME+=(
        "libgfortran.so.5"
    )
fi

rm -rf /usr/local/cuda*

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
if [[ -z "$BUILD_PYTHONLESS" ]]; then
    BUILD_SCRIPT=build_common.sh
else
    BUILD_SCRIPT=build_libtorch.sh
fi
source ${SOURCE_DIR}/${BUILD_SCRIPT}

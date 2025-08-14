#!/bin/bash
set -ex

# Set ROCM_HOME isn't available, use ROCM_PATH if set or /opt/rocm
ROCM_HOME="${ROCM_HOME:-${ROCM_PATH:-/opt/rocm}}"

# Find rocm_version.h header file for ROCm version extract
rocm_version_h="${ROCM_HOME}/include/rocm-core/rocm_version.h"
if [ ! -f "$rocm_version_h" ]; then
    rocm_version_h="${ROCM_HOME}/include/rocm_version.h"
fi

# Error out if rocm_version.h not found
if [ ! -f "$rocm_version_h" ]; then
    echo "Error: rocm_version.h not found in expected locations." >&2
    exit 1
fi

# Extract major, minor and patch ROCm version numbers
MAJOR_VERSION=$(grep 'ROCM_VERSION_MAJOR' "$rocm_version_h" | awk '{print $3}')
MINOR_VERSION=$(grep 'ROCM_VERSION_MINOR' "$rocm_version_h" | awk '{print $3}')
PATCH_VERSION=$(grep 'ROCM_VERSION_PATCH' "$rocm_version_h" | awk '{print $3}')
ROCM_INT=$(($MAJOR_VERSION * 10000 + $MINOR_VERSION * 100 + $PATCH_VERSION))
echo "ROCm version: $ROCM_INT"

# Check TRITON_ROCM_DIR is set
if [[ -z "${TRITON_ROCM_DIR}" ]]; then
    export TRITON_ROCM_DIR=third_party/amd/backend
fi

# Remove packaged libs and headers
rm -rf $TRITON_ROCM_DIR/include/*

LIBNUMA_PATH="/usr/lib64/libnuma.so.1"
LIBELF_PATH="/usr/lib64/libelf.so.1"
OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    LIBTINFO_PATH="/usr/lib64/libtinfo.so.5"
else
    LIBTINFO_PATH="/usr/lib64/libtinfo.so.6"
fi

OS_SO_PATHS=(
    $LIBELF_PATH
    $LIBNUMA_PATH
    $LIBTINFO_PATH
)

for lib in "${OS_SO_PATHS[@]}"
do
    cp $lib $TRITON_ROCM_DIR/lib/
done

# Required ROCm libraries - ROCm 6.0
ROCM_SO=(
    "libamdhip64.so"
    "libhsa-runtime64.so"
    "libdrm.so"
    "libdrm_amdgpu.so"
    "libamd_comgr.so"
    "librocprofiler-register.so"
)

for lib in "${ROCM_SO[@]}"
do
    file_path=($(find $ROCM_HOME/lib/ -name "$lib")) # First search in lib
    if [[ -z $file_path ]]; then
        if [ -d "$ROCM_HOME/lib64/" ]; then
            file_path=($(find $ROCM_HOME/lib64/ -name "$lib")) # Then search in lib64
        fi
    fi
    if [[ -z $file_path ]]; then
        file_path=($(find $ROCM_HOME/ -name "$lib")) # Then search in ROCM_HOME
    fi
    if [[ -z $file_path ]]; then
        file_path=($(find /opt/ -name "$lib")) # Then search in /opt
    fi
    if [[ -z $file_path ]]; then
            echo "Error: Library file $lib is not found." >&2
            exit 1
    fi

    cp $file_path $TRITON_ROCM_DIR/lib
done

# Copy Include Files
cp -r $ROCM_HOME/include/hip $TRITON_ROCM_DIR/include
cp -r $ROCM_HOME/include/roctracer $TRITON_ROCM_DIR/include
cp -r $ROCM_HOME/include/hsa $TRITON_ROCM_DIR/include

# Copy linker
mkdir -p $TRITON_ROCM_DIR/llvm/bin
cp $ROCM_HOME/llvm/bin/ld.lld $TRITON_ROCM_DIR/llvm/bin/

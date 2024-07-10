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

LIBTINFO_PATH="/usr/lib64/libtinfo.so.5"
LIBNUMA_PATH="/usr/lib64/libnuma.so.1"
LIBELF_PATH="/usr/lib64/libelf.so.1"

OS_SO_PATHS=(
    $LIBELF_PATH
    $LIBNUMA_PATH
    $LIBTINFO_PATH
)

for lib in "${OS_SO_PATHS[@]}"
do
    cp $lib $TRITON_ROCM_DIR/lib/
done

# Required ROCm libraries
if [[ "${MAJOR_VERSION}" == "6" ]]; then
    libamdhip="libamdhip64.so.6"
else
    libamdhip="libamdhip64.so.5"
fi

# Required ROCm libraries - ROCm 6.0
ROCM_SO=(
    "${libamdhip}"
    "libhsa-runtime64.so.1"
    "libamd_comgr.so.2"
    "libdrm.so.2"
    "libdrm_amdgpu.so.1"
)

if [[ $ROCM_INT -ge 60100 ]]; then
    ROCM_SO+=("librocprofiler-register.so.0")
fi

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
    # When running locally, and not building a wheel, we need to satisfy shared objects requests that don't look for versions
    LINKNAME=$(echo $lib | sed -e 's/\.so.*/.so/g')
    ln -sf $lib $TRITON_ROCM_DIR/lib/$LINKNAME

done

# Copy Include Files
cp -r $ROCM_HOME/include/hip $TRITON_ROCM_DIR/include

# Copy linker
mkdir -p $TRITON_ROCM_DIR/llvm/bin
cp $ROCM_HOME/llvm/bin/ld.lld $TRITON_ROCM_DIR/llvm/bin/

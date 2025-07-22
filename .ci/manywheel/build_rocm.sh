#!/usr/bin/env bash

set -ex

export ROCM_HOME=/opt/rocm
export MAGMA_HOME=$ROCM_HOME/magma
# TODO: libtorch_cpu.so is broken when building with Debug info
export BUILD_DEBUG_INFO=0

# TODO Are these all used/needed?
export TH_BINARY_BUILD=1
export USE_STATIC_CUDNN=1
export USE_STATIC_NCCL=1
export ATEN_STATIC_CUDA=1
export USE_CUDA_STATIC_LINK=1
export INSTALL_TEST=0 # dont install test binaries into site-packages
# Set RPATH instead of RUNPATH when using patchelf to avoid LD_LIBRARY_PATH override
export FORCE_RPATH="--force-rpath"

# Keep an array of cmake variables to add to
if [[ -z "$CMAKE_ARGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build()
    CMAKE_ARGS=()
fi
if [[ -z "$EXTRA_CAFFE2_CMAKE_FLAGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build_caffe2()
    EXTRA_CAFFE2_CMAKE_FLAGS=()
fi

# Determine ROCm version and architectures to build for
#
# NOTE: We should first check `DESIRED_CUDA` when determining `ROCM_VERSION`
if [[ -n "$DESIRED_CUDA" ]]; then
    if ! echo "${DESIRED_CUDA}"| grep "^rocm" >/dev/null 2>/dev/null; then
        export DESIRED_CUDA="rocm${DESIRED_CUDA}"
    fi
    # rocm3.7, rocm3.5.1
    ROCM_VERSION="$DESIRED_CUDA"
    echo "Using $ROCM_VERSION as determined by DESIRED_CUDA"
else
    echo "Must set DESIRED_CUDA"
    exit 1
fi

# Package directories
WHEELHOUSE_DIR="wheelhouse$ROCM_VERSION"
LIBTORCH_HOUSE_DIR="libtorch_house$ROCM_VERSION"
if [[ -z "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    if [[ -z "$BUILD_PYTHONLESS" ]]; then
        PYTORCH_FINAL_PACKAGE_DIR="/remote/wheelhouse$ROCM_VERSION"
    else
        PYTORCH_FINAL_PACKAGE_DIR="/remote/libtorch_house$ROCM_VERSION"
    fi
fi
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true

# To make version comparison easier, create an integer representation.
ROCM_VERSION_CLEAN=$(echo ${ROCM_VERSION} | sed s/rocm//)
save_IFS="$IFS"
IFS=. ROCM_VERSION_ARRAY=(${ROCM_VERSION_CLEAN})
IFS="$save_IFS"
if [[ ${#ROCM_VERSION_ARRAY[@]} == 2 ]]; then
    ROCM_VERSION_MAJOR=${ROCM_VERSION_ARRAY[0]}
    ROCM_VERSION_MINOR=${ROCM_VERSION_ARRAY[1]}
    ROCM_VERSION_PATCH=0
elif [[ ${#ROCM_VERSION_ARRAY[@]} == 3 ]]; then
    ROCM_VERSION_MAJOR=${ROCM_VERSION_ARRAY[0]}
    ROCM_VERSION_MINOR=${ROCM_VERSION_ARRAY[1]}
    ROCM_VERSION_PATCH=${ROCM_VERSION_ARRAY[2]}
else
    echo "Unhandled ROCM_VERSION ${ROCM_VERSION}"
    exit 1
fi
ROCM_INT=$(($ROCM_VERSION_MAJOR * 10000 + $ROCM_VERSION_MINOR * 100 + $ROCM_VERSION_PATCH))

# Required ROCm libraries
ROCM_SO_FILES=(
    "libMIOpen.so"
    "libamdhip64.so"
    "libhipblas.so"
    "libhipfft.so"
    "libhiprand.so"
    "libhipsolver.so"
    "libhipsparse.so"
    "libhsa-runtime64.so"
    "libamd_comgr.so"
    "libmagma.so"
    "librccl.so"
    "librocblas.so"
    "librocfft.so"
    "librocm_smi64.so"
    "librocrand.so"
    "librocsolver.so"
    "librocsparse.so"
    "libroctracer64.so"
    "libroctx64.so"
    "libhipblaslt.so"
    "libhipsparselt.so"
    "libhiprtc.so"
)

if [[ $ROCM_INT -ge 60100 ]]; then
    ROCM_SO_FILES+=("librocprofiler-register.so")
fi

if [[ $ROCM_INT -ge 60200 ]]; then
    ROCM_SO_FILES+=("librocm-core.so")
fi

OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`
if [[ "$OS_NAME" == *"CentOS Linux"* || "$OS_NAME" == *"AlmaLinux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
    LIBNUMA_PATH="/usr/lib64/libnuma.so.1"
    LIBELF_PATH="/usr/lib64/libelf.so.1"
    if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
        LIBTINFO_PATH="/usr/lib64/libtinfo.so.5"
    else
        LIBTINFO_PATH="/usr/lib64/libtinfo.so.6"
    fi
    LIBDRM_PATH="/opt/amdgpu/lib64/libdrm.so.2"
    LIBDRM_AMDGPU_PATH="/opt/amdgpu/lib64/libdrm_amdgpu.so.1"
    if [[ $ROCM_INT -ge 60100 && $ROCM_INT -lt 60300 ]]; then
        # Below libs are direct dependencies of libhipsolver
        LIBSUITESPARSE_CONFIG_PATH="/lib64/libsuitesparseconfig.so.4"
        if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
            LIBCHOLMOD_PATH="/lib64/libcholmod.so.2"
            # Below libs are direct dependencies of libsatlas
            LIBGFORTRAN_PATH="/lib64/libgfortran.so.3"
        else
            LIBCHOLMOD_PATH="/lib64/libcholmod.so.3"
            # Below libs are direct dependencies of libsatlas
            LIBGFORTRAN_PATH="/lib64/libgfortran.so.5"
        fi
        # Below libs are direct dependencies of libcholmod
        LIBAMD_PATH="/lib64/libamd.so.2"
        LIBCAMD_PATH="/lib64/libcamd.so.2"
        LIBCCOLAMD_PATH="/lib64/libccolamd.so.2"
        LIBCOLAMD_PATH="/lib64/libcolamd.so.2"
        LIBSATLAS_PATH="/lib64/atlas/libsatlas.so.3"
        # Below libs are direct dependencies of libsatlas
        LIBQUADMATH_PATH="/lib64/libquadmath.so.0"
    fi
    MAYBE_LIB64=lib64
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    LIBNUMA_PATH="/usr/lib/x86_64-linux-gnu/libnuma.so.1"
    LIBELF_PATH="/usr/lib/x86_64-linux-gnu/libelf.so.1"
    if [[ $ROCM_INT -ge 50300 ]]; then
        LIBTINFO_PATH="/lib/x86_64-linux-gnu/libtinfo.so.6"
    else
        LIBTINFO_PATH="/lib/x86_64-linux-gnu/libtinfo.so.5"
    fi
    LIBDRM_PATH="/usr/lib/x86_64-linux-gnu/libdrm.so.2"
    LIBDRM_AMDGPU_PATH="/usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1"
    if [[ $ROCM_INT -ge 60100 && $ROCM_INT -lt 60300 ]]; then
        # Below libs are direct dependencies of libhipsolver
        LIBCHOLMOD_PATH="/lib/x86_64-linux-gnu/libcholmod.so.3"
        # Below libs are direct dependencies of libcholmod
        LIBSUITESPARSE_CONFIG_PATH="/lib/x86_64-linux-gnu/libsuitesparseconfig.so.5"
        LIBAMD_PATH="/lib/x86_64-linux-gnu/libamd.so.2"
        LIBCAMD_PATH="/lib/x86_64-linux-gnu/libcamd.so.2"
        LIBCCOLAMD_PATH="/lib/x86_64-linux-gnu/libccolamd.so.2"
        LIBCOLAMD_PATH="/lib/x86_64-linux-gnu/libcolamd.so.2"
        LIBMETIS_PATH="/lib/x86_64-linux-gnu/libmetis.so.5"
        LIBLAPACK_PATH="/lib/x86_64-linux-gnu/liblapack.so.3"
        LIBBLAS_PATH="/lib/x86_64-linux-gnu/libblas.so.3"
        # Below libs are direct dependencies of libblas
        LIBGFORTRAN_PATH="/lib/x86_64-linux-gnu/libgfortran.so.5"
        LIBQUADMATH_PATH="/lib/x86_64-linux-gnu/libquadmath.so.0"
    fi
    MAYBE_LIB64=lib
fi
OS_SO_PATHS=($LIBGOMP_PATH $LIBNUMA_PATH\
             $LIBELF_PATH $LIBTINFO_PATH\
             $LIBDRM_PATH $LIBDRM_AMDGPU_PATH\
             $LIBSUITESPARSE_CONFIG_PATH\
             $LIBCHOLMOD_PATH $LIBAMD_PATH\
             $LIBCAMD_PATH $LIBCCOLAMD_PATH\
             $LIBCOLAMD_PATH $LIBSATLAS_PATH\
             $LIBGFORTRAN_PATH $LIBQUADMATH_PATH\
             $LIBMETIS_PATH $LIBLAPACK_PATH\
             $LIBBLAS_PATH)
OS_SO_FILES=()
for lib in "${OS_SO_PATHS[@]}"
do
    file_name="${lib##*/}" # Substring removal of path to get filename
    OS_SO_FILES[${#OS_SO_FILES[@]}]=$file_name # Append lib to array
done

ARCH=$(echo $PYTORCH_ROCM_ARCH | sed 's/;/|/g') # Replace ; separated arch list to bar for grep

# rocBLAS library files
ROCBLAS_LIB_SRC=$ROCM_HOME/lib/rocblas/library
ROCBLAS_LIB_DST=lib/rocblas/library
ROCBLAS_ARCH_SPECIFIC_FILES=$(ls $ROCBLAS_LIB_SRC | grep -E $ARCH)
ROCBLAS_OTHER_FILES=$(ls $ROCBLAS_LIB_SRC | grep -v gfx)
ROCBLAS_LIB_FILES=($ROCBLAS_ARCH_SPECIFIC_FILES $OTHER_FILES)

# hipblaslt library files
HIPBLASLT_LIB_SRC=$ROCM_HOME/lib/hipblaslt/library
HIPBLASLT_LIB_DST=lib/hipblaslt/library
HIPBLASLT_ARCH_SPECIFIC_FILES=$(ls $HIPBLASLT_LIB_SRC | grep -E $ARCH)
HIPBLASLT_OTHER_FILES=$(ls $HIPBLASLT_LIB_SRC | grep -v gfx)
HIPBLASLT_LIB_FILES=($HIPBLASLT_ARCH_SPECIFIC_FILES $HIPBLASLT_OTHER_FILES)

# hipsparselt library files
HIPSPARSELT_LIB_SRC=$ROCM_HOME/lib/hipsparselt/library
HIPSPARSELT_LIB_DST=lib/hipsparselt/library
HIPSPARSELT_ARCH_SPECIFIC_FILES=$(ls $HIPSPARSELT_LIB_SRC | grep -E $ARCH)
#HIPSPARSELT_OTHER_FILES=$(ls $HIPSPARSELT_LIB_SRC | grep -v gfx)
HIPSPARSELT_LIB_FILES=($HIPSPARSELT_ARCH_SPECIFIC_FILES $HIPSPARSELT_OTHER_FILES)

# ROCm library files
ROCM_SO_PATHS=()
for lib in "${ROCM_SO_FILES[@]}"
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
        echo "Error: Library file $lib is not found." >&2
        exit 1
    fi
    ROCM_SO_PATHS[${#ROCM_SO_PATHS[@]}]="$file_path" # Append lib to array
done

DEPS_LIST=(
    ${ROCM_SO_PATHS[*]}
    ${OS_SO_PATHS[*]}
)

DEPS_SONAME=(
    ${ROCM_SO_FILES[*]}
    ${OS_SO_FILES[*]}
)

DEPS_AUX_SRCLIST=(
    "${ROCBLAS_LIB_FILES[@]/#/$ROCBLAS_LIB_SRC/}"
    "${HIPBLASLT_LIB_FILES[@]/#/$HIPBLASLT_LIB_SRC/}"
    "${HIPSPARSELT_LIB_FILES[@]/#/$HIPSPARSELT_LIB_SRC/}"
    "/opt/amdgpu/share/libdrm/amdgpu.ids"
)

DEPS_AUX_DSTLIST=(
    "${ROCBLAS_LIB_FILES[@]/#/$ROCBLAS_LIB_DST/}"
    "${HIPBLASLT_LIB_FILES[@]/#/$HIPBLASLT_LIB_DST/}"
    "${HIPSPARSELT_LIB_FILES[@]/#/$HIPSPARSELT_LIB_DST/}"
    "share/libdrm/amdgpu.ids"
)

# MIOpen library files
MIOPEN_SHARE_SRC=$ROCM_HOME/share/miopen/db
MIOPEN_SHARE_DST=share/miopen/db
MIOPEN_SHARE_FILES=($(ls $MIOPEN_SHARE_SRC | grep -E $ARCH))
DEPS_AUX_SRCLIST+=(${MIOPEN_SHARE_FILES[@]/#/$MIOPEN_SHARE_SRC/})
DEPS_AUX_DSTLIST+=(${MIOPEN_SHARE_FILES[@]/#/$MIOPEN_SHARE_DST/})

# RCCL library files
RCCL_SHARE_SRC=$ROCM_HOME/share/rccl/msccl-algorithms
RCCL_SHARE_DST=share/rccl/msccl-algorithms
RCCL_SHARE_FILES=($(ls $RCCL_SHARE_SRC))
DEPS_AUX_SRCLIST+=(${RCCL_SHARE_FILES[@]/#/$RCCL_SHARE_SRC/})
DEPS_AUX_DSTLIST+=(${RCCL_SHARE_FILES[@]/#/$RCCL_SHARE_DST/})

echo "PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
if [[ -z "$BUILD_PYTHONLESS" ]]; then
    BUILD_SCRIPT=build_common.sh
else
    BUILD_SCRIPT=build_libtorch.sh
fi
source $SCRIPTPATH/${BUILD_SCRIPT}

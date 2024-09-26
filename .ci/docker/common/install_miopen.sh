#!/bin/bash
# Script used only in CD pipeline

set -ex

ROCM_VERSION=$1

if [[ -z $ROCM_VERSION ]]; then
    echo "missing ROCM_VERSION"
    exit 1;
fi

IS_UBUNTU=0
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    IS_UBUNTU=1
    ;;
  centos)
    IS_UBUNTU=0
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac

# To make version comparison easier, create an integer representation.
save_IFS="$IFS"
IFS=. ROCM_VERSION_ARRAY=(${ROCM_VERSION})
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

# Install custom MIOpen + COMgr for ROCm >= 4.0.1
if [[ $ROCM_INT -lt 40001 ]]; then
    echo "ROCm version < 4.0.1; will not install custom MIOpen"
    exit 0
fi

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Build custom MIOpen to use comgr for offline compilation.

## Need a sanitized ROCM_VERSION without patchlevel; patchlevel version 0 must be added to paths.
ROCM_DOTS=$(echo ${ROCM_VERSION} | tr -d -c '.' | wc -c)
if [[ ${ROCM_DOTS} == 1 ]]; then
    ROCM_VERSION_NOPATCH="${ROCM_VERSION}"
    ROCM_INSTALL_PATH="/opt/rocm-${ROCM_VERSION}.0"
else
    ROCM_VERSION_NOPATCH="${ROCM_VERSION%.*}"
    ROCM_INSTALL_PATH="/opt/rocm-${ROCM_VERSION}"
fi

# MIOPEN_USE_HIP_KERNELS is a Workaround for COMgr issues
MIOPEN_CMAKE_COMMON_FLAGS="
-DMIOPEN_USE_COMGR=ON
-DMIOPEN_BUILD_DRIVER=OFF
"
# Pull MIOpen repo and set DMIOPEN_EMBED_DB based on ROCm version
if [[ $ROCM_INT -ge 60300 ]]; then
    echo "ROCm 6.3+ MIOpen does not need any patches, do not build from source"
    exit 0
elif [[ $ROCM_INT -ge 60200 ]] && [[ $ROCM_INT -lt 60300 ]]; then
    MIOPEN_BRANCH="release/rocm-rel-6.2-staging"
elif [[ $ROCM_INT -ge 60100 ]] && [[ $ROCM_INT -lt 60200 ]]; then
    echo "ROCm 6.1 MIOpen does not need any patches, do not build from source"
    exit 0
elif [[ $ROCM_INT -ge 60000 ]] && [[ $ROCM_INT -lt 60100 ]]; then
    echo "ROCm 6.0 MIOpen does not need any patches, do not build from source"
    exit 0
elif [[ $ROCM_INT -ge 50700 ]] && [[ $ROCM_INT -lt 60000 ]]; then
    echo "ROCm 5.7 MIOpen does not need any patches, do not build from source"
    exit 0
elif [[ $ROCM_INT -ge 50600 ]] && [[ $ROCM_INT -lt 50700 ]]; then
    MIOPEN_BRANCH="release/rocm-rel-5.6-staging"
elif [[ $ROCM_INT -ge 50500 ]] && [[ $ROCM_INT -lt 50600 ]]; then
    MIOPEN_BRANCH="release/rocm-rel-5.5-gfx11"
elif [[ $ROCM_INT -ge 50400 ]] && [[ $ROCM_INT -lt 50500 ]]; then
    MIOPEN_CMAKE_DB_FLAGS="-DMIOPEN_EMBED_DB=gfx900_56;gfx906_60;gfx90878;gfx90a6e;gfx1030_36 -DMIOPEN_USE_MLIR=Off"
    MIOPEN_BRANCH="release/rocm-rel-5.4-staging"
elif [[ $ROCM_INT -ge 50300 ]] && [[ $ROCM_INT -lt 50400 ]]; then
    MIOPEN_CMAKE_DB_FLAGS="-DMIOPEN_EMBED_DB=gfx900_56;gfx906_60;gfx90878;gfx90a6e;gfx1030_36 -DMIOPEN_USE_MLIR=Off"
    MIOPEN_BRANCH="release/rocm-rel-5.3-staging"
elif [[ $ROCM_INT -ge 50200 ]] && [[ $ROCM_INT -lt 50300 ]]; then
    MIOPEN_CMAKE_DB_FLAGS="-DMIOPEN_EMBED_DB=gfx900_56;gfx906_60;gfx90878;gfx90a6e;gfx1030_36 -DMIOPEN_USE_MLIR=Off"
    MIOPEN_BRANCH="release/rocm-rel-5.2-staging"
elif [[ $ROCM_INT -ge 50100 ]] && [[ $ROCM_INT -lt 50200 ]]; then
    MIOPEN_CMAKE_DB_FLAGS="-DMIOPEN_EMBED_DB=gfx900_56;gfx906_60;gfx90878;gfx90a6e;gfx1030_36"
    MIOPEN_BRANCH="release/rocm-rel-5.1-staging"
elif [[ $ROCM_INT -ge 50000 ]] && [[ $ROCM_INT -lt 50100 ]]; then
    MIOPEN_CMAKE_DB_FLAGS="-DMIOPEN_EMBED_DB=gfx900_56;gfx906_60;gfx90878;gfx90a6e;gfx1030_36"
    MIOPEN_BRANCH="release/rocm-rel-5.0-staging"
else
    echo "Unhandled ROCM_VERSION ${ROCM_VERSION}"
    exit 1
fi


if [[ ${IS_UBUNTU} == 1 ]]; then
  apt-get remove -y miopen-hip
else
  yum remove -y miopen-hip
fi

git clone https://github.com/ROCm/MIOpen -b ${MIOPEN_BRANCH}
pushd MIOpen
# remove .git to save disk space since CI runner was running out
rm -rf .git
# Don't build CK to save docker build time
if [[ $ROCM_INT -ge 60200 ]]; then
    sed -i '/composable_kernel/d' requirements.txt
fi
# Don't build MLIR to save docker build time
# since we are disabling MLIR backend for MIOpen anyway
if [[ $ROCM_INT -ge 50400 ]] && [[ $ROCM_INT -lt 50500 ]]; then
    sed -i '/rocMLIR/d' requirements.txt
elif [[ $ROCM_INT -ge 50200 ]] && [[ $ROCM_INT -lt 50400 ]]; then
    sed -i '/llvm-project-mlir/d' requirements.txt
fi
## MIOpen minimum requirements
cmake -P install_deps.cmake --minimum

# clean up since CI runner was running out of disk space
rm -rf /tmp/*
if [[ ${IS_UBUNTU} == 1 ]]; then
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
else
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
fi

## Build MIOpen
mkdir -p build
cd build
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig CXX=${ROCM_INSTALL_PATH}/llvm/bin/clang++ cmake .. \
    ${MIOPEN_CMAKE_COMMON_FLAGS} \
    ${MIOPEN_CMAKE_DB_FLAGS} \
    -DCMAKE_PREFIX_PATH="${ROCM_INSTALL_PATH}/hip;${ROCM_INSTALL_PATH}"
make MIOpen -j $(nproc)

# Build MIOpen package
make -j $(nproc) package

# clean up since CI runner was running out of disk space
rm -rf /usr/local/cget

if [[ ${IS_UBUNTU} == 1 ]]; then
  sudo dpkg -i miopen-hip*.deb
else
  yum install -y miopen-*.rpm
fi

popd
rm -rf MIOpen

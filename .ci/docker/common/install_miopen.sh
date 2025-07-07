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
  centos|almalinux)
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

MIOPEN_CMAKE_COMMON_FLAGS="
-DMIOPEN_USE_COMGR=ON
-DMIOPEN_BUILD_DRIVER=OFF
"
if [[ $ROCM_INT -ge 60200 ]] && [[ $ROCM_INT -lt 60204 ]]; then
    MIOPEN_BRANCH="release/rocm-rel-6.2-staging"
else
    echo "ROCm ${ROCM_VERSION} does not need any patches, do not build from source"
    exit 0
fi


if [[ ${IS_UBUNTU} == 1 ]]; then
  apt-get remove -y miopen-hip
else
  # Workaround since almalinux manylinux image already has this and cget doesn't like that
  rm -rf /usr/local/lib/pkgconfig/sqlite3.pc

  # Versioned package name needs regex match
  # Use --noautoremove to prevent other rocm packages from being uninstalled
  yum remove -y miopen-hip* --noautoremove
fi

git clone https://github.com/ROCm/MIOpen -b ${MIOPEN_BRANCH}
pushd MIOpen
# remove .git to save disk space since CI runner was running out
rm -rf .git
# Don't build CK to save docker build time
sed -i '/composable_kernel/d' requirements.txt
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
    -DCMAKE_PREFIX_PATH="${ROCM_INSTALL_PATH}"
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

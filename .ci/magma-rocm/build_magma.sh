#!/usr/bin/env bash

set -eou pipefail

# Environment variables
# The script expects DESIRED_CUDA and PACKAGE_NAME to be set
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# https://github.com/icl-utk-edu/magma/pull/77
MAGMA_VERSION=a68b9257ac435afa1ebdfb8f50d67668950aef61

# Folders for the build
PACKAGE_FILES=${ROOT_DIR}/magma-rocm/package_files # metadata
PACKAGE_DIR=${ROOT_DIR}/magma-rocm/${PACKAGE_NAME} # build workspace
PACKAGE_OUTPUT=${ROOT_DIR}/magma-rocm/output # where tarballs are stored
PACKAGE_BUILD=${PACKAGE_DIR} # where the content of the tarball is prepared
PACKAGE_RECIPE=${PACKAGE_BUILD}/info/recipe
PACKAGE_LICENSE=${PACKAGE_BUILD}/info/licenses
mkdir -p ${PACKAGE_DIR} ${PACKAGE_OUTPUT}/linux-64 ${PACKAGE_BUILD} ${PACKAGE_RECIPE} ${PACKAGE_LICENSE}

# Fetch magma sources and verify checksum
pushd ${PACKAGE_DIR}
git clone https://github.com/k-artem/magma.git
pushd magma
git checkout ${MAGMA_VERSION}
popd
popd

# build
pushd ${PACKAGE_DIR}/magma
# The build.sh script expects to be executed from the sources root folder
INSTALL_DIR=${PACKAGE_BUILD} ${PACKAGE_FILES}/build.sh
popd

# Package recipe, license and tarball
# Folder and package name are backward compatible for the build workflow
cp ${PACKAGE_FILES}/build.sh ${PACKAGE_RECIPE}/build.sh
cp ${PACKAGE_DIR}/magma/COPYRIGHT ${PACKAGE_LICENSE}/COPYRIGHT
pushd ${PACKAGE_BUILD}
tar cjf ${PACKAGE_OUTPUT}/linux-64/${PACKAGE_NAME}-${MAGMA_VERSION}-1.tar.bz2 include lib info
echo Built in ${PACKAGE_OUTPUT}/linux-64/${PACKAGE_NAME}-${MAGMA_VERSION}-1.tar.bz2
popd

#!/usr/bin/env bash

set -eou pipefail

# Environment variables
# The script expects DESIRED_CUDA and PACKAGE_NAME to be set
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAGMA_VERSION=2.6.1

# Folders for the build
PACKAGE_FILES=${ROOT_DIR}/magma/package_files # source patches and metadata
PACKAGE_DIR=${ROOT_DIR}/magma/${PACKAGE_NAME} # build workspace
PACKAGE_OUTPUT=${ROOT_DIR}/magma/output # where tarballs are stored
PACKAGE_BUILD=${PACKAGE_DIR}/build # where the content of the tarball is prepared
PACKAGE_RECIPE=${PACKAGE_BUILD}/info/recipe
PACKAGE_LICENSE=${PACKAGE_BUILD}/info/licenses
mkdir -p ${PACKAGE_DIR} ${PACKAGE_OUTPUT}/linux-64 ${PACKAGE_BUILD} ${PACKAGE_RECIPE} ${PACKAGE_LICENSE}

# Fetch magma sources and verify checksum
pushd ${PACKAGE_DIR}
curl -LO http://icl.utk.edu/projectsfiles/magma/downloads/magma-${MAGMA_VERSION}.tar.gz
tar zxf magma-${MAGMA_VERSION}.tar.gz
sha256sum --check < ${PACKAGE_FILES}/magma-${MAGMA_VERSION}.sha256
popd

# Apply patches and build
pushd ${PACKAGE_DIR}/magma-${MAGMA_VERSION}
patch < ${PACKAGE_FILES}/CMake.patch
patch < ${PACKAGE_FILES}/cmakelists.patch
patch -p0 < ${PACKAGE_FILES}/thread_queue.patch
patch -p1 < ${PACKAGE_FILES}/getrf_shfl.patch
patch -p1 < ${PACKAGE_FILES}/getrf_nbparam.patch
# The build.sh script expects to be executed from the sources root folder
INSTALL_DIR=${PACKAGE_BUILD} ${PACKAGE_FILES}/build.sh
popd

# Package recipe, license and tarball
# Folder and package name are backward compatible for the build workflow
cp ${PACKAGE_FILES}/build.sh ${PACKAGE_RECIPE}/build.sh
cp ${PACKAGE_FILES}/thread_queue.patch ${PACKAGE_RECIPE}/thread_queue.patch
cp ${PACKAGE_FILES}/cmakelists.patch ${PACKAGE_RECIPE}/cmakelists.patch
cp ${PACKAGE_FILES}/getrf_shfl.patch ${PACKAGE_RECIPE}/getrf_shfl.patch
cp ${PACKAGE_FILES}/getrf_nbparam.patch ${PACKAGE_RECIPE}/getrf_nbparam.patch
cp ${PACKAGE_FILES}/CMake.patch ${PACKAGE_RECIPE}/CMake.patch
cp ${PACKAGE_FILES}/magma-${MAGMA_VERSION}.sha256 ${PACKAGE_RECIPE}/magma-${MAGMA_VERSION}.sha256
cp ${PACKAGE_DIR}/magma-${MAGMA_VERSION}/COPYRIGHT ${PACKAGE_LICENSE}/COPYRIGHT
pushd ${PACKAGE_BUILD}
tar cjf ${PACKAGE_OUTPUT}/linux-64/${PACKAGE_NAME}-${MAGMA_VERSION}-1.tar.bz2 include lib info
echo Built in ${PACKAGE_OUTPUT}/linux-64/${PACKAGE_NAME}-${MAGMA_VERSION}-1.tar.bz2
popd

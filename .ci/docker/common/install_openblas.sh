#!/bin/bash
# Script used only in CD pipeline

set -ex

OPENBLAS_VERSION=${OPENBLAS_VERSION:-"v0.3.33"}
OPENBLAS_CHECKOUT_DIR="OpenBLAS"

if [[ "$BUILD_ENVIRONMENT" == *riscv64* ]]; then
  # FIXME: build OpenBLAS from scratch, it just takes too long right now
  apt-get update
  apt-get install -y libopenblas-dev

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

  exit 0
elif [[ "$BUILD_ENVIRONMENT" == *aarch64* ]]; then
  OPENBLAS_TARGET="ARMV8"
fi

# Clone OpenBLAS
git clone https://github.com/OpenMathLib/OpenBLAS.git -b "${OPENBLAS_VERSION}" --depth 1 --shallow-submodules "${OPENBLAS_CHECKOUT_DIR}"

OPENBLAS_BUILD_FLAGS="
CC=gcc
NUM_THREADS=128
USE_OPENMP=1
NO_SHARED=0
DYNAMIC_ARCH=1
TARGET=${OPENBLAS_TARGET}
CFLAGS=-O3
FFLAGS=-Wno-maybe-uninitialized
BUILD_BFLOAT16=1
BUILD_HFLOAT16=1
BUILD_SINGLE=1
BUILD_DOUBLE=1
BUILD_COMPLEX=1
BUILD_COMPLEX16=1
"

make -j8 ${OPENBLAS_BUILD_FLAGS} -C $OPENBLAS_CHECKOUT_DIR
sudo make install ${OPENBLAS_BUILD_FLAGS} -C $OPENBLAS_CHECKOUT_DIR

rm -rf $OPENBLAS_CHECKOUT_DIR

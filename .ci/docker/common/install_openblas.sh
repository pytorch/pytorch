#!/bin/bash
# Script used only in CD pipeline

set -ex

OPENBLAS_VERSION=${OPENBLAS_VERSION:-"v0.3.30"}
MAX_JOBS=${MAX_JOBS:-$(( $(nproc) - 2 ))}

# Optional ccache support
USE_CCACHE=${USE_CCACHE:-0}

if [ "${USE_CCACHE}" != "0" ]; then
  echo "Using ccache for OpenBLAS build"
  CC_FOR_OPENBLAS="ccache gcc"
else
  CC_FOR_OPENBLAS="gcc"
fi

# Clone OpenBLAS
git clone https://github.com/OpenMathLib/OpenBLAS.git -b "${OPENBLAS_VERSION}" --depth 1 --shallow-submodules

OPENBLAS_CHECKOUT_DIR="OpenBLAS"
make -j"${MAX_JOBS}" \
  CC="${CC_FOR_OPENBLAS}" \
  NUM_THREADS=128 \
  USE_OPENMP=1 \
  NO_SHARED=0 \
  DYNAMIC_ARCH=1 \
  TARGET=ARMV8 \
  CFLAGS=-O3 \
  BUILD_BFLOAT16=1 \
  -C "$OPENBLAS_CHECKOUT_DIR"

sudo make install -C $OPENBLAS_CHECKOUT_DIR

rm -rf $OPENBLAS_CHECKOUT_DIR
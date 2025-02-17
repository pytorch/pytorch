#!/bin/bash
# Script used only in CD pipeline

set -ex

cd /
git clone https://github.com/OpenMathLib/OpenBLAS.git -b develop --depth 1 --shallow-submodules


OPENBLAS_BUILD_FLAGS="
NUM_THREADS=128
USE_OPENMP=1
NO_SHARED=0
DYNAMIC_ARCH=1
TARGET=ARMV8
CFLAGS=-O3
BUILD_BFLOAT16=1
"

OPENBLAS_CHECKOUT_DIR="OpenBLAS"

make -j$(nproc) ${OPENBLAS_BUILD_FLAGS} -C ${OPENBLAS_CHECKOUT_DIR}
make -j$(nproc) ${OPENBLAS_BUILD_FLAGS} install -C ${OPENBLAS_CHECKOUT_DIR}

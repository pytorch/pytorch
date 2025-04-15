#!/bin/bash
# Script used only in CD pipeline

set -ex

cd /
OPENBLAS_HASH="b30dc9701f8e971720a02e24068acea274fd9cee" #Use SVE kernel for S/DGEMVT for SVE machines
OPENBLAS_CHECKOUT_DIR="OpenBLAS"
git clone https://github.com/OpenMathLib/OpenBLAS.git -b develop --shallow-submodules
git -C $OPENBLAS_CHECKOUT_DIR fetch --depth 1 origin $OPENBLAS_HASH
git -C $OPENBLAS_CHECKOUT_DIR checkout $OPENBLAS_HASH

OPENBLAS_BUILD_FLAGS="
NUM_THREADS=128
USE_OPENMP=1
NO_SHARED=0
DYNAMIC_ARCH=1
TARGET=ARMV8
CFLAGS=-O3
BUILD_BFLOAT16=1
"

make -j8 ${OPENBLAS_BUILD_FLAGS} -C ${OPENBLAS_CHECKOUT_DIR}
make -j8 ${OPENBLAS_BUILD_FLAGS} install -C ${OPENBLAS_CHECKOUT_DIR}

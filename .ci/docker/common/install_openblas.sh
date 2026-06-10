#!/bin/bash
# Script used only in CD pipeline

set -euo pipefail

OPENBLAS_VERSION=${OPENBLAS_VERSION:-"v0.3.33"}
MAX_JOBS=${MAX_JOBS:-$(nproc --ignore=2)}

OPENBLAS_CHECKOUT_DIR="OpenBLAS"
OPENBLAS_REPO_URL="https://github.com/OpenMathLib/OpenBLAS.git"

# Clone OpenBLAS
mkdir -p "$OPENBLAS_CHECKOUT_DIR"
(
  # shallow clone ACL_GIT_REF
  cd "$OPENBLAS_CHECKOUT_DIR"
  git init
  git remote add origin "$OPENBLAS_REPO_URL"
  git fetch --depth=1 --recurse-submodules=no origin "$OPENBLAS_VERSION"
  git checkout -f FETCH_HEAD
)

make -j"${MAX_JOBS}" \
  CC=gcc \
  NUM_THREADS=128 \
  USE_OPENMP=1 \
  NO_SHARED=0 \
  DYNAMIC_ARCH=1 \
  TARGET=ARMV8 \
  CFLAGS=-O3 \
  BUILD_BFLOAT16=1 \
  -C "$OPENBLAS_CHECKOUT_DIR"
sudo make install -C $OPENBLAS_CHECKOUT_DIR

# Clean up checkout
rm -rf $OPENBLAS_CHECKOUT_DIR
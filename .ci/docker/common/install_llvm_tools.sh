#!/bin/bash

set -eux

yum install -y llvm-devel lld

LLVM_VERSION="$(rpm -q --qf '%{VERSION}' llvm-devel)"
LLVM_REPO="https://github.com/llvm/llvm-project.git"

git clone --depth 1 --filter=blob:none --sparse --branch "llvmorg-${LLVM_VERSION}" "${LLVM_REPO}" /llvm

git -C /llvm sparse-checkout set bolt cmake llvm

cmake -S /llvm/bolt -B /llvm/build \
  -DCMAKE_INSTALL_PREFIX=/llvm/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=AArch64 \
  -DBOLT_TARGETS_TO_BUILD=AArch64 \
  -DBOLT_INCLUDE_TESTS=OFF

cmake --build /llvm/build --target install-bolt --parallel

/llvm/install/bin/llvm-bolt --version

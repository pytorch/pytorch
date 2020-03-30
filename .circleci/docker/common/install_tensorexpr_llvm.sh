#!/bin/bash

set -ex

if [ -n "$TENSOREXPR_LLVM" ]; then

    cd $HOME
    git clone https://github.com/llvm/llvm-project.git
    cd llvm-project
    git checkout llvmorg-9.0.1
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_INSTALL_PREFIX=$HOME/install-llvm -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_RTTI=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_BUILD_TOOLS=OFF  ../llvm
    cmake --build .
    make install
fi


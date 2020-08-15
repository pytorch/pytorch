#!/bin/bash

set -ex
llvm_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/llvm-9.0.1.src.tar.xz"

mkdir /opt/llvm

pushd /tmp
wget --no-verbose --output-document=llvm.tar.xz "$llvm_url"
mkdir llvm
tar -xf llvm.tar.xz -C llvm --strip-components 1
rm -f llvm.tar.xz

cd llvm
mkdir build
cd build
cmake -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_BUILD_UTILS=OFF \
  -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON \
  ../

make -j4
sudo make install

popd

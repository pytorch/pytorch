#!/bin/bash

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Use openmp from conda which supports 11.0. Otherwise we'll end up with
# whatever version comes with homebrew which only supports the build machine's
# OS version or higher
OMP_PREFIX=/opt/llvm-openmp
sudo mkdir -p ${OMP_PREFIX}
sudo chown -R $USER: ${OMP_PREFIX}
# need zstd to extract
retry brew install zstd
pushd ${OMP_PREFIX}
  llvm_openmp_version="21.1.8-h4a912ad_0"
  retry curl -OLs https://conda.anaconda.org/conda-forge/osx-arm64/llvm-openmp-${llvm_openmp_version}.conda
  tar -xvf llvm-openmp-${llvm_openmp_version}.conda
  rm llvm-openmp-${llvm_openmp_version}.conda
  tar -xvf pkg-llvm-openmp-${llvm_openmp_version}.tar.zst
  rm pkg-llvm-openmp-${llvm_openmp_version}.tar.zst
  rm info-llvm-openmp-${llvm_openmp_version}.tar.zst
  rm lib/libiomp5.dylib
  install_name_tool -id ${OMP_PREFIX}/lib/libomp.dylib lib/libomp.dylib
  codesign -f -s - lib/libomp.dylib
popd

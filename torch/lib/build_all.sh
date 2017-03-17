#!/usr/bin/env bash

set -e

WITH_CUDA=0
WITH_NCCL=0
WITH_DISTRIBUTED=0
for arg in "$@"; do
    if [[ "$arg" == "--with-cuda" ]]; then
        WITH_CUDA=1
    elif [[ "$arg" == "--with-nccl" ]]; then
        WITH_NCCL=1
    elif [[ "$arg" == "--with-distributed" ]]; then
        WITH_DISTRIBUTED=1
    else
        echo "Unknown argument: $arg"
    fi
done

cd "$(dirname "$0")/../.."
BASE_DIR=$(pwd)
cd torch/lib
INSTALL_DIR="$(pwd)/tmp_install"
BASIC_C_FLAGS=" -DTH_INDEX_BASE=0 -I$INSTALL_DIR/include \
  -I$INSTALL_DIR/include/TH -I$INSTALL_DIR/include/THC \
  -I$INSTALL_DIR/include/THS -I$INSTALL_DIR/include/THCS \
  -I$INSTALL_DIR/include/THPP "
LDFLAGS="-L$INSTALL_DIR/lib "
LD_POSTFIX=".so.1"
LD_POSTFIX_UNVERSIONED=".so"
if [[ $(uname) == 'Darwin' ]]; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,@loader_path"
    LD_POSTFIX=".1.dylib"
    LD_POSTFIX_UNVERSIONED=".dylib"
else
    LDFLAGS="$LDFLAGS -Wl,-rpath,\$ORIGIN"
fi
C_FLAGS="$BASIC_C_FLAGS $LDFLAGS"
function build() {
  mkdir -p build/$1
  cd build/$1
  cmake ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$C_FLAGS" \
              -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS" \
              -DCUDA_NVCC_FLAGS="$BASIC_C_FLAGS" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
              -DTH_LIB_PATH="$INSTALL_DIR/lib" \
              -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
              -DTHPP_LIBRARIES="$INSTALL_DIR/lib/libTHPP$LD_POSTFIX" \
              -DTHS_LIBRARIES="$INSTALL_DIR/lib/libTHS$LD_POSTFIX" \
              -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
              -DTHCS_LIBRARIES="$INSTALL_DIR/lib/libTHCS$LD_POSTFIX" \
              -DTH_SO_VERSION=1 \
              -DTHC_SO_VERSION=1 \
              -DTHNN_SO_VERSION=1 \
              -DTHCUNN_SO_VERSION=1 \
              -DTHD_SO_VERSION=1 \
              -DNO_CUDA=$((1-$WITH_CUDA)) \
              -DCMAKE_BUILD_TYPE=$([ $DEBUG ] && echo Debug || echo Release)
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  rm -rf $INSTALL_DIR/lib/lib$1$LD_POSTFIX_UNVERSIONED

  if [[ $(uname) == 'Darwin' ]]; then
    cd tmp_install/lib
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    cd ../..
  fi
}
function build_nccl() {
   mkdir -p build/nccl
   cd build/nccl
   cmake ../../nccl -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
               -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
               -DCMAKE_C_FLAGS="$C_FLAGS" \
               -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS"
   make install
   cp "lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so.1"
   cd ../..
}

mkdir -p tmp_install
build TH
build THS
build THNN
if [[ $WITH_CUDA -eq 1 ]]; then
    build THC
    build THCS
    build THCUNN
fi
if [[ $WITH_NCCL -eq 1 ]]; then
    build_nccl
fi

build THPP
CPP_FLAGS=" -std=c++11 "
build libshm

if [[ $WITH_DISTRIBUTED -eq 1 ]]; then
    build THD
fi

cp $INSTALL_DIR/lib/* .
cp THNN/generic/THNN.h .
cp THCUNN/generic/THCUNN.h .
cp -r $INSTALL_DIR/include .
cp $INSTALL_DIR/bin/* .

# this is for binary builds
if [[ $PYTORCH_BINARY_BUILD && $PYTORCH_SO_DEPS ]]
then
    echo "Copying over dependency libraries $PYTORCH_SO_DEPS"
    # copy over dependency libraries into the current dir
    cp $PYTORCH_SO_DEPS .
fi

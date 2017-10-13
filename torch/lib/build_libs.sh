#!/usr/bin/env bash

# Shell script used to build the torch/lib/* dependencies prior to
# linking the libraries and passing the headers to the Python extension
# compilation stage. This file is used from setup.py, but can also be
# called standalone to compile the libraries outside of the overall PyTorch
# build process.

set -e

# Options for building only a subset of the libraries
WITH_CUDA=0
if [[ "$1" == "--with-cuda" ]]; then
  WITH_CUDA=1
  shift
fi

cd "$(dirname "$0")/../.."
BASE_DIR=$(pwd)
cd torch/lib
INSTALL_DIR="$(pwd)/tmp_install"
CMAKE_VERSION=${CMAKE_VERSION:="cmake"}
C_FLAGS=" -DTH_INDEX_BASE=0 -I$INSTALL_DIR/include \
  -I$INSTALL_DIR/include/TH -I$INSTALL_DIR/include/THC \
  -I$INSTALL_DIR/include/THS -I$INSTALL_DIR/include/THCS \
  -I$INSTALL_DIR/include/THPP -I$INSTALL_DIR/include/THNN \
  -I$INSTALL_DIR/include/THCUNN"
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
CPP_FLAGS=" -std=c++11 "
GLOO_FLAGS=""
NCCL_ROOT_DIR=${NCCL_ROOT_DIR:-$INSTALL_DIR}
if [[ $WITH_CUDA -eq 1 ]]; then
    GLOO_FLAGS="-DUSE_CUDA=1 -DNCCL_ROOT_DIR=$NCCL_ROOT_DIR"
fi
CWRAP_FILES="\
$BASE_DIR/torch/lib/ATen/Declarations.cwrap;\
$BASE_DIR/torch/lib/ATen/Local.cwrap;\
$BASE_DIR/torch/lib/THNN/generic/THNN.h;\
$BASE_DIR/torch/lib/THCUNN/generic/THCUNN.h"

# Used to build an individual library, e.g. build TH
function build() {
  # We create a build directory for the library, which will
  # contain the cmake output
  mkdir -p build/$1
  cd build/$1
  BUILD_C_FLAGS=''
  case $1 in
      THCS | THCUNN ) BUILD_C_FLAGS=$C_FLAGS;;
      nanopb ) BUILD_C_FLAGS=$C_FLAGS" -fPIC -fexceptions";;
      *) BUILD_C_FLAGS=$C_FLAGS" -fexceptions";;
  esac
  ${CMAKE_VERSION} ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$BUILD_C_FLAGS" \
              -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS" \
              -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
              -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
              -DCMAKE_INSTALL_LIBDIR="$INSTALL_DIR/lib" \
              -DCUDA_NVCC_FLAGS="$C_FLAGS" \
              -Dcwrap_files="$CWRAP_FILES" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
              -DTH_LIB_PATH="$INSTALL_DIR/lib" \
              -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
              -DTHPP_LIBRARIES="$INSTALL_DIR/lib/libTHPP$LD_POSTFIX" \
              -DATEN_LIBRARIES="$INSTALL_DIR/lib/libATen$LD_POSTFIX" \
              -DTHNN_LIBRARIES="$INSTALL_DIR/lib/libTHNN$LD_POSTFIX" \
              -DTHCUNN_LIBRARIES="$INSTALL_DIR/lib/libTHCUNN$LD_POSTFIX" \
              -DTHS_LIBRARIES="$INSTALL_DIR/lib/libTHS$LD_POSTFIX" \
              -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
              -DTHCS_LIBRARIES="$INSTALL_DIR/lib/libTHCS$LD_POSTFIX" \
              -DTH_SO_VERSION=1 \
              -DTHC_SO_VERSION=1 \
              -DTHNN_SO_VERSION=1 \
              -DTHCUNN_SO_VERSION=1 \
              -DTHD_SO_VERSION=1 \
              -DNO_CUDA=$((1-$WITH_CUDA)) \
              -DNCCL_EXTERNAL=1 \
              -Dnanopb_BUILD_GENERATOR=0 \
              -DCMAKE_DEBUG_POSTFIX="" \
              -DCMAKE_BUILD_TYPE=$([ $DEBUG ] && echo Debug || echo Release) \
              ${@:2}
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  local lib_prefix=$INSTALL_DIR/lib/lib$1
  if [ -f "$lib_prefix$LD_POSTFIX" ]; then
    rm -rf -- "$lib_prefix$LD_POSTFIX_UNVERSIONED"
  fi

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
   ${CMAKE_VERSION} ../../nccl -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
               -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
               -DCMAKE_C_FLAGS="$C_FLAGS" \
               -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS"
   make install
   cp "lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so.1"
   if [ ! -f "${INSTALL_DIR}/lib/libnccl.so" ]; then
     ln -s "${INSTALL_DIR}/lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so"
   fi
   cd ../..
}

# In the torch/lib directory, create an installation directory
mkdir -p tmp_install

# Build
for arg in "$@"; do
    if [[ "$arg" == "nccl" ]]; then
        build_nccl
    elif [[ "$arg" == "gloo" ]]; then
        build gloo $GLOO_FLAGS
    else
        build $arg
    fi
done

# If all the builds succeed we copy the libraries, headers,
# binaries to torch/lib
rm -rf $INSTALL_DIR/lib/cmake
rm -rf $INSTALL_DIR/lib/python
cp $INSTALL_DIR/lib/* .
if [ -d "$INSTALL_DIR/lib64/" ]; then
    cp $INSTALL_DIR/lib64/* .
fi
cp THNN/generic/THNN.h .
cp THCUNN/generic/THCUNN.h .
cp -r $INSTALL_DIR/include .
if [ -d "$INSTALL_DIR/bin/" ]; then
    cp $INSTALL_DIR/bin/* .
fi

# this is for binary builds
if [[ $PYTORCH_BINARY_BUILD && $PYTORCH_SO_DEPS ]]
then
    echo "Copying over dependency libraries $PYTORCH_SO_DEPS"
    # copy over dependency libraries into the current dir
    cp $PYTORCH_SO_DEPS .
fi

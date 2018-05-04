#!/usr/bin/env bash

# Shell script used to build the aten/* and third_party/* dependencies prior to
# linking the libraries and passing the headers to the Python extension
# compilation stage. This file is used from setup.py, but can also be
# called standalone to compile the libraries outside of the overall PyTorch
# build process.
#
# TODO: Replace this with a CMakeLists.txt

set -ex

# Options for building only a subset of the libraries
WITH_CUDA=0
if [[ "$1" == "--with-cuda" ]]; then
  WITH_CUDA=1
  shift
fi

WITH_NNPACK=0
if [[ "$1" == "--with-nnpack" ]]; then
  WITH_NNPACK=1
  shift
fi

WITH_MKLDNN=0
if [[ "$1" == "--with-mkldnn" ]]; then
  WITH_MKLDNN=1
  shift
fi

WITH_GLOO_IBVERBS=0
if [[ "$1" == "--with-gloo-ibverbs" ]]; then
  WITH_GLOO_IBVERBS=1
  shift
fi

WITH_DISTRIBUTED_MW=0
if [[ "$1" == "--with-distributed-mw" ]]; then
  WITH_DISTRIBUTED_MW=1
  shift
fi

CMAKE_INSTALL=${CMAKE_INSTALL-make install}

# Save user specified env vars, we will manually propagate them
# to cmake.  We copy distutils semantics, referring to
# cpython/Lib/distutils/sysconfig.py as the source of truth
USER_CFLAGS=""
USER_LDFLAGS=""
if [[ -n "$LDFLAGS" ]]; then
  USER_LDFLAGS="$USER_LDFLAGS $LDFLAGS"
fi
if [[ -n "$CFLAGS" ]]; then
  USER_CFLAGS="$USER_CFLAGS $CFLAGS"
  USER_LDFLAGS="$USER_LDFLAGS $CFLAGS"
fi
if [[ -n "$CPPFLAGS" ]]; then
  # Unlike distutils, NOT modifying CXX
  USER_C_CFLAGS="$USER_CFLAGS $CPPFLAGS"
  USER_LDFLAGS="$USER_LDFLAGS $CPPFLAGS"
fi

cd "$(dirname "$0")/.."
PWD=`printf "%q\n" "$(pwd)"`
BASE_DIR="$PWD"
TORCH_LIB_DIR="$BASE_DIR/torch/lib"
INSTALL_DIR="$TORCH_LIB_DIR/tmp_install"
THIRD_PARTY_DIR="$BASE_DIR/third_party"

CMAKE_VERSION=${CMAKE_VERSION:="cmake"}
C_FLAGS=" -DTH_INDEX_BASE=0 -I\"$INSTALL_DIR/include\" \
  -I\"$INSTALL_DIR/include/TH\" -I\"$INSTALL_DIR/include/THC\" \
  -I\"$INSTALL_DIR/include/THS\" -I\"$INSTALL_DIR/include/THCS\" \
  -I\"$INSTALL_DIR/include/THNN\" -I\"$INSTALL_DIR/include/THCUNN\""
# Workaround OpenMPI build failure
# ImportError: /build/pytorch-0.2.0/.pybuild/pythonX.Y_3.6/build/torch/_C.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN3MPI8Datatype4FreeEv
# https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=686926
C_FLAGS="${C_FLAGS} -DOMPI_SKIP_MPICXX=1"
LDFLAGS="-L\"$INSTALL_DIR/lib\" "
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
THD_FLAGS=""
NCCL_ROOT_DIR=${NCCL_ROOT_DIR:-$INSTALL_DIR}
if [[ $WITH_CUDA -eq 1 ]]; then
    GLOO_FLAGS="-DUSE_CUDA=1 -DNCCL_ROOT_DIR=$NCCL_ROOT_DIR"
fi
# Gloo infiniband support
if [[ $WITH_GLOO_IBVERBS -eq 1 ]]; then
    GLOO_FLAGS+=" -DUSE_IBVERBS=1 -DBUILD_SHARED_LIBS=1"
    THD_FLAGS="-DWITH_GLOO_IBVERBS=1"
fi
if [[ $WITH_DISTRIBUTED_MW -eq 1 ]]; then
    THD_FLAGS+="-DWITH_DISTRIBUTED_MW=1"
fi
CWRAP_FILES="\
$BASE_DIR/torch/lib/ATen/Declarations.cwrap;\
$BASE_DIR/torch/lib/THNN/generic/THNN.h;\
$BASE_DIR/torch/lib/THCUNN/generic/THCUNN.h;\
$BASE_DIR/torch/lib/ATen/nn.yaml"
CUDA_NVCC_FLAGS=$C_FLAGS
if [[ $CUDA_DEBUG -eq 1 ]]; then
  CUDA_NVCC_FLAGS="$CUDA_NVCC_FLAGS -g -G"
fi
if [ -z "$NUM_JOBS" ]; then
  NUM_JOBS="$(getconf _NPROCESSORS_ONLN)"
fi

BUILD_TYPE="Release"
if [[ "$DEBUG" ]]; then
  BUILD_TYPE="Debug"
elif [[ "$REL_WITH_DEB_INFO" ]]; then
  BUILD_TYPE="RelWithDebInfo"
fi

echo "Building in $BUILD_TYPE mode"

# Used to build an individual library
function build() {
  # We create a build directory for the library, which will
  # contain the cmake output
  mkdir -p build/$1
  pushd build/$1
  BUILD_C_FLAGS=''
  case $1 in
      THCS | THCUNN ) BUILD_C_FLAGS=$C_FLAGS;;
      nanopb ) BUILD_C_FLAGS=$C_FLAGS" -fPIC -fexceptions";;
      *) BUILD_C_FLAGS=$C_FLAGS" -fexceptions";;
  esac
  ${CMAKE_VERSION} ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              ${CMAKE_GENERATOR} \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$BUILD_C_FLAGS $USER_CFLAGS" \
              -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS $USER_CFLAGS" \
              -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS $USER_LDFLAGS" \
              -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS $USER_LDFLAGS" \
              -DCMAKE_INSTALL_LIBDIR="$INSTALL_DIR/lib" \
              -DCUDA_NVCC_FLAGS="$CUDA_NVCC_FLAGS" \
              -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
              -Dcwrap_files="$CWRAP_FILES" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
              -DTH_LIB_PATH="$INSTALL_DIR/lib" \
              -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
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
              -DNO_NNPACK=$((1-$WITH_NNPACK)) \
              -DNCCL_EXTERNAL=1 \
              -Dnanopb_BUILD_GENERATOR=0 \
              -DCMAKE_DEBUG_POSTFIX="" \
              -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
              ${@:2} \
              -DCMAKE_EXPORT_COMPILE_COMMANDS=1
  ${CMAKE_INSTALL} -j"$NUM_JOBS"
  popd

  local lib_prefix=$INSTALL_DIR/lib/lib$1
  if [ -f "$lib_prefix$LD_POSTFIX" ]; then
    rm -rf -- "$lib_prefix$LD_POSTFIX_UNVERSIONED"
  fi

  if [[ $(uname) == 'Darwin' ]]; then
    pushd "$INSTALL_DIR/lib"
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    popd
  fi
}

function build_nccl() {
  mkdir -p build/nccl
  pushd build/nccl
  ${CMAKE_VERSION} ../../nccl -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              ${CMAKE_GENERATOR} \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$C_FLAGS $USER_CFLAGS" \
              -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS $USER_CFLAGS" \
              -DCMAKE_SHARED_LINKER_FLAGS="$USER_LDFLAGS"
  ${CMAKE_INSTALL}
  mkdir -p ${INSTALL_DIR}/lib
  cp "lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so.1"
  if [ ! -f "${INSTALL_DIR}/lib/libnccl.so" ]; then
    ln -s "${INSTALL_DIR}/lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so"
  fi
  popd
}

# purpusefully not using build() because we need ATen to build the same
# regardless of whether it is inside pytorch or not, so it
# cannot take any special flags
# special flags need to be part of the ATen build itself
#
# However, we do explicitly pass library paths when setup.py has already
# detected them (to ensure that we have a consistent view between the
# PyTorch and ATen builds.)
function build_aten() {
  mkdir -p build
  pushd build
  ${CMAKE_VERSION} .. \
  ${CMAKE_GENERATOR} \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DNO_CUDA=$((1-$WITH_CUDA)) \
      -DNO_NNPACK=$((1-$WITH_NNPACK)) \
      -DCUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR \
      -DCUDNN_LIB_DIR=$CUDNN_LIB_DIR \
      -DCUDNN_LIBRARY=$CUDNN_LIBRARY \
      -DNO_MKLDNN=$((1-$WITH_MKLDNN)) \
      -DMKLDNN_INCLUDE_DIR=$MKLDNN_INCLUDE_DIR \
      -DMKLDNN_LIB_DIR=$MKLDNN_LIB_DIR \
      -DMKLDNN_LIBRARY=$MKLDNN_LIBRARY \
      -DATEN_NO_CONTRIB=1 \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -DCMAKE_C_FLAGS="$USER_CFLAGS" \
      -DCMAKE_CXX_FLAGS="$USER_CFLAGS" \
      -DCMAKE_EXE_LINKER_FLAGS="$USER_LDFLAGS" \
      -DCMAKE_SHARED_LINKER_FLAGS="$USER_LDFLAGS"
      # STOP!!! Are you trying to add a C or CXX flag?  Add it
      # to aten/CMakeLists.txt, not here.  We need the vanilla
      # cmake build to work.
  ${CMAKE_INSTALL} -j"$NUM_JOBS"
  popd
}

# In the torch/lib directory, create an installation directory
mkdir -p torch/lib/tmp_install

# Build
for arg in "$@"; do
    if [[ "$arg" == "nccl" ]]; then
        pushd $THIRD_PARTY_DIR
        build_nccl
        popd
    elif [[ "$arg" == "gloo" ]]; then
        pushd "$THIRD_PARTY_DIR"
        build gloo $GLOO_FLAGS
        popd
    elif [[ "$arg" == "ATen" ]]; then
        pushd "$BASE_DIR/aten"
        build_aten
        popd
    elif [[ "$arg" == "THD" ]]; then
        pushd "$TORCH_LIB_DIR"
        build THD $THD_FLAGS
        popd
    elif [[ "$arg" == "libshm" ]] || [[ "$arg" == "libshm_windows" ]]; then
        pushd "$TORCH_LIB_DIR"
        build $arg
        popd
    else
        pushd "$THIRD_PARTY_DIR"
        build $arg
        popd
    fi
done

pushd torch/lib

# If all the builds succeed we copy the libraries, headers,
# binaries to torch/lib
rm -rf "$INSTALL_DIR/lib/cmake"
rm -rf "$INSTALL_DIR/lib/python"
cp "$INSTALL_DIR/lib"/* .
if [ -d "$INSTALL_DIR/lib64/" ]; then
    cp "$INSTALL_DIR/lib64"/* .
fi
cp ../../aten/src/THNN/generic/THNN.h .
cp ../../aten/src/THCUNN/generic/THCUNN.h .
cp -r "$INSTALL_DIR/include" .
if [ -d "$INSTALL_DIR/bin/" ]; then
    cp "$INSTALL_DIR/bin/"/* .
fi

popd

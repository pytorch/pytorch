#!/usr/bin/env bash

# Shell script used to build the aten/*, caffe2/*, and third_party/*
# dependencies prior to linking libraries and passing headers to the Python
# extension compilation stage. This file is used from setup.py, but can also be
# called standalone to compile the libraries outside of the overall PyTorch
# build process.
#
# TODO: Replace this with the root-level CMakeLists.txt

set -ex

# Options for building only a subset of the libraries
USE_CUDA=0
USE_ROCM=0
USE_NNPACK=0
USE_MKLDNN=0
USE_GLOO_IBVERBS=0
USE_DISTRIBUTED_MW=0
FULL_CAFFE2=0
while [[ $# -gt 0 ]]; do
    case "$1" in
      --use-cuda)
          USE_CUDA=1
          ;;
      --use-rocm)
          USE_ROCM=1
          ;;
      --use-nnpack)
          USE_NNPACK=1
          ;;
      --use-mkldnn)
          USE_MKLDNN=1
          ;;
      --use-gloo-ibverbs)
          USE_GLOO_IBVERBS=1
          ;;
      --use-distributed-mw)
          USE_DISTRIBUTED_MW=1
          ;;
      --full-caffe2)
          FULL_CAFFE2=1
          ;;
      --cuda-static-link)
          CAFFE2_STATIC_LINK_CUDA=1
          ;;
      *)
          break
          ;;
    esac
    shift
done

CMAKE_INSTALL=${CMAKE_INSTALL-make install}

BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS-ON}

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

# Use ccache if available (this path is where Homebrew installs ccache symlinks)
if [ "$(uname)" == 'Darwin' ]; then
  if [ -d '/usr/local/opt/ccache/libexec' ]; then
    CCACHE_WRAPPER_PATH=/usr/local/opt/ccache/libexec
  fi
fi

BASE_DIR=$(cd $(dirname "$0")/.. && printf "%q\n" "$(pwd)")
TORCH_LIB_DIR="$BASE_DIR/torch/lib"
INSTALL_DIR="$TORCH_LIB_DIR/tmp_install"
THIRD_PARTY_DIR="$BASE_DIR/third_party"

CMAKE_VERSION=${CMAKE_VERSION:="cmake"}
C_FLAGS=" -I\"$INSTALL_DIR/include\" \
  -I\"$INSTALL_DIR/include/TH\" -I\"$INSTALL_DIR/include/THC\" \
  -I\"$INSTALL_DIR/include/THS\" -I\"$INSTALL_DIR/include/THCS\" \
  -I\"$INSTALL_DIR/include/THNN\" -I\"$INSTALL_DIR/include/THCUNN\""
# Workaround OpenMPI build failure
# ImportError: /build/pytorch-0.2.0/.pybuild/pythonX.Y_3.6/build/torch/_C.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN3MPI8Datatype4FreeEv
# https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=686926
C_FLAGS="${C_FLAGS} -DOMPI_SKIP_MPICXX=1"
LDFLAGS="-L\"$INSTALL_DIR/lib\" "
LD_POSTFIX=".so"
if [[ $(uname) == 'Darwin' ]]; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,@loader_path"
    LD_POSTFIX=".dylib"
else
    if [[ $USE_ROCM -eq 1 ]]; then
        LDFLAGS="$LDFLAGS -Wl,-rpath,\\\\\\\$ORIGIN"
    else
        LDFLAGS="$LDFLAGS -Wl,-rpath,\$ORIGIN"
    fi
fi
CPP_FLAGS=" -std=c++11 "
GLOO_FLAGS=""
THD_FLAGS=""
NCCL_ROOT_DIR=${NCCL_ROOT_DIR:-$INSTALL_DIR}
if [[ $USE_CUDA -eq 1 ]]; then
    GLOO_FLAGS="-DUSE_CUDA=1 -DNCCL_ROOT_DIR=$NCCL_ROOT_DIR"
fi
# Gloo infiniband support
if [[ $USE_GLOO_IBVERBS -eq 1 ]]; then
    GLOO_FLAGS+=" -DUSE_IBVERBS=1 -DBUILD_SHARED_LIBS=1"
    THD_FLAGS="-DUSE_GLOO_IBVERBS=1"
fi
if [[ $USE_DISTRIBUTED_MW -eq 1 ]]; then
    THD_FLAGS+="-DUSE_DISTRIBUTED_MW=1"
fi
CWRAP_FILES="\
$BASE_DIR/torch/lib/ATen/Declarations.cwrap;\
$BASE_DIR/torch/lib/THNN/generic/THNN.h;\
$BASE_DIR/torch/lib/THCUNN/generic/THCUNN.h;\
$BASE_DIR/torch/lib/ATen/nn.yaml"
CUDA_NVCC_FLAGS=$C_FLAGS
if [[ -z "$CUDA_DEVICE_DEBUG" ]]; then
  CUDA_DEVICE_DEBUG=0
fi
if [ -z "$MAX_JOBS" ]; then
  MAX_JOBS="$(getconf _NPROCESSORS_ONLN)"
fi

BUILD_TYPE="Release"
if [[ -n "$DEBUG" && $DEBUG -ne 0 ]]; then
  BUILD_TYPE="Debug"
elif [[ -n "$REL_WITH_DEB_INFO" && $REL_WITH_DEB_INFO -ne 0 ]]; then
  BUILD_TYPE="RelWithDebInfo"
fi

echo "Building in $BUILD_TYPE mode"

# Used to build an individual library
function build() {
  if [[ -z "$CMAKE_ARGS" ]]; then
    CMAKE_ARGS=()
  fi
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
  # TODO: The *_LIBRARIES cmake variables should eventually be
  # deprecated because we are using .cmake files to handle finding
  # installed libraries instead
  ${CMAKE_VERSION} ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/Modules_CUDA_fix" \
              ${CMAKE_GENERATOR} \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$BUILD_C_FLAGS $USER_CFLAGS" \
              -DCMAKE_CXX_FLAGS="$BUILD_C_FLAGS $CPP_FLAGS $USER_CFLAGS" \
              -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS $USER_LDFLAGS" \
              -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS $USER_LDFLAGS" \
              -DCMAKE_INSTALL_LIBDIR="$INSTALL_DIR/lib" \
              -DCUDA_NVCC_FLAGS="$CUDA_NVCC_FLAGS" \
              -DCUDA_DEVICE_DEBUG=$CUDA_DEVICE_DEBUG \
              -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
              -Dcwrap_files="$CWRAP_FILES" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
              -DTH_LIB_PATH="$INSTALL_DIR/lib" \
              -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
              -DCAFFE2_LIBRARIES="$INSTALL_DIR/lib/libcaffe2$LD_POSTFIX" \
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
              -DUSE_CUDA=$USE_CUDA \
              -DNO_NNPACK=$((1-$USE_NNPACK)) \
              -DNCCL_EXTERNAL=1 \
              -Dnanopb_BUILD_GENERATOR=0 \
              -DCMAKE_DEBUG_POSTFIX="" \
              -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
              ${@:2} \
              -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ${CMAKE_ARGS[@]}
  ${CMAKE_INSTALL} -j"$MAX_JOBS"
  popd

  # Fix rpaths of shared libraries
  if [[ $(uname) == 'Darwin' ]]; then
    pushd "$INSTALL_DIR/lib"
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    popd
  fi
}

function path_remove {
  # Delete path by parts so we can never accidentally remove sub paths
  PATH=${PATH//":$1:"/":"} # delete any instances in the middle
  PATH=${PATH/#"$1:"/} # delete any instance at the beginning
  PATH=${PATH/%":$1"/} # delete any instance in the at the end
}

function build_nccl() {
  mkdir -p build/nccl
  pushd build/nccl
  ${CMAKE_VERSION} ../../nccl -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/Modules_CUDA_fix" \
              ${CMAKE_GENERATOR} \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$C_FLAGS $USER_CFLAGS" \
              -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS $USER_CFLAGS" \
              -DCMAKE_SHARED_LINKER_FLAGS="$USER_LDFLAGS" \
              -DCMAKE_UTILS_PATH="$BASE_DIR/cmake/public/utils.cmake" \
              -DNUM_JOBS="$MAX_JOBS"
  ${CMAKE_INSTALL} -j"$MAX_JOBS"
  mkdir -p ${INSTALL_DIR}/lib
  cp "lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so.1"
  if [ ! -f "${INSTALL_DIR}/lib/libnccl.so" ]; then
    ln -s "${INSTALL_DIR}/lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so"
  fi
  popd
}

# purposefully not using build() because we need Caffe2 to build the same
# regardless of whether it is inside PyTorch or not, so it
# cannot take any special flags
# special flags need to be part of the Caffe2 build itself
#
# However, we do explicitly pass library paths when setup.py has already
# detected them (to ensure that we have a consistent view between the
# PyTorch and Caffe2 builds.)
function build_caffe2() {
  if [[ -z $EXTRA_CAFFE2_CMAKE_FLAGS ]]; then
    EXTRA_CAFFE2_CMAKE_FLAGS=()
  fi
  if [[ -n $CCACHE_WRAPPER_PATH ]]; then
    EXTRA_CAFFE2_CMAKE_FLAGS+=("-DCMAKE_C_COMPILER=$CCACHE_WRAPPER_PATH/gcc")
    EXTRA_CAFFE2_CMAKE_FLAGS+=("-DCMAKE_CXX_COMPILER=$CCACHE_WRAPPER_PATH/g++")
  fi
  if [[ -n $CMAKE_PREFIX_PATH ]]; then
    EXTRA_CAFFE2_CMAKE_FLAGS+=("-DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH")
  fi

  ${CMAKE_VERSION} $BASE_DIR \
  ${CMAKE_GENERATOR} \
      -DPYTHON_EXECUTABLE=$PYTORCH_PYTHON \
      -DBUILDING_WITH_TORCH_LIBS=ON \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DBUILD_CAFFE2=$FULL_CAFFE2 \
      -DBUILD_TORCH=$BUILD_TORCH \
      -DBUILD_PYTHON=$FULL_CAFFE2 \
      -DBUILD_BINARY=OFF \
      -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
      -DONNX_NAMESPACE=$ONNX_NAMESPACE \
      -DUSE_CUDA=$USE_CUDA \
      -DCAFFE2_STATIC_LINK_CUDA=$CAFFE2_STATIC_LINK_CUDA \
      -DUSE_ROCM=$USE_ROCM \
      -DUSE_NNPACK=$USE_NNPACK \
      -DCUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR \
      -DCUDNN_LIB_DIR=$CUDNN_LIB_DIR \
      -DCUDNN_LIBRARY=$CUDNN_LIBRARY \
      -DUSE_MKLDNN=$USE_MKLDNN \
      -DMKLDNN_INCLUDE_DIR=$MKLDNN_INCLUDE_DIR \
      -DMKLDNN_LIB_DIR=$MKLDNN_LIB_DIR \
      -DMKLDNN_LIBRARY=$MKLDNN_LIBRARY \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -DCMAKE_C_FLAGS="$USER_CFLAGS" \
      -DCMAKE_CXX_FLAGS="$USER_CFLAGS" \
      -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS $USER_LDFLAGS" \
      -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS $USER_LDFLAGS" ${EXTRA_CAFFE2_CMAKE_FLAGS[@]}
      # STOP!!! Are you trying to add a C or CXX flag?  Add it
      # to CMakeLists.txt and aten/CMakeLists.txt, not here.
      # We need the vanilla cmake build to work.
  ${CMAKE_INSTALL} -j"$MAX_JOBS"

  # Install Python proto files
  if [[ $FULL_CAFFE2 -ne 0 ]]; then
    find . -name proto
    for proto_file in ./caffe2/proto/*.py; do
      cp $proto_file "../caffe2/proto/"
    done
  fi

  # Fix rpaths of shared libraries
  if [[ $(uname) == 'Darwin' ]]; then
    pushd "$INSTALL_DIR/lib"
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    popd
  fi
}

# In the torch/lib directory, create an installation directory
mkdir -p $INSTALL_DIR

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
    elif [[ "$arg" == "caffe2" ]]; then
        build_caffe2
    elif [[ "$arg" == "THD" ]]; then
        pushd "$TORCH_LIB_DIR"
        build THD $THD_FLAGS
        popd
    elif [[ "$arg" == "libshm" ]] || [[ "$arg" == "libshm_windows" ]]; then
        pushd "$TORCH_LIB_DIR"
        build $arg
        popd
    elif [[ "$arg" == "c10d" ]]; then
        pushd "$TORCH_LIB_DIR"
        build c10d
        popd
    else
        pushd "$THIRD_PARTY_DIR"
        build $arg
        popd
    fi
done

pushd $TORCH_LIB_DIR

# If all the builds succeed we copy the libraries, headers,
# binaries to torch/lib
rm -rf "$INSTALL_DIR/lib/cmake"
rm -rf "$INSTALL_DIR/lib/python"
cp -r "$INSTALL_DIR/lib"/* .
if [ -d "$INSTALL_DIR/lib64/" ]; then
    cp -r "$INSTALL_DIR/lib64"/* .
fi
cp ../../aten/src/THNN/generic/THNN.h .
cp ../../aten/src/THCUNN/generic/THCUNN.h .
cp -r "$INSTALL_DIR/include" .
if [ -d "$INSTALL_DIR/bin/" ]; then
    cp -r "$INSTALL_DIR/bin/"/* .
fi

popd

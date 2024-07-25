#!/bin/bash -xe
##############################################################################
# Example command to build the iOS target.
##############################################################################
#
# This script shows how one can build a Caffe2 binary for the iOS platform
# using ios-cmake. This is very similar to the android-cmake - see
# build_android.sh for more details.

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"

if [ -z "$PYTHON" ]; then
  PYTHON=python
  PYTHON_VERSION_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info[0])')
  if [ "${PYTHON_VERSION_MAJOR}" -le 2 ]; then
    echo "Default python executable is Python-2, trying to use python3 alias"
    PYTHON=python3
  fi
fi

echo "Bash: $(/bin/bash --version | head -1)"
echo "Python: $($PYTHON -c 'import sys; print(sys.version)')"
echo "Caffe2 path: $CAFFE2_ROOT"

CMAKE_ARGS=()

# Build PyTorch mobile
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("purelib"))')")
CMAKE_ARGS+=("-DPython_EXECUTABLE=$($PYTHON -c 'import sys; print(sys.executable)')")
CMAKE_ARGS+=("-DBUILD_CUSTOM_PROTOBUF=OFF")

# custom build with selected ops
if [ -n "${SELECTED_OP_LIST}" ]; then
  SELECTED_OP_LIST="$(cd $(dirname $SELECTED_OP_LIST); pwd -P)/$(basename $SELECTED_OP_LIST)"
  echo "Choose SELECTED_OP_LIST file: $SELECTED_OP_LIST"
  if [ ! -r ${SELECTED_OP_LIST} ]; then
    echo "Error: SELECTED_OP_LIST file ${SELECTED_OP_LIST} not found."
    exit 1
  fi
  CMAKE_ARGS+=("-DSELECTED_OP_LIST=${SELECTED_OP_LIST}")
fi

# bitcode
if [ "${ENABLE_BITCODE:-}" == '1' ]; then
  CMAKE_ARGS+=("-DCMAKE_C_FLAGS=-fembed-bitcode")
  CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-fembed-bitcode")
fi

# Use ios-cmake to build iOS project from CMake.
# This projects sets CMAKE_C_COMPILER to /usr/bin/gcc and
# CMAKE_CXX_COMPILER to /usr/bin/g++. In order to use ccache (if it is available) we
# must override these variables via CMake arguments.
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$CAFFE2_ROOT/cmake/iOS.cmake")
if [ -n "${CCACHE_WRAPPER_PATH:-}"]; then
  CCACHE_WRAPPER_PATH=/usr/local/opt/ccache/libexec
fi
if [ -d "$CCACHE_WRAPPER_PATH" ]; then
  CMAKE_ARGS+=("-DCMAKE_C_COMPILER=$CCACHE_WRAPPER_PATH/gcc")
  CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$CCACHE_WRAPPER_PATH/g++")
fi

# IOS_PLATFORM controls type of iOS platform (see ios-cmake)
if [ -n "${IOS_PLATFORM:-}" ]; then
  CMAKE_ARGS+=("-DIOS_PLATFORM=${IOS_PLATFORM}")
  if [ "${IOS_PLATFORM}" == "WATCHOS" ]; then
      # enable bitcode by default for watchos
      CMAKE_ARGS+=("-DCMAKE_C_FLAGS=-fembed-bitcode")
      CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-fembed-bitcode")
      # disable the QNNPACK
      CMAKE_ARGS+=("-DUSE_PYTORCH_QNNPACK=OFF")
  fi
else
  # IOS_PLATFORM is not set, default to OS, which builds iOS.
  CMAKE_ARGS+=("-DIOS_PLATFORM=OS")
fi

if [ -n "${IOS_ARCH:-}" ]; then
  CMAKE_ARGS+=("-DIOS_ARCH=${IOS_ARCH}")
fi

if [ "${BUILD_LITE_INTERPRETER}" == 0 ]; then
  CMAKE_ARGS+=("-DBUILD_LITE_INTERPRETER=OFF")
else
  CMAKE_ARGS+=("-DBUILD_LITE_INTERPRETER=ON")
fi
if [ "${TRACING_BASED}" == 1 ]; then
  CMAKE_ARGS+=("-DTRACING_BASED=ON")
else
  CMAKE_ARGS+=("-DTRACING_BASED=OFF")
fi
if [ "${USE_LIGHTWEIGHT_DISPATCH}" == 1 ]; then
  CMAKE_ARGS+=("-DUSE_LIGHTWEIGHT_DISPATCH=ON")
  CMAKE_ARGS+=("-DSTATIC_DISPATCH_BACKEND=CPU")
else
  CMAKE_ARGS+=("-DUSE_LIGHTWEIGHT_DISPATCH=OFF")
fi

CMAKE_ARGS+=("-DUSE_LITE_INTERPRETER_PROFILER=OFF")

# Don't build binaries or tests (only the library)
CMAKE_ARGS+=("-DBUILD_TEST=OFF")
CMAKE_ARGS+=("-DBUILD_BINARY=OFF")
CMAKE_ARGS+=("-DBUILD_PYTHON=OFF")

# Disable unused dependencies
CMAKE_ARGS+=("-DUSE_CUDA=OFF")
CMAKE_ARGS+=("-DUSE_ITT=OFF")
CMAKE_ARGS+=("-DUSE_GFLAGS=OFF")
CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
CMAKE_ARGS+=("-DUSE_MPI=OFF")
CMAKE_ARGS+=("-DUSE_NUMPY=OFF")
CMAKE_ARGS+=("-DUSE_NNPACK=OFF")
CMAKE_ARGS+=("-DUSE_MKLDNN=OFF")

# Metal
if [ "${USE_PYTORCH_METAL:-}" == "1" ]; then
  CMAKE_ARGS+=("-DUSE_PYTORCH_METAL=ON")
fi

# Core ML
if [ "${USE_COREML_DELEGATE}" == "1" ]; then
  CMAKE_ARGS+=("-DUSE_COREML_DELEGATE=ON")
fi

# pthreads
CMAKE_ARGS+=("-DCMAKE_THREAD_LIBS_INIT=-lpthread")
CMAKE_ARGS+=("-DCMAKE_HAVE_THREADS_LIBRARY=1")
CMAKE_ARGS+=("-DCMAKE_USE_PTHREADS_INIT=1")

# Only toggle if VERBOSE=1
if [ "${VERBOSE:-}" == '1' ]; then
  CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=1")
fi

# enable ARC
CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-fobjc-arc")

# Now, actually build the iOS target.
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build_ios"}
INSTALL_PREFIX=${BUILD_ROOT}/install
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT
cmake "$CAFFE2_ROOT" \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DBUILD_SHARED_LIBS=OFF \
    ${CMAKE_ARGS[@]} \
    $@

cmake --build . -- "-j$(sysctl -n hw.ncpu)"

# copy headers and libs to install directory
echo "Will install headers and libs to $INSTALL_PREFIX for further Xcode project usage."
make install
echo "Installation completed, now you can copy the headers/libs from $INSTALL_PREFIX to your Xcode project directory."

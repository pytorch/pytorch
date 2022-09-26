#!/bin/bash
##############################################################################
# Example command to build the android target.
##############################################################################
#
# This script shows how one can build a Caffe2 binary for the Android platform
# using android-cmake. A few notes:
#
# (1) This build also does a host build for protobuf. You will need autoconf
#     to carry out this. If autoconf is not possible, you will need to provide
#     a pre-built protoc binary that is the same version as the protobuf
#     version under third_party.
#     If you are building on Mac, you might need to install autotool and
#     libtool. The easiest way is via homebrew:
#         brew install automake
#         brew install libtool
# (2) You will need to have android ndk installed. The current script assumes
#     that you set ANDROID_NDK to the location of ndk.
# (3) The toolchain and the build target platform can be specified with the
#     cmake arguments below. For more details, check out android-cmake's doc.

set -e

# Android specific flags
if [ -z "$ANDROID_ABI" ]; then
  ANDROID_ABI="armeabi-v7a with NEON"
fi
ANDROID_NATIVE_API_LEVEL="21"
echo "Build with ANDROID_ABI[$ANDROID_ABI], ANDROID_NATIVE_API_LEVEL[$ANDROID_NATIVE_API_LEVEL]"

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
if [ -z "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not set; please set it to the Android NDK directory"
  exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not a directory; did you install it under $ANDROID_NDK?"
  exit 1
fi

if [ -z "$PYTHON" ]; then
  PYTHON=python
  PYTHON_VERSION_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info[0])')
  if [ "${PYTHON_VERSION_MAJOR}" -le 2 ]; then
    echo "Default python executable is Python-2, trying to use python3 alias"
    PYTHON=python3
  fi
fi

ANDROID_NDK_PROPERTIES="$ANDROID_NDK/source.properties"
[ -f "$ANDROID_NDK_PROPERTIES" ] && ANDROID_NDK_VERSION=$(sed -n 's/^Pkg.Revision[^=]*= *\([0-9]*\)\..*$/\1/p' "$ANDROID_NDK_PROPERTIES")

echo "Bash: $(/bin/bash --version | head -1)"
echo "Python: $($PYTHON -c 'import sys; print(sys.version)')"
echo "Caffe2 path: $CAFFE2_ROOT"
echo "Using Android NDK at $ANDROID_NDK"
echo "Android NDK version: $ANDROID_NDK_VERSION"

CMAKE_ARGS=()

# Build PyTorch mobile
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("purelib"))')")
CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=$($PYTHON -c 'import sys; print(sys.executable)')")
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

# If Ninja is installed, prefer it to Make
if [ -x "$(command -v ninja)" ]; then
  CMAKE_ARGS+=("-GNinja")
fi

# Use android-cmake to build Android project from CMake.
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake")

if [ -z "$BUILD_MOBILE_BENCHMARK" ]; then
  BUILD_MOBILE_BENCHMARK=0
fi

if [ -z "$BUILD_MOBILE_TEST" ]; then
  BUILD_MOBILE_TEST=0
fi
# Don't build artifacts we don't need
CMAKE_ARGS+=("-DBUILD_TEST=OFF")
CMAKE_ARGS+=("-DBUILD_BINARY=OFF")

# If there exists env variable and it equals to 0, build full jit interpreter.
# Default behavior is to build lite interpreter
# cmd:  BUILD_LITE_INTERPRETER=0 ./scripts/build_android.sh
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

CMAKE_ARGS+=("-DBUILD_MOBILE_BENCHMARK=$BUILD_MOBILE_BENCHMARK")
CMAKE_ARGS+=("-DBUILD_MOBILE_TEST=$BUILD_MOBILE_TEST")
CMAKE_ARGS+=("-DBUILD_PYTHON=OFF")
CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=OFF")
if (( "${ANDROID_NDK_VERSION:-0}" < 18 )); then
  CMAKE_ARGS+=("-DANDROID_TOOLCHAIN=gcc")
else
  CMAKE_ARGS+=("-DANDROID_TOOLCHAIN=clang")
fi
# Disable unused dependencies
CMAKE_ARGS+=("-DUSE_CUDA=OFF")
CMAKE_ARGS+=("-DUSE_ITT=OFF")
CMAKE_ARGS+=("-DUSE_GFLAGS=OFF")
CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
CMAKE_ARGS+=("-DUSE_LMDB=OFF")
CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")
CMAKE_ARGS+=("-DUSE_MPI=OFF")
CMAKE_ARGS+=("-DUSE_OPENMP=OFF")
# Only toggle if VERBOSE=1
if [ "${VERBOSE:-}" == '1' ]; then
  CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=1")
fi

# Android specific flags
CMAKE_ARGS+=("-DANDROID_NDK=$ANDROID_NDK")
CMAKE_ARGS+=("-DANDROID_ABI=$ANDROID_ABI")
CMAKE_ARGS+=("-DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL")
CMAKE_ARGS+=("-DANDROID_CPP_FEATURES=rtti exceptions")
if [ "${ANDROID_STL_SHARED:-}" == '1' ]; then
  CMAKE_ARGS+=("-DANDROID_STL=c++_shared")
fi
if [ "${ANDROID_DEBUG_SYMBOLS:-}" == '1' ]; then
  CMAKE_ARGS+=("-DANDROID_DEBUG_SYMBOLS=1")
fi

if [ -n "${USE_VULKAN}" ]; then
  CMAKE_ARGS+=("-DUSE_VULKAN=ON")
  if [ -n "${USE_VULKAN_FP16_INFERENCE}" ]; then
    CMAKE_ARGS+=("-DUSE_VULKAN_FP16_INFERENCE=ON")
  fi
  if [ -n "${USE_VULKAN_RELAXED_PRECISION}" ]; then
    CMAKE_ARGS+=("-DUSE_VULKAN_RELAXED_PRECISION=ON")
  fi
  if [ -n "${USE_VULKAN_SHADERC_RUNTIME}" ]; then
    CMAKE_ARGS+=("-DUSE_VULKAN_SHADERC_RUNTIME=ON")
  fi
fi

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

# Now, actually build the Android target.
BUILD_ROOT=${BUILD_ROOT:-"$CAFFE2_ROOT/build_android"}
INSTALL_PREFIX=${BUILD_ROOT}/install
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT
cmake "$CAFFE2_ROOT" \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ -z "$MAX_JOBS" ]; then
  if [ "$(uname)" == 'Darwin' ]; then
    MAX_JOBS=$(sysctl -n hw.ncpu)
  else
    MAX_JOBS=$(nproc)
  fi
fi

echo "Will install headers and libs to $INSTALL_PREFIX for further Android project usage."
cmake --build . --target install -- "-j${MAX_JOBS}"
echo "Installation completed, now you can copy the headers/libs from $INSTALL_PREFIX to your Android project directory."

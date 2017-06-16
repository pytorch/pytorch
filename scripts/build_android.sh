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

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "Caffe2 codebase root is: $CAFFE2_ROOT"

if [ -z "$ANDROID_NDK" ]; then
    echo "Did you set ANDROID_NDK variable?"
    exit 1
fi

if [ -d "$ANDROID_NDK" ]; then
    echo "Using Android ndk at $ANDROID_NDK"
else
    echo "Cannot find ndk: did you install it under $ANDROID_NDK?"
    exit 1
fi
# We are going to build the target into build_android.
BUILD_ROOT=$CAFFE2_ROOT/build_android
mkdir -p $BUILD_ROOT
echo "Build Caffe2 Android into: $BUILD_ROOT"

# Build protobuf from third_party so we have a host protoc binary.
echo "Building protoc"
$CAFFE2_ROOT/scripts/build_host_protoc.sh || exit 1

# Now, actually build the android target.
echo "Building caffe2"
cd $BUILD_ROOT

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../third_party/android-cmake/android.toolchain.cmake \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DANDROID_NDK=$ANDROID_NDK \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DANDROID_NATIVE_API_LEVEL=21 \
    -DUSE_CUDA=OFF \
    -DBUILD_TEST=OFF \
    -DUSE_LMDB=OFF \
    -DUSE_LEVELDB=OFF \
    -DBUILD_PYTHON=OFF \
    -DPROTOBUF_PROTOC_EXECUTABLE=$CAFFE2_ROOT/build_host_protoc/bin/protoc \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DUSE_MPI=OFF \
    -DUSE_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_FLAGS_RELEASE=-s \
    -DUSE_OPENCV=OFF \
    $@ \
    || exit 1

# Cross-platform parallel build
if [ "$(uname)" = 'Darwin' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
    cmake --build . -- "-j$(nproc)"
fi

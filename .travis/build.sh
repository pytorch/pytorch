#!/bin/bash
set -e
set -x

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$(dirname "$LOCAL_DIR")
cd "$ROOT_DIR"

mkdir build
cd build

# Special cases - run script and exit
if [ "$BUILD_ANDROID" = 'true' ]; then
    export ANDROID_NDK=/opt/android_ndk
    sh "${ROOT_DIR}/scripts/build_android.sh"
    exit 0
fi
if [ "$BUILD_IOS" = 'true' ]; then
    sh "${ROOT_DIR}/scripts/build_ios.sh" -DCMAKE_OSX_ARCHITECTURES=arm64
    exit 0
fi

# Configure
CMAKE_ARGS=('-DCMAKE_VERBOSE_MAKEFILE=ON')
if [ "$BUILD_CUDA" = 'true' ]; then
    CMAKE_ARGS+=('-DUSE_CUDA=ON')
    CMAKE_ARGS+=('-DCUDA_ARCH_NAME=Pascal')
    CMAKE_ARGS+=('-DCUDA_NVCC_EXECUTABLE=/usr/local/bin/nvcc')
    export PATH="/usr/local/cuda/bin:${PATH}"
    CMAKE_ARGS+=('-DUSE_NNPACK=OFF')
else
    CMAKE_ARGS+=('-DUSE_CUDA=OFF')
fi
if [ "$BUILD_MKL" = 'true' ]; then
    CMAKE_ARGS+=('-DBLAS=MKL')
fi
if [ "$BUILD_TESTS" = 'false' ]; then
    CMAKE_ARGS+=('-DBUILD_TEST=OFF')
fi
cmake .. ${CMAKE_ARGS[*]}

# Build
if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    cmake --build . -- "-j$(nproc)"
elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    cmake --build . -- "-j$(sysctl -n hw.ncpu)"
fi

#!/bin/bash
# This script should be sourced, not executed
set -e

export BUILD_ANDROID=false
export BUILD_CUDA=false
export BUILD_GCC5=false
export BUILD_IOS=false
export BUILD_MKL=false
export BUILD_NNPACK=true
export BUILD_TESTS=true

if [ "$BUILD" = 'linux' ]; then
    :
elif [ "$BUILD" = 'linux-gcc5' ]; then
    export BUILD_GCC5=true
elif [ "$BUILD" = 'linux-cuda' ]; then
    export BUILD_CUDA=true
    export BUILD_NNPACK=false
    export BUILD_TESTS=false
elif [ "$BUILD" = 'linux-mkl' ]; then
    export BUILD_MKL=true
elif [ "$BUILD" = 'linux-android' ]; then
    export BUILD_ANDROID=true
    export BUILD_TESTS=false
elif [ "$BUILD" = 'osx' ]; then
    # TODO(lukeyeager): enable after caffe2/caffe2#768, caffe2/caffe2#785
    export BUILD_TESTS=false
elif [ "$BUILD" = 'osx-ios' ]; then
    export BUILD_IOS=true
    export BUILD_TESTS=false
elif [ "$BUILD" = 'osx-android' ]; then
    export BUILD_ANDROID=true
    export BUILD_TESTS=false
else
    echo "BUILD \"$BUILD\" is unknown"
    exit 1
fi

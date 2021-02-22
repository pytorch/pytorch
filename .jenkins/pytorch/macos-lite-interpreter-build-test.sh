#!/bin/bash

echo "BUILD_LITE_INTERPRETER: ${BUILD_LITE_INTERPRETER}"
if [ "${BUILD_LITE_INTERPRETER}" == 1 ]; then
    echo "Testing libtorch (lite interpreter)."

    CPP_BUILD="$PWD/../cpp-build"
    rm -rf $CPP_BUILD
    mkdir -p $CPP_BUILD/caffe2

    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    pushd $CPP_BUILD/caffe2
    VERBOSE=1 DEBUG=1 python $BUILD_LIBTORCH_PY
    popd

    python tools/download_mnist.py --quiet -d test/cpp/api/mnist

    # Unfortunately it seems like the test can't load from miniconda3
    # without these paths being set
    export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$PWD/miniconda3/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/miniconda3/lib"
    TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" "$CPP_BUILD"/caffe2/bin/test_api

    assert_git_not_dirty
else
    echo "Skipping libtorch (lite interpreter)."
fi

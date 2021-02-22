#!/bin/bash

# shellcheck disable=SC2034
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

git submodule sync --recursive
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${WORKSPACE_DIR}/miniconda3/

# Build PyTorch
if [ -z "${IN_CI}" ]; then
  export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
fi

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
    # TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" "$CPP_BUILD"/caffe2/bin/test_api
    echo "Finding test_lite_interpreter_runtime"
    find . -regex '.*test_lite_interpreter_runtime'

    $CPP_BUILD/caffe2/bin/test_lite_interpreter_runtime

    assert_git_not_dirty
else
    echo "Skipping libtorch (lite interpreter)."
fi

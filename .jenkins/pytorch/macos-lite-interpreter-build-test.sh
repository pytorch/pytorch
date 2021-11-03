#!/bin/bash

# shellcheck source=./macos-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

# Build PyTorch
if [ -z "${IN_CI}" ]; then
  export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
fi

echo "BUILD_LITE_INTERPRETER: ${BUILD_LITE_INTERPRETER}"
if [ "${BUILD_LITE_INTERPRETER}" == 1 ]; then
    echo "Testing libtorch (lite interpreter)."

    CPP_BUILD="$(pwd)/../cpp_build"
    # Ensure the removal of the tmp directory
    trap 'rm -rfv ${CPP_BUILD}' EXIT
    rm -rf "${CPP_BUILD}"
    mkdir -p "${CPP_BUILD}/caffe2"

    # It looks libtorch need to be built in "${CPP_BUILD}/caffe2"
    # folder.
    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    pushd "${CPP_BUILD}/caffe2" || exit
    VERBOSE=1 DEBUG=1 python "${BUILD_LIBTORCH_PY}"
    popd || exit

    "${CPP_BUILD}/caffe2/build/bin/test_lite_interpreter_runtime"

    assert_git_not_dirty
else
    echo "Skipping libtorch (lite interpreter)."
fi

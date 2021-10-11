#!/bin/bash

# shellcheck source=./macos-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

git submodule sync --recursive
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${WORKSPACE_DIR}/miniconda3/

# Build PyTorch
if [ -z "${IN_CI}" ]; then
  export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
fi

echo "TRACING_BASED: ${TRACING_BASED}"
if [ "${TRACING_BASED}" == 1 ]; then
    echo "Testing model tracer."

    CPP_BUILD="$PWD/../cpp-build"
    rm -rf "${CPP_BUILD}"
    mkdir -p "${CPP_BUILD}/caffe2"

    # It looks libtorch need to be built in "${CPP_BUILD}/caffe2"
    # folder.
    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    pushd "${CPP_BUILD}/caffe2" || exit
    VERBOSE=1 DEBUG=1 python "${BUILD_LIBTORCH_PY}"
    popd || exit

    "ls ${CPP_BUILD}/caffe2/build/bin/model_tracer"

    # Change the permission manually from 755 to 644 to keep git clean
    chmod 644 "${HOME}/project/.jenkins/pytorch/macos-tracer-build-test.sh"
    assert_git_not_dirty
else
    echo "Skipping model tracer."
fi

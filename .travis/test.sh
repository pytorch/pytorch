#!/bin/bash
set -e
set -x

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$(dirname "$LOCAL_DIR")
cd "$ROOT_DIR"

if [ "$BUILD_TESTS" = 'false' ]; then
    echo 'Skipping tests'
    exit 0
fi

# Ctests
pushd build
CTEST_OUTPUT_ON_FAILURE=1 make test
popd

# Python tests
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/install"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ROOT_DIR}/install/lib"
python -m pytest -v install/caffe2/python

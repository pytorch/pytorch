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

cd build
CTEST_OUTPUT_ON_FAILURE=1 make test

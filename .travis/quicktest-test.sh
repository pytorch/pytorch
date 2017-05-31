#!/bin/bash
set -e
set -x

cd build
CTEST_OUTPUT_ON_FAILURE=1 make test -j$(nproc)

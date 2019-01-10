#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-multigpu-test"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Testing pytorch (distributed only)"
time python test/run_test.py --verbose -i distributed

#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# echo "Testing pytorch"
time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/test_sharded_tensor
assert_git_not_dirty

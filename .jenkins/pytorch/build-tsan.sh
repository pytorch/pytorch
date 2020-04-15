#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Clang version:"
clang --version


# TODO: Make the TSAN flags a centralized env var and unify with USE_ASAN option
CC="clang" CXX="clang++" LDSHARED="clang --shared" \
  CFLAGS="-fsanitize=thread" \
  USE_TSAN=1 USE_CUDA=0 USE_MKLDNN=0 \
  python setup.py install

assert_git_not_dirty

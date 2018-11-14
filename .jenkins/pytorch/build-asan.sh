#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-build"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Clang version:"
clang --version

# detect_leaks=0: Python is very leaky, so we need suppress it
# symbolize=1: Gives us much better errors when things go wrong
export ASAN_OPTIONS=detect_leaks=0:symbolize=1

# TODO: Make the ASAN flags a more unified env var
CC="clang" CXX="clang++" LDSHARED="clang --shared" \
  CFLAGS="-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -shared-libasan" \
  NO_CUDA=1 USE_MKLDNN=0 \
  python setup.py install

#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

echo "Clang version:"
clang --version

python tools/stats/export_test_times.py

if [ -n "$(which conda)" ]; then
  export CMAKE_PREFIX_PATH=/opt/conda
fi

CC="clang" CXX="clang++" LDSHARED="clang --shared" \
  CFLAGS="-fsanitize=thread" \
  USE_TSAN=1 USE_CUDA=0 USE_MKLDNN=0 \
  python setup.py bdist_wheel
  pip_install_whl "$(echo dist/*.whl)"

print_sccache_stats

assert_git_not_dirty

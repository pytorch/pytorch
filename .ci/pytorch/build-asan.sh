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

# detect_leaks=0: Python is very leaky, so we need suppress it
# symbolize=1: Gives us much better errors when things go wrong
export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1:symbolize=1:detect_odr_violation=0
if [ -n "$(which conda)" ]; then
  export CMAKE_PREFIX_PATH=/opt/conda
fi

# TODO: Make the ASAN flags a centralized env var and unify with USE_ASAN option
CC="clang" CXX="clang++" LDSHARED="clang --shared" \
  CFLAGS="-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-address-use-after-scope -shared-libasan" \
  USE_ASAN=1 USE_CUDA=0 USE_MKLDNN=0 \
  python setup.py bdist_wheel
  pip_install_whl "$(echo dist/*.whl)"

# Test building via the sdist source tarball
python setup.py sdist
mkdir -p /tmp/tmp
pushd /tmp/tmp
tar zxf "$(dirname "${BASH_SOURCE[0]}")/../../dist/"*.tar.gz
cd torch-*
python setup.py build --cmake-only
popd

print_sccache_stats

assert_git_not_dirty

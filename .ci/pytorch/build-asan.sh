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
  USE_ASAN=1 USE_CUDA=0 USE_MKLDNN=0 \
  UBSAN_FLAGS="-fno-sanitize-recover=all" \
  python setup.py bdist_wheel
  pip_install_whl "$(echo dist/*.whl)"

# Test building via the sdist source tarball
python setup.py sdist
mkdir -p /tmp/tmp
pushd /tmp/tmp
tar zxf "$(dirname "${BASH_SOURCE[0]}")/../../dist/"*.tar.gz
cd torch-*
# TODO: Remove USE_MKLDNN=OFF once https://github.com/pytorch/pytorch/issues/103212 is resolved
USE_MKLDNN=OFF python setup.py build --cmake-only
popd

print_sccache_stats

assert_git_not_dirty

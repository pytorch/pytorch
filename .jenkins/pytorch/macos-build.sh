#!/bin/bash

# shellcheck disable=SC2034
# shellcheck source=./macos-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

export CMAKE_PREFIX_PATH=${WORKSPACE_DIR}/miniconda3/

# Build PyTorch
if [ -z "${IN_CI}" ]; then
  export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
fi

# This helper function wraps calls to binaries with sccache, but only if they're not already wrapped with sccache.
# For example, `clang` will be `sccache clang`, but `sccache clang` will not become `sccache sccache clang`.
# The way this is done is by detecting the command of the parent pid of the current process and checking whether
# that is sccache, and wrapping sccache around the process if its parent were not already sccache.
function write_sccache_stub() {
  printf "#!/bin/sh\nif [ \$(ps auxc \$(ps auxc -o ppid \$\$ | grep \$\$ | rev | cut -d' ' -f1 | rev) | tr '\\\\n' ' ' | rev | cut -d' ' -f2 | rev) != sccache ]; then\n  exec sccache %s \"\$@\"\nelse\n  exec %s \"\$@\"\nfi" "$(which "$1")" "$(which "$1")" > "${WORKSPACE_DIR}/$1"
  chmod a+x "${WORKSPACE_DIR}/$1"
}

if which sccache > /dev/null; then
  write_sccache_stub clang++
  write_sccache_stub clang

  export PATH="${WORKSPACE_DIR}:$PATH"
fi

if [ -z "${CROSS_COMPILE_ARM64}" ]; then
  USE_DISTRIBUTED=1 python setup.py install
else
  export MACOSX_DEPLOYMENT_TARGET=11.0
  USE_DISTRIBUTED=1 CMAKE_OSX_ARCHITECTURES=arm64 USE_MKLDNN=OFF USE_NNPACK=OFF USE_QNNPACK=OFF BUILD_TEST=OFF python setup.py bdist_wheel
fi

assert_git_not_dirty

# Upload torch binaries when the build job is finished
if [ -z "${IN_CI}" ]; then
  7z a "${IMAGE_COMMIT_TAG}".7z "${WORKSPACE_DIR}"/miniconda3/lib/python3.6/site-packages/torch*
  aws s3 cp "${IMAGE_COMMIT_TAG}".7z s3://ossci-macos-build/pytorch/"${IMAGE_COMMIT_TAG}".7z --acl public-read
fi

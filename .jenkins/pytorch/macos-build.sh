#!/bin/bash

# shellcheck disable=SC2034
# shellcheck source=./macos-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"

# Build PyTorch
if [ -z "${IN_CI}" ]; then
  export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
fi

# This helper function wraps calls to binaries with sccache, but only if they're not already wrapped with sccache.
# For example, `clang` will be `sccache clang`, but `sccache clang` will not become `sccache sccache clang`.
# The way this is done is by detecting the command of the parent pid of the current process and checking whether
# that is sccache, and wrapping sccache around the process if its parent were not already sccache.
function write_sccache_stub() {
  output=$1
  binary=$(basename "${output}")

  printf "#!/bin/sh\nif [ \$(ps auxc \$(ps auxc -o ppid \$\$ | grep \$\$ | rev | cut -d' ' -f1 | rev) | tr '\\\\n' ' ' | rev | cut -d' ' -f2 | rev) != sccache ]; then\n  exec sccache %s \"\$@\"\nelse\n  exec %s \"\$@\"\nfi" "$(which "${binary}")" "$(which "${binary}")" > "${output}"
  chmod a+x "${output}"
}

if which sccache > /dev/null; then
  # Create temp directory for sccache shims
  tmp_dir=$(mktemp -d)
  trap 'rm -rfv ${tmp_dir}' EXIT
  write_sccache_stub "${tmp_dir}/clang++"
  write_sccache_stub "${tmp_dir}/clang"

  export PATH="${tmp_dir}:$PATH"
fi

if [[ ${BUILD_ENVIRONMENT} = *arm64* ]]; then
  # Cross compilation for arm64
  USE_DISTRIBUTED=1 CMAKE_OSX_ARCHITECTURES=arm64 MACOSX_DEPLOYMENT_TARGET=11.0 USE_MKLDNN=OFF USE_NNPACK=OFF USE_QNNPACK=OFF BUILD_TEST=OFF python setup.py bdist_wheel
else
  USE_DISTRIBUTED=1 python setup.py install
fi

assert_git_not_dirty

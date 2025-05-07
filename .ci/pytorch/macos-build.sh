#!/bin/bash

# shellcheck disable=SC2034
# shellcheck source=./macos-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/macos-common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

# Build PyTorch
if [ -z "${CI}" ]; then
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

print_cmake_info
if [[ ${BUILD_ENVIRONMENT} == *"distributed"* ]]; then
  # Needed for inductor benchmarks, as lots of HF networks make `torch.distribtued` calls
  USE_DISTRIBUTED=1 USE_OPENMP=1 WERROR=1 python setup.py bdist_wheel
else
  # Explicitly set USE_DISTRIBUTED=0 to align with the default build config on mac. This also serves as the sole CI config that tests
  # that building with USE_DISTRIBUTED=0 works at all. See https://github.com/pytorch/pytorch/issues/86448
  USE_DISTRIBUTED=0 USE_OPENMP=1 MACOSX_DEPLOYMENT_TARGET=11.0 WERROR=1 BUILD_TEST=OFF USE_PYTORCH_METAL=1 python setup.py bdist_wheel
fi
if which sccache > /dev/null; then
  print_sccache_stats
fi

python tools/stats/export_test_times.py

assert_git_not_dirty

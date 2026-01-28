#!/bin/bash
# Example: test.sh after replacing env config logic with lumen_cli
#
# This shows a hybrid approach where:
# - Environment variables are computed by Python (lumen_cli)
# - Build verification (ASAN, debug asserts, CUDA init) is done by Python (lumen_cli)
# - Runtime actions (trap, chown, patching, oneAPI sourcing) stay in bash
#
# What lumen replaces:
# - ~100 lines of environment variable if-else logic
# - ASAN/UBSAN verification tests
# - Debug assertion verification
# - CUDA initialization verification
# - ROCm architecture detection

set -ex -o pipefail

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

# ===========================================================================
# RUNTIME ACTIONS - Must stay in bash (trap/cleanup, system calls, sourcing)
# ===========================================================================

# Workspace permissions with trap cleanup (lines 17-33)
# Must stay in bash for proper trap EXIT handling
if [[ "$BUILD_ENVIRONMENT" != *rocm* && "$BUILD_ENVIRONMENT" != *s390x* && -d /var/lib/jenkins/workspace ]]; then
  WORKSPACE_ORIGINAL_OWNER_ID=$(stat -c '%u' "/var/lib/jenkins/workspace")
  cleanup_workspace() {
    echo "sudo may print the following warning message that can be ignored."
    sudo chown -R "$WORKSPACE_ORIGINAL_OWNER_ID" /var/lib/jenkins/workspace
  }
  # shellcheck disable=SC2064
  trap_add cleanup_workspace EXIT
  sudo chown -R jenkins /var/lib/jenkins/workspace
  git config --global --add safe.directory /var/lib/jenkins/workspace
fi

# Numba CUDA-13 patch (lines 36-45)
if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
  NUMBA_CUDA_DIR=$(python -c "import os;import numba.cuda; print(os.path.dirname(numba.cuda.__file__))" 2>/dev/null || true)
  if [ -n "$NUMBA_CUDA_DIR" ]; then
    NUMBA_PATCH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/numba-cuda-13.patch"
    pushd "$NUMBA_CUDA_DIR"
    patch -p4 <"$NUMBA_PATCH"
    popd
  fi
fi

# Core dump settings (lines 114-129)
if [[ "${PYTORCH_TEST_RERUN_DISABLED_TESTS}" == "1" ]] || [[ "${CONTINUE_THROUGH_ERROR}" == "1" ]]; then
  ulimit -c 0
fi

# XPU oneAPI sourcing (lines 200-217)
# Must stay in bash - sources external scripts that set many env vars
if [[ "$BUILD_ENVIRONMENT" == *xpu* ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/compiler/latest/env/vars.sh
  if [ -f /opt/intel/oneapi/umf/latest/env/vars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/umf/latest/env/vars.sh
  fi
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/ccl/latest/env/vars.sh
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/mpi/latest/env/vars.sh
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/pti/latest/env/vars.sh
  timeout 30 xpu-smi discovery || true
fi

# ===========================================================================
# ENVIRONMENT CONFIGURATION - Use lumen_cli
# Replaces ~100 lines of bash if-else logic for env vars
# ===========================================================================
eval "$(lumen test pytorch env \
    --build-environment "$BUILD_ENVIRONMENT" \
    --test-config "${TEST_CONFIG}" \
    --shard-id "${SHARD_NUMBER:1}" \
    --num-shards "${NUM_TEST_SHARDS:1}" \
    --export)"

echo "Environment variables:"
env

pip install -e .ci/lumen_cli

lumen test pytorch env \
    --build-environment "$BUILD_ENVIRONMENT" \
    --test-config "${TEST_CONFIG}" \
    --shard-id "${SHARD_NUMBER:1}" \
    --num-shards "${NUM_TEST_SHARDS:1}" \
    --verify-build-env

# ===========================================================================
# REST OF test.sh CONTINUES FROM HERE (line 313+)
# Test execution logic...
# ===========================================================================

echo "Testing pytorch"

# Example: Run tests with include clause from lumen
# INCLUDE_CLAUSE is set by `lumen test pytorch env --export` if TESTS_TO_INCLUDE is set
if [[ -n "$INCLUDE_CLAUSE" ]]; then
  python test/run_test.py $INCLUDE_CLAUSE
else
  python test/run_test.py
fi

# ... rest of test.sh (test execution logic) ...

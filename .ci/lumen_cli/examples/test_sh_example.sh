#!/bin/bash
# Example: test.sh after replacing env config logic with lumen_cli
#
# This shows a hybrid approach where:
# - Environment variables are computed by Python (lumen_cli)
# - Runtime actions (trap, chown, patching) stay in bash
#
# Lines replaced by `lumen test pytorch env --export`:
# - Lines 10, 64, 147: Base env (TERM, TORCH_SERIALIZATION_DEBUG, LANG)
# - Lines 60-61: SHARD_NUMBER, NUM_TEST_SHARDS defaults
# - Lines 66-104, 108-112, 189-191, 226-229: VALGRIND setup
# - Lines 151-158: CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES
# - Lines 160-163: PYTORCH_TEST_WITH_SLOW, PYTORCH_TEST_SKIP_FAST
# - Lines 165-170: PYTORCH_TEST_WITH_SLOW_GRADCHECK, PYTORCH_TEST_CUDA_MEM_LEAK_CHECK
# - Lines 172-183: PYTORCH_TESTING_DEVICE_ONLY_FOR, PYTHON_TEST_EXTRA_OPTION, NO_TEST_TIMEOUT
# - Lines 185-187: PYTORCH_TEST_WITH_CROSSREF
# - Lines 219-224: PATH setup for ninja
# - Lines 234-279: ASAN/UBSAN settings (except LD_PRELOAD)
# - Lines 302-306: ATEN_CPU_CAPABILITY
# - Lines 308-311: USE_LEGACY_DRIVER

set -ex -o pipefail

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

# ===========================================================================
# RUNTIME ACTIONS - Keep in bash (trap/cleanup, commands, patching)
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

# Numba CUDA patch (lines 36-45)
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

# Detect CUDA architecture (line 106)
detect_cuda_arch

# ROCm info (lines 193-197)
if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  rocminfo
  rocminfo | grep -E 'Name:.*\sgfx|Marketing'
  MAYBE_ROCM="rocm/"
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
# ===========================================================================

# Apply all computed environment variables from Python
# This replaces ~100 lines of bash if-else logic
eval $(lumen test pytorch env --export)

echo "Environment variables:"
env

# ===========================================================================
# ASAN-SPECIFIC RUNTIME ACTIONS (lines 276-287)
# LD_PRELOAD requires command execution, verification tests run Python
# ===========================================================================

if [[ "$BUILD_ENVIRONMENT" == *asan* ]]; then
    # LD_PRELOAD must be computed at runtime (line 276-277)
    LD_PRELOAD=$(clang --print-file-name=libclang_rt.asan-x86_64.so)
    export LD_PRELOAD

    # Verification tests (lines 281-286)
    (cd test && python -c "import torch; print(torch.__version__, torch.version.git_version)")
    echo "The next four invocations are expected to crash; if they don't that means ASAN/UBSAN is misconfigured"
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_asan(3)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_vptr_ubsan()")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_aten_asan(3)")
fi

# Debug assert verification (lines 289-300)
if [[ "$BUILD_ENVIRONMENT" == *-debug* ]]; then
    echo "We are in debug mode: $BUILD_ENVIRONMENT. Expect the python assertion to fail"
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_debug_asserts_fail(424242)")
elif [[ "$BUILD_ENVIRONMENT" != *-bazel-* ]]; then
    echo "We are not in debug mode: $BUILD_ENVIRONMENT. Expect the assertion to pass"
    (cd test && python -c "import torch; torch._C._crash_if_debug_asserts_fail(424242)")
fi

# Legacy NVIDIA driver CUDA init test (line 310)
if [[ "${TEST_CONFIG}" == "legacy_nvidia_driver" ]]; then
    (cd test && python -c "import torch; torch.rand(2, 2, device='cuda')")
fi

# ===========================================================================
# REST OF test.sh CONTINUES FROM HERE (line 313+)
# Test execution logic...
# ===========================================================================

echo "Testing pytorch"

# ... rest of test.sh (test execution logic) ...

#!/bin/bash

# Linux test environment setup.
# Sourced by test.sh before running any test functions.
#
# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

# Only change workspace permissions if passwordless sudo is available
# (e.g. ROCm and s390x CI jobs lack it, and changing permissions
# can leave the workspace in a bad state for cancelled jobs)
if sudo -n true 2>/dev/null && [[ -d /var/lib/jenkins/workspace ]]; then
  # Workaround for dind-rootless userid mapping (https://github.com/pytorch/ci-infra/issues/96)
  WORKSPACE_ORIGINAL_OWNER_ID=$(stat -c '%u' "/var/lib/jenkins/workspace")
  cleanup_workspace() {
    echo "sudo may print the following warning message that can be ignored. The chown command will still run."
    echo "    sudo: setrlimit(RLIMIT_STACK): Operation not permitted"
    echo "For more details refer to https://github.com/sudo-project/sudo/issues/42"
    sudo chown -R "$WORKSPACE_ORIGINAL_OWNER_ID" /var/lib/jenkins/workspace
  }
  # Disable shellcheck SC2064 as we want to parse the original owner immediately.
  # shellcheck disable=SC2064
  trap_add cleanup_workspace EXIT
  sudo chown -R jenkins /var/lib/jenkins/workspace
  git config --global --add safe.directory /var/lib/jenkins/workspace
fi


# Patch numba to avoid CUDA-13 crash, see https://github.com/pytorch/pytorch/issues/162878
if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
  NUMBA_CUDA_DIR=$(python -c "import os;import numba.cuda; print(os.path.dirname(numba.cuda.__file__))" 2>/dev/null || true)
  if [ -n "$NUMBA_CUDA_DIR" ]; then
    NUMBA_PATCH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/numba-cuda-13.patch"
    pushd "$NUMBA_CUDA_DIR"
    patch -p4 <"$NUMBA_PATCH"
    popd
  fi
fi

echo "Environment variables:"
env

#Set Default values for these variables in case they are not set
SHARD_NUMBER="${SHARD_NUMBER:=1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:=1}"

# enable debug asserts in serialization
export TORCH_SERIALIZATION_DEBUG=1

export VALGRIND=ON
# export TORCH_INDUCTOR_INSTALL_GXX=ON
if [[ "$BUILD_ENVIRONMENT" == *clang9* || "$BUILD_ENVIRONMENT" == *xpu* ]]; then
  # clang9 appears to miscompile code involving std::optional<c10::SymInt>,
  # such that valgrind complains along these lines:
  #
  # Conditional jump or move depends on uninitialised value(s)
  #    at 0x40303A: ~optional_base (Optional.h:281)
  #    by 0x40303A: call (Dispatcher.h:448)
  #    by 0x40303A: call(at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::SymInt>) (basic.cpp:10)
  #    by 0x403700: main (basic.cpp:16)
  #  Uninitialised value was created by a stack allocation
  #    at 0x402AAA: call(at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::SymInt>) (basic.cpp:6)
  #
  # The problem does not appear with gcc or newer versions of clang (we tested
  # clang14).  So we suppress valgrind testing for clang9 specifically.
  # You may need to suppress it for other versions of clang if they still have
  # the bug.
  #
  # A minimal repro for the valgrind error is below:
  #
  # #include <ATen/ATen.h>
  # #include <ATen/core/dispatch/Dispatcher.h>
  #
  # using namespace at;
  #
  # Tensor call(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, std::optional<c10::SymInt> storage_offset) {
  #   auto op = c10::Dispatcher::singleton()
  #       .findSchemaOrThrow(at::_ops::as_strided::name, at::_ops::as_strided::overload_name)
  #       .typed<at::_ops::as_strided::schema>();
  #   return op.call(self, size, stride, storage_offset);
  # }
  #
  # int main(int argv) {
  #   Tensor b = empty({3, 4});
  #   auto z = call(b, b.sym_sizes(), b.sym_strides(), std::nullopt);
  # }
  export VALGRIND=OFF
fi

detect_cuda_arch

if [[ "$BUILD_ENVIRONMENT" == *s390x* ]]; then
  # There are additional warnings on s390x, maybe due to newer gcc.
  # Skip this check for now
  export VALGRIND=OFF
fi

if [[ "${PYTORCH_TEST_RERUN_DISABLED_TESTS}" == "1" ]] || [[ "${CONTINUE_THROUGH_ERROR}" == "1" ]]; then
  # When rerunning disable tests, do not generate core dumps as it could consume
  # the runner disk space when crashed tests are run multiple times. Running out
  # of space is a nasty issue because there is no space left to even download the
  # GHA to clean up the disk
  #
  # We also want to turn off core dump when CONTINUE_THROUGH_ERROR is set as there
  # is a small risk of having multiple core files generated. Arguably, they are not
  # that useful in this case anyway and the test will still continue
  ulimit -c 0

  # Note that by piping the core dump to a script set in /proc/sys/kernel/core_pattern
  # as documented in https://man7.org/linux/man-pages/man5/core.5.html, we could
  # dynamically stop generating more core file when the disk space drops below a
  # certain threshold. However, this is not supported inside Docker container atm
fi

# Get fully qualified path using realpath
if [[ "$BUILD_ENVIRONMENT" != *bazel* ]]; then
  CUSTOM_TEST_ARTIFACT_BUILD_DIR=$(realpath "${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-"build/custom_test_artifacts"}")
fi

echo "Environment variables"
env

echo "Testing pytorch"

export LANG=C.UTF-8

PR_NUMBER=${PR_NUMBER:-${CIRCLE_PR_NUMBER:-}}

if [[ -d "${HF_CACHE}" ]]; then
  export HF_HOME="${HF_CACHE}"
fi

if [[ "$TEST_CONFIG" == 'default' ]]; then
  export CUDA_VISIBLE_DEVICES=0
  export HIP_VISIBLE_DEVICES=0
fi

if [[ "$TEST_CONFIG" == 'distributed' ]] && [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  export HIP_VISIBLE_DEVICES=0,1,2,3
fi

if [[ "$TEST_CONFIG" == 'slow' ]]; then
  export PYTORCH_TEST_WITH_SLOW=1
  export PYTORCH_TEST_SKIP_FAST=1
fi

if [[ "$BUILD_ENVIRONMENT" == *slow-gradcheck* ]]; then
  export PYTORCH_TEST_WITH_SLOW_GRADCHECK=1
  # TODO: slow gradcheck tests run out of memory a lot recently, so setting this
  # to run them sequentially with only one process to mitigate the issue
  export PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1
fi

if [[ "$BUILD_ENVIRONMENT" == *cuda* || "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # Used so that only cuda/rocm specific versions of tests are generated
  # mainly used so that we're not spending extra cycles testing cpu
  # devices on expensive gpu machines
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
elif [[ "$BUILD_ENVIRONMENT" == *xpu* ]]; then
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="xpu"
  # setting PYTHON_TEST_EXTRA_OPTION
  export PYTHON_TEST_EXTRA_OPTION="--xpu"
  # disable timeout due to shard not balance for xpu
  export NO_TEST_TIMEOUT=True
elif [[ "$BUILD_ENVIRONMENT" == *pallas-tpu* ]]; then
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="tpu"
fi

if [[ "$TEST_CONFIG" == *crossref* ]]; then
  export PYTORCH_TEST_WITH_CROSSREF=1
fi

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # regression in ROCm 6.0 on MI50 CI runners due to hipblaslt; remove in 6.1
  export VALGRIND=OFF
  # Print GPU info
  rocminfo
  rocminfo | grep -E 'Name:.*\sgfx|Marketing'

fi

if [[ "$BUILD_ENVIRONMENT" == *xpu* ]]; then
  # Source Intel oneAPI envrioment script to enable xpu runtime related libraries
  # refer to https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html
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
  # Check XPU status before testing
  timeout 30 xpu-smi discovery || true
fi

if [[ "$BUILD_ENVIRONMENT" != *-bazel-* ]] ; then
  # JIT C++ extensions require ninja (installed from requirements-ci.txt).
  # ninja is installed in $HOME/.local/bin, e.g., /var/lib/jenkins/.local/bin for CI user jenkins
  # but this script should be runnable by any user, including root
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ "$BUILD_ENVIRONMENT" == *aarch64* ]]; then
  # TODO: revisit this once the CI is stabilized on aarch64 linux
  export VALGRIND=OFF
fi

# DANGER WILL ROBINSON.  The LD_PRELOAD here could cause you problems
# if you're not careful.  Check this if you made some changes and the
# ASAN test is not working
if [[ "$BUILD_ENVIRONMENT" == *asan* ]]; then
    export ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_stack_use_after_return=true:strict_init_order=true:detect_odr_violation=1:detect_container_overflow=0:check_initialization_order=true:debug=true
    if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
        export ASAN_OPTIONS="${ASAN_OPTIONS}:protect_shadow_gap=0"
    fi
    export UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PWD/ubsan.supp
    export PYTORCH_TEST_WITH_ASAN=1
    export PYTORCH_TEST_WITH_UBSAN=1
    # TODO: Figure out how to avoid hard-coding these paths
    export ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-18/bin/llvm-symbolizer
    export TORCH_USE_RTLD_GLOBAL=1
    # NB: We load libtorch.so with RTLD_GLOBAL for UBSAN, unlike our
    # default behavior.
    #
    # The reason for this is that without RTLD_GLOBAL, if we load multiple
    # libraries that depend on libtorch (as is the case with C++ extensions), we
    # will get multiple copies of libtorch in our address space.  When UBSAN is
    # turned on, it will do a bunch of virtual pointer consistency checks which
    # won't work correctly.  When this happens, you get a violation like:
    #
    #    member call on address XXXXXX which does not point to an object of
    #    type 'std::_Sp_counted_base<__gnu_cxx::_Lock_policy::_S_atomic>'
    #    XXXXXX note: object is of type
    #    'std::_Sp_counted_ptr<torch::nn::LinearImpl*, (__gnu_cxx::_Lock_policy)2>'
    #
    # (NB: the textual types of the objects here are misleading, because
    # they actually line up; it just so happens that there's two copies
    # of the type info floating around in the address space, so they
    # don't pointer compare equal.  See also
    #   https://github.com/google/sanitizers/issues/1175
    #
    # UBSAN is kind of right here: if we relied on RTTI across C++ extension
    # modules they would indeed do the wrong thing;  but in our codebase, we
    # don't use RTTI (because it doesn't work in mobile).  To appease
    # UBSAN, however, it's better if we ensure all the copies agree!
    #
    # By the way, an earlier version of this code attempted to load
    # libtorch_python.so with LD_PRELOAD, which has a similar effect of causing
    # it to be loaded globally.  This isn't really a good idea though, because
    # it depends on a ton of dynamic libraries that most programs aren't gonna
    # have, and it applies to child processes.

    LD_PRELOAD=$(clang --print-file-name=libclang_rt.asan-x86_64.so)
    export LD_PRELOAD
    # Disable valgrind for asan
    export VALGRIND=OFF

    (cd test && python -c "import torch; print(torch.__version__, torch.version.git_version)")
    echo "The next four invocations are expected to crash; if they don't that means ASAN/UBSAN is misconfigured"
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_asan(3)")
    #(cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_ubsan(0)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_vptr_ubsan()")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_aten_asan(3)")
fi

# The torch._C._crash_if_debug_asserts_fail() function should only fail if both of the following are true:
# 1. The build is in debug mode
# 2. The value 424242 is passed in
# This tests that the debug asserts are working correctly.
if [[ "$BUILD_ENVIRONMENT" == *-debug* ]]; then
    echo "We are in debug mode: $BUILD_ENVIRONMENT. Expect the python assertion to fail"
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_debug_asserts_fail(424242)")
elif [[ "$BUILD_ENVIRONMENT" != *-bazel-* ]]; then
    # Noop when debug is disabled. Skip bazel jobs because torch isn't available there yet.
    echo "We are not in debug mode: $BUILD_ENVIRONMENT. Expect the assertion to pass"
    (cd test && python -c "import torch; torch._C._crash_if_debug_asserts_fail(424242)")
fi

if [[ $TEST_CONFIG == 'nogpu_NO_AVX2' ]]; then
  export ATEN_CPU_CAPABILITY=default
elif [[ $TEST_CONFIG == 'nogpu_AVX512' ]]; then
  export ATEN_CPU_CAPABILITY=avx2
fi

#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

set -ex -o pipefail

# Suppress ANSI color escape sequences
export TERM=vt100

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Do not change workspace permissions for ROCm CI jobs
# as it can leave workspace with bad permissions for cancelled jobs
if [[ "$BUILD_ENVIRONMENT" != *rocm* && -d /var/lib/jenkins/workspace ]]; then
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

echo "Environment variables:"
env

TORCH_INSTALL_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/torch
TORCH_BIN_DIR="$TORCH_INSTALL_DIR"/bin
TORCH_LIB_DIR="$TORCH_INSTALL_DIR"/lib
TORCH_TEST_DIR="$TORCH_INSTALL_DIR"/test

BUILD_DIR="build"
BUILD_RENAMED_DIR="build_renamed"
BUILD_BIN_DIR="$BUILD_DIR"/bin

#Set Default values for these variables in case they are not set
SHARD_NUMBER="${SHARD_NUMBER:=1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:=1}"

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

# Reduce set of tests to include when running run_test.py
if [[ -n $TESTS_TO_INCLUDE ]]; then
  echo "Setting INCLUDE_CLAUSE"
  INCLUDE_CLAUSE="--include $TESTS_TO_INCLUDE"
fi

echo "Environment variables"
env

echo "Testing pytorch"

export LANG=C.UTF-8

PR_NUMBER=${PR_NUMBER:-${CIRCLE_PR_NUMBER:-}}

if [[ "$TEST_CONFIG" == 'default' ]]; then
  export CUDA_VISIBLE_DEVICES=0
  export HIP_VISIBLE_DEVICES=0
fi

if [[ "$TEST_CONFIG" == 'distributed' ]] && [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  export HIP_VISIBLE_DEVICES=0,1
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
  # Check XPU status before testing
  xpu-smi discovery
fi

if [[ "$BUILD_ENVIRONMENT" != *-bazel-* ]] ; then
  # JIT C++ extensions require ninja.
  pip_install --user "ninja==1.10.2"
  # ninja is installed in $HOME/.local/bin, e.g., /var/lib/jenkins/.local/bin for CI user jenkins
  # but this script should be runnable by any user, including root
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ "$BUILD_ENVIRONMENT" == *aarch64* ]]; then
  # TODO: revisit this once the CI is stabilized on aarch64 linux
  export VALGRIND=OFF
fi

install_tlparse

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
    export ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-15/bin/llvm-symbolizer
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

test_python_legacy_jit() {
  time python test/run_test.py --include test_jit_legacy test_jit_fuser_legacy --verbose
  assert_git_not_dirty
}

test_python_shard() {
  if [[ -z "$NUM_TEST_SHARDS" ]]; then
    echo "NUM_TEST_SHARDS must be defined to run a Python test shard"
    exit 1
  fi

  # Bare --include flag is not supported and quoting for lint ends up with flag not being interpreted correctly
  # shellcheck disable=SC2086

  # modify LD_LIBRARY_PATH to ensure it has the conda env.
  # This set of tests has been shown to be buggy without it for the split-build
  time python test/run_test.py --exclude-jit-executor --exclude-distributed-tests $INCLUDE_CLAUSE --shard "$1" "$NUM_TEST_SHARDS" --verbose $PYTHON_TEST_EXTRA_OPTION --upload-artifacts-while-running

  assert_git_not_dirty
}

test_python() {
  # shellcheck disable=SC2086
  time python test/run_test.py --exclude-jit-executor --exclude-distributed-tests $INCLUDE_CLAUSE --verbose $PYTHON_TEST_EXTRA_OPTION
  assert_git_not_dirty
}


test_dynamo_wrapped_shard() {
  if [[ -z "$NUM_TEST_SHARDS" ]]; then
    echo "NUM_TEST_SHARDS must be defined to run a Python test shard"
    exit 1
  fi
  python tools/dynamo/verify_dynamo.py
  # PLEASE DO NOT ADD ADDITIONAL EXCLUDES HERE.
  # Instead, use @skipIfTorchDynamo on your tests.
  time python test/run_test.py --dynamo \
    --exclude-inductor-tests \
    --exclude-jit-executor \
    --exclude-distributed-tests \
    --exclude-torch-export-tests \
    --exclude-aot-dispatch-tests \
    --shard "$1" "$NUM_TEST_SHARDS" \
    --verbose \
    --upload-artifacts-while-running
  assert_git_not_dirty
}

test_inductor_distributed() {
  # Smuggle a few multi-gpu tests here so that we don't have to request another large node
  echo "Testing multi_gpu tests in test_torchinductor"
  python test/run_test.py -i inductor/test_torchinductor.py -k test_multi_gpu --verbose
  python test/run_test.py -i inductor/test_aot_inductor.py -k test_non_default_cuda_device --verbose
  python test/run_test.py -i inductor/test_aot_inductor.py -k test_replicate_on_devices --verbose
  python test/run_test.py -i distributed/test_c10d_functional_native.py --verbose
  python test/run_test.py -i distributed/_tensor/test_dtensor_compile.py --verbose
  python test/run_test.py -i distributed/tensor/parallel/test_micro_pipeline_tp.py --verbose
  python test/run_test.py -i distributed/_composable/test_replicate_with_compiler.py --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_comm.py --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_training.py -k test_train_parity_multi_group --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_training.py -k test_train_parity_with_activation_checkpointing --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_training.py -k test_train_parity_hsdp --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_training.py -k test_train_parity_2d_transformer_checkpoint_resume --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_training.py -k test_gradient_accumulation --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_state_dict.py -k test_dp_state_dict_save_load --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_frozen.py --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_mixed_precision.py -k test_compute_dtype --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_mixed_precision.py -k test_reduce_dtype --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_clip_grad_norm_.py -k test_clip_grad_norm_2d --verbose
  python test/run_test.py -i distributed/_composable/fsdp/test_fully_shard_compile.py --verbose
  python test/run_test.py -i distributed/fsdp/test_fsdp_tp_integration.py -k test_fsdp_tp_integration --verbose

  # this runs on both single-gpu and multi-gpu instance. It should be smart about skipping tests that aren't supported
  # with if required # gpus aren't available
  python test/run_test.py --include distributed/test_dynamo_distributed distributed/test_inductor_collectives distributed/test_compute_comm_reordering --verbose
  assert_git_not_dirty
}

test_inductor_shard() {
  if [[ -z "$NUM_TEST_SHARDS" ]]; then
    echo "NUM_TEST_SHARDS must be defined to run a Python test shard"
    exit 1
  fi

  python tools/dynamo/verify_dynamo.py
  python test/run_test.py --inductor \
    --include test_modules test_ops test_ops_gradients test_torch \
    --shard "$1" "$NUM_TEST_SHARDS" \
    --verbose

  # Do not add --inductor for the following inductor unit tests, otherwise we will fail because of nested dynamo state
  python test/run_test.py \
    --include inductor/test_torchinductor inductor/test_torchinductor_opinfo inductor/test_aot_inductor \
    --shard "$1" "$NUM_TEST_SHARDS" \
    --verbose
}

test_inductor_aoti() {
  # docker build uses bdist_wheel which does not work with test_aot_inductor
  # TODO: need a faster way to build
  if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
    # We need to hipify before building again
    python3 tools/amd_build/build_amd.py
  fi
  BUILD_AOT_INDUCTOR_TEST=1 python setup.py develop
  CPP_TESTS_DIR="${BUILD_BIN_DIR}" LD_LIBRARY_PATH="${TORCH_LIB_DIR}" python test/run_test.py --cpp --verbose -i cpp/test_aoti_abi_check cpp/test_aoti_inference
}

test_inductor_cpp_wrapper_shard() {
  if [[ -z "$NUM_TEST_SHARDS" ]]; then
    echo "NUM_TEST_SHARDS must be defined to run a Python test shard"
    exit 1
  fi

  export TORCHINDUCTOR_CPP_WRAPPER=1
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  if [[ "$1" -eq "2" ]]; then
    # For now, manually put the opinfo tests in shard 2, and all other tests in
    # shard 1.  Test specific things triggering past bugs, for now.
    python test/run_test.py \
      --include inductor/test_torchinductor_opinfo \
      -k 'linalg or to_sparse' \
      --verbose
  fi

  # Run certain inductor unit tests with cpp wrapper. In the end state, we
  # should be able to run all the inductor unit tests with cpp_wrapper.
  python test/run_test.py --include inductor/test_torchinductor --verbose

  # Run inductor benchmark tests with cpp wrapper.
  # Skip benchmark tests if it's in rerun-disabled-mode.
  if [[ "${PYTORCH_TEST_RERUN_DISABLED_TESTS}" == "1" ]]; then
    echo "skip dynamo benchmark tests for rerun-disabled-test"
  else
    echo "run dynamo benchmark tests with cpp wrapper"
    python benchmarks/dynamo/timm_models.py --device cuda --accuracy --amp \
    --training --inductor --disable-cudagraphs --only vit_base_patch16_224 \
    --output "$TEST_REPORTS_DIR/inductor_cpp_wrapper_training.csv"
    python benchmarks/dynamo/check_accuracy.py \
      --actual "$TEST_REPORTS_DIR/inductor_cpp_wrapper_training.csv" \
      --expected "benchmarks/dynamo/ci_expected_accuracy/inductor_timm_training.csv"

    python benchmarks/dynamo/torchbench.py --device cuda --accuracy \
      --bfloat16 --inference --inductor --only hf_T5 --output "$TEST_REPORTS_DIR/inductor_cpp_wrapper_inference.csv"
    python benchmarks/dynamo/torchbench.py --device cuda --accuracy \
      --bfloat16 --inference --inductor --only llama --output "$TEST_REPORTS_DIR/inductor_cpp_wrapper_inference.csv"
    python benchmarks/dynamo/torchbench.py --device cuda --accuracy \
      --bfloat16 --inference --inductor --only moco --output "$TEST_REPORTS_DIR/inductor_cpp_wrapper_inference.csv"
    python benchmarks/dynamo/check_accuracy.py \
      --actual "$TEST_REPORTS_DIR/inductor_cpp_wrapper_inference.csv" \
      --expected "benchmarks/dynamo/ci_expected_accuracy/inductor_torchbench_inference.csv"
  fi
}

# "Global" flags for inductor benchmarking controlled by TEST_CONFIG
# For example 'dynamic_aot_eager_torchbench' TEST_CONFIG means we run
# the benchmark script with '--dynamic-shapes --backend aot_eager --device cuda'
# The matrix of test options is specified in .github/workflows/inductor.yml,
# .github/workflows/inductor-periodic.yml, and
# .github/workflows/inductor-perf-test-nightly.yml
DYNAMO_BENCHMARK_FLAGS=()

pr_time_benchmarks() {

  pip_install --user "fbscribelogger"

  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"
  PYTHONPATH=$(pwd)/benchmarks/dynamo/pr_time_benchmarks source benchmarks/dynamo/pr_time_benchmarks/benchmark_runner.sh "$TEST_REPORTS_DIR/pr_time_benchmarks_results.csv" "benchmarks/dynamo/pr_time_benchmarks/benchmarks"
  echo "benchmark results on current PR: "
  cat  "$TEST_REPORTS_DIR/pr_time_benchmarks_results.csv"
  PYTHONPATH=$(pwd)/benchmarks/dynamo/pr_time_benchmarks python benchmarks/dynamo/pr_time_benchmarks/check_results.py "benchmarks/dynamo/pr_time_benchmarks/expected_results.csv" "$TEST_REPORTS_DIR/pr_time_benchmarks_results.csv" "$TEST_REPORTS_DIR/new_expected_results.csv"
}

if [[ "${TEST_CONFIG}" == *pr_time_benchmarks* ]]; then
  pr_time_benchmarks
  exit 0
elif [[ "${TEST_CONFIG}" == *dynamo_eager* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--backend eager)
elif [[ "${TEST_CONFIG}" == *aot_eager* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--backend aot_eager)
elif [[ "${TEST_CONFIG}" == *aot_inductor* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--export-aot-inductor)
elif [[ "${TEST_CONFIG}" == *inductor* && "${TEST_CONFIG}" != *perf* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--inductor)
fi

if [[ "${TEST_CONFIG}" == *dynamic* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--dynamic-shapes --dynamic-batch-only)
fi

if [[ "${TEST_CONFIG}" == *cpu* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--device cpu)
else
  DYNAMO_BENCHMARK_FLAGS+=(--device cuda)
fi

test_perf_for_dashboard() {
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  local suite="$1"
  shift

  local backend=inductor
  local modes=()
  if [[ "$DASHBOARD_TAG" == *training-true* ]]; then
    modes+=(training)
  fi
  if [[ "$DASHBOARD_TAG" == *inference-true* ]]; then
    modes+=(inference)
  fi
  # TODO: All the accuracy tests can be skipped once the CI accuracy checking is stable enough
  local targets=(accuracy performance)

  local device=cuda
  if [[ "${TEST_CONFIG}" == *cpu* ]]; then
    if [[ "${TEST_CONFIG}" == *cpu_x86* ]]; then
      device=cpu_x86
    elif [[ "${TEST_CONFIG}" == *cpu_aarch64* ]]; then
      device=cpu_aarch64
    fi
    test_inductor_set_cpu_affinity
  elif [[ "${TEST_CONFIG}" == *cuda_a10g* ]]; then
    device=cuda_a10g
  fi

  for mode in "${modes[@]}"; do
    if [[ "$mode" == "inference" ]]; then
      dtype=bfloat16
    elif [[ "$mode" == "training" ]]; then
      dtype=amp
    fi
    for target in "${targets[@]}"; do
      local target_flag=("--${target}")
      if [[ "$target" == "performance" ]]; then
        target_flag+=( --cold-start-latency)
      elif [[ "$target" == "accuracy" ]]; then
        target_flag+=( --no-translation-validation)
      fi

      if [[ "$DASHBOARD_TAG" == *default-true* ]]; then
        $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --backend "$backend" --disable-cudagraphs "$@" \
            --output "$TEST_REPORTS_DIR/${backend}_no_cudagraphs_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *cudagraphs-true* ]]; then
        $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --backend "$backend" "$@" \
            --output "$TEST_REPORTS_DIR/${backend}_with_cudagraphs_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *dynamic-true* ]]; then
        $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --backend "$backend" --dynamic-shapes \
            --dynamic-batch-only "$@" \
            --output "$TEST_REPORTS_DIR/${backend}_dynamic_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *cppwrapper-true* ]] && [[ "$mode" == "inference" ]]; then
        TORCHINDUCTOR_CPP_WRAPPER=1 $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --backend "$backend" --disable-cudagraphs "$@" \
            --output "$TEST_REPORTS_DIR/${backend}_cpp_wrapper_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *freezing_cudagraphs-true* ]] && [[ "$mode" == "inference" ]]; then
        $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --backend "$backend" "$@" --freezing \
            --output "$TEST_REPORTS_DIR/${backend}_with_cudagraphs_freezing_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *freeze_autotune_cudagraphs-true* ]] && [[ "$mode" == "inference" ]]; then
        TORCHINDUCTOR_MAX_AUTOTUNE=1 $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --backend "$backend" "$@" --freezing \
            --output "$TEST_REPORTS_DIR/${backend}_with_cudagraphs_freezing_autotune_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *aotinductor-true* ]] && [[ "$mode" == "inference" ]]; then
        if [[ "$target" == "accuracy" ]]; then
          # Also collect Export pass rate and display as a separate row
          $TASKSET python "benchmarks/dynamo/$suite.py" \
              "${target_flag[@]}" --"$mode" --"$dtype" --export --disable-cudagraphs "$@" \
              --output "$TEST_REPORTS_DIR/${backend}_export_${suite}_${dtype}_${mode}_${device}_${target}.csv"
        fi
        $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --export-aot-inductor --disable-cudagraphs "$@" \
            --output "$TEST_REPORTS_DIR/${backend}_aot_inductor_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *maxautotune-true* ]]; then
        TORCHINDUCTOR_MAX_AUTOTUNE=1 $TASKSET python "benchmarks/dynamo/$suite.py" \
            "${target_flag[@]}" --"$mode" --"$dtype" --backend "$backend" "$@" \
            --output "$TEST_REPORTS_DIR/${backend}_max_autotune_${suite}_${dtype}_${mode}_${device}_${target}.csv"
      fi
      if [[ "$DASHBOARD_TAG" == *cudagraphs_low_precision-true* ]] && [[ "$mode" == "inference" ]]; then
        # TODO: This has a new dtype called quant and the benchmarks script needs to be updated to support this.
        # The tentative command is as follows. It doesn't work now, but it's ok because we only need mock data
        # to fill the dashboard.
        $TASKSET python "benchmarks/dynamo/$suite.py" \
          "${target_flag[@]}" --"$mode" --quant --backend "$backend" "$@" \
          --output "$TEST_REPORTS_DIR/${backend}_cudagraphs_low_precision_${suite}_quant_${mode}_${device}_${target}.csv" || true
        # Copy cudagraph results as mock data, easiest choice?
        cp "$TEST_REPORTS_DIR/${backend}_with_cudagraphs_${suite}_${dtype}_${mode}_${device}_${target}.csv" \
          "$TEST_REPORTS_DIR/${backend}_cudagraphs_low_precision_${suite}_quant_${mode}_${device}_${target}.csv"
      fi
    done
  done
}

test_single_dynamo_benchmark() {
  # Usage: test_single_dynamo_benchmark inductor_inference huggingface 0 --args-for-script

  # Use test-reports directory under test folder will allow the CI to automatically pick up
  # the test reports and upload them to S3. Need to use full path here otherwise the script
  # will bark about file not found later on
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  local name="$1"
  shift
  local suite="$1"
  shift
  # shard id is mandatory, even if it is not passed
  local shard_id="$1"
  shift

  local partition_flags=()
  if [[ -n "$NUM_TEST_SHARDS" && -n "$shard_id" ]]; then
    partition_flags=( --total-partitions "$NUM_TEST_SHARDS" --partition-id "$shard_id" )
  fi

  if [[ "${TEST_CONFIG}" == *perf_compare* ]]; then
    python "benchmarks/dynamo/$suite.py" \
      --ci --performance --disable-cudagraphs --inductor \
      "${DYNAMO_BENCHMARK_FLAGS[@]}" "$@" "${partition_flags[@]}" \
      --output "$TEST_REPORTS_DIR/${name}_${suite}.csv"
  elif [[ "${TEST_CONFIG}" == *perf* ]]; then
    test_perf_for_dashboard "$suite" \
      "${DYNAMO_BENCHMARK_FLAGS[@]}" "$@" "${partition_flags[@]}"
  else
    if [[ "${TEST_CONFIG}" == *_avx2* ]]; then
      TEST_CONFIG=${TEST_CONFIG//_avx2/}
    fi
    if [[ "${TEST_CONFIG}" == *_avx512* ]]; then
      TEST_CONFIG=${TEST_CONFIG//_avx512/}
    fi
    python "benchmarks/dynamo/$suite.py" \
      --ci --accuracy --timing --explain \
      "${DYNAMO_BENCHMARK_FLAGS[@]}" \
      "$@" "${partition_flags[@]}" \
      --output "$TEST_REPORTS_DIR/${name}_${suite}.csv"
    python benchmarks/dynamo/check_accuracy.py \
      --actual "$TEST_REPORTS_DIR/${name}_$suite.csv" \
      --expected "benchmarks/dynamo/ci_expected_accuracy/${TEST_CONFIG}_${name}.csv"
    python benchmarks/dynamo/check_graph_breaks.py \
      --actual "$TEST_REPORTS_DIR/${name}_$suite.csv" \
      --expected "benchmarks/dynamo/ci_expected_accuracy/${TEST_CONFIG}_${name}.csv"
  fi
}

test_inductor_micro_benchmark() {
  # torchao requires cuda 8.0 or above for bfloat16 support
  if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
  fi
  install_torchao
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  if [[ "${TEST_CONFIG}" == *cpu* ]]; then
    test_inductor_set_cpu_affinity
  fi
  python benchmarks/gpt_fast/benchmark.py --output "${TEST_REPORTS_DIR}/gpt_fast_benchmark.csv"
}

test_inductor_halide() {
  python test/run_test.py --include inductor/test_halide.py --verbose
  assert_git_not_dirty
}

test_inductor_triton_cpu() {
  python test/run_test.py --include inductor/test_triton_cpu_backend.py --verbose
  assert_git_not_dirty
}

test_dynamo_benchmark() {
  # Usage: test_dynamo_benchmark huggingface 0
  TEST_REPORTS_DIR=$(pwd)/test/test-reports

  local suite="$1"
  shift
  local shard_id="$1"
  shift

  if [[ "${TEST_CONFIG}" == *perf_compare* ]]; then
    test_single_dynamo_benchmark "training" "$suite" "$shard_id" --training --amp "$@"
  elif [[ "${TEST_CONFIG}" == *perf* ]]; then
    test_single_dynamo_benchmark "dashboard" "$suite" "$shard_id" "$@"
  else
    if [[ "${TEST_CONFIG}" == *cpu* ]]; then
      local dt="float32"
      if [[ "${TEST_CONFIG}" == *amp* ]]; then
        dt="amp"
      fi
      if [[ "${TEST_CONFIG}" == *freezing* ]]; then
        test_single_dynamo_benchmark "inference" "$suite" "$shard_id" --inference --"$dt" --freezing "$@"
      else
        test_single_dynamo_benchmark "inference" "$suite" "$shard_id" --inference --"$dt" "$@"
      fi
    elif [[ "${TEST_CONFIG}" == *aot_inductor* ]]; then
      test_single_dynamo_benchmark "inference" "$suite" "$shard_id" --inference --bfloat16 "$@"
    else
      test_single_dynamo_benchmark "inference" "$suite" "$shard_id" --inference --bfloat16 "$@"
      test_single_dynamo_benchmark "training" "$suite" "$shard_id" --training --amp "$@"
    fi
  fi
}

test_inductor_torchbench_smoketest_perf() {
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  python benchmarks/dynamo/torchbench.py --device cuda --performance --backend inductor --float16 --training \
    --batch-size-file "$(realpath benchmarks/dynamo/torchbench_models_list.txt)" --only hf_Bert \
    --output "$TEST_REPORTS_DIR/inductor_training_smoketest.csv"
  # The threshold value needs to be actively maintained to make this check useful
  python benchmarks/dynamo/check_perf_csv.py -f "$TEST_REPORTS_DIR/inductor_training_smoketest.csv" -t 1.4

  # Check memory compression ratio for a few models
  for test in hf_Albert timm_vision_transformer; do
    python benchmarks/dynamo/torchbench.py --device cuda --performance --backend inductor --amp --training \
      --disable-cudagraphs --batch-size-file "$(realpath benchmarks/dynamo/torchbench_models_list.txt)" \
      --only $test --output "$TEST_REPORTS_DIR/inductor_training_smoketest_$test.csv"
    cat "$TEST_REPORTS_DIR/inductor_training_smoketest_$test.csv"
    python benchmarks/dynamo/check_memory_compression_ratio.py --actual \
      "$TEST_REPORTS_DIR/inductor_training_smoketest_$test.csv" \
      --expected benchmarks/dynamo/expected_ci_perf_inductor_torchbench.csv
  done

  # Perform some "warm-start" runs for a few huggingface models.
  for test in AlbertForQuestionAnswering AllenaiLongformerBase DistilBertForMaskedLM DistillGPT2 GoogleFnet YituTechConvBert; do
    python benchmarks/dynamo/huggingface.py --accuracy --training --amp --inductor --device cuda --warm-start-latency \
      --only $test --output "$TEST_REPORTS_DIR/inductor_warm_start_smoketest_$test.csv"
    python benchmarks/dynamo/check_accuracy.py \
      --actual "$TEST_REPORTS_DIR/inductor_warm_start_smoketest_$test.csv" \
      --expected "benchmarks/dynamo/ci_expected_accuracy/inductor_huggingface_training.csv"
  done
}

test_inductor_get_core_number() {
  if [[ "${TEST_CONFIG}" == *aarch64* ]]; then
    echo "$(($(lscpu | grep 'Cluster(s):' | awk '{print $2}') * $(lscpu | grep 'Core(s) per cluster:' | awk '{print $4}')))"
  else
    echo "$(($(lscpu | grep 'Socket(s):' | awk '{print $2}') * $(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')))"
  fi
}

test_inductor_set_cpu_affinity(){
  #set jemalloc
  JEMALLOC_LIB="$(find /usr/lib -name libjemalloc.so.2)"
  export LD_PRELOAD="$JEMALLOC_LIB":"$LD_PRELOAD"
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

  if [[ "${TEST_CONFIG}" != *aarch64* ]]; then
    # Use Intel OpenMP for x86
    IOMP_LIB="$(dirname "$(which python)")/../lib/libiomp5.so"
    export LD_PRELOAD="$IOMP_LIB":"$LD_PRELOAD"
    export KMP_AFFINITY=granularity=fine,compact,1,0
    export KMP_BLOCKTIME=1
  fi
  cores=$(test_inductor_get_core_number)
  # Set number of cores to 16 on Aarch64 for performance runs.
  if [[ "${TEST_CONFIG}" == *aarch64* && $cores -gt 16 ]]; then
    cores=16
  fi
  export OMP_NUM_THREADS=$cores
  end_core=$((cores-1))
  export TASKSET="taskset -c 0-$end_core"
}

test_inductor_torchbench_cpu_smoketest_perf(){
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  test_inductor_set_cpu_affinity
  MODELS_SPEEDUP_TARGET=benchmarks/dynamo/expected_ci_speedup_inductor_torchbench_cpu.csv

  grep -v '^ *#' < "$MODELS_SPEEDUP_TARGET" | while IFS=',' read -r -a model_cfg
  do
    local model_name=${model_cfg[0]}
    local data_type=${model_cfg[2]}
    local speedup_target=${model_cfg[5]}
    local backend=${model_cfg[1]}
    if [[ ${model_cfg[4]} == "cpp" ]]; then
      export TORCHINDUCTOR_CPP_WRAPPER=1
    else
      unset TORCHINDUCTOR_CPP_WRAPPER
    fi
    local output_name="$TEST_REPORTS_DIR/inductor_inference_${model_cfg[0]}_${model_cfg[1]}_${model_cfg[2]}_${model_cfg[3]}_cpu_smoketest.csv"

    if [[ ${model_cfg[3]} == "dynamic" ]]; then
      $TASKSET python benchmarks/dynamo/torchbench.py \
        --inference --performance --"$data_type" -dcpu -n50 --only "$model_name" --dynamic-shapes \
        --dynamic-batch-only --freezing --timeout 9000 --"$backend" --output "$output_name"
    else
      $TASKSET python benchmarks/dynamo/torchbench.py \
        --inference --performance --"$data_type" -dcpu -n50 --only "$model_name" \
        --freezing --timeout 9000 --"$backend" --output "$output_name"
    fi
    cat "$output_name"
    # The threshold value needs to be actively maintained to make this check useful.
    # Allow 1% variance for CPU perf to accommodate perf fluctuation
    python benchmarks/dynamo/check_perf_csv.py -f "$output_name" -t "$speedup_target" -s 0.99
  done
}

test_torchbench_gcp_smoketest(){
  pushd "${TORCHBENCHPATH}"
  python test.py -v
  popd
}

test_python_gloo_with_tls() {
  source "$(dirname "${BASH_SOURCE[0]}")/run_glootls_test.sh"
  assert_git_not_dirty
}


test_aten() {
  # Test ATen
  # The following test(s) of ATen have already been skipped by caffe2 in rocm environment:
  # scalar_tensor_test, basic, native_test
  echo "Running ATen tests with pytorch lib"

  if [[ -n "$IN_WHEEL_TEST" ]]; then
    echo "Running test with the install folder"
    # Rename the build folder when running test to ensure it
    # is not depended on the folder
    mv "$BUILD_DIR" "$BUILD_RENAMED_DIR"
    TEST_BASE_DIR="$TORCH_TEST_DIR"
  else
    echo "Running test with the build folder"
    TEST_BASE_DIR="$BUILD_BIN_DIR"
  fi

  # NB: the ATen test binaries don't have RPATH set, so it's necessary to
  # put the dynamic libraries somewhere were the dynamic linker can find them.
  # This is a bit of a hack.
  ${SUDO} ln -sf "$TORCH_LIB_DIR"/libc10* "$TEST_BASE_DIR"
  ${SUDO} ln -sf "$TORCH_LIB_DIR"/libcaffe2* "$TEST_BASE_DIR"
  ${SUDO} ln -sf "$TORCH_LIB_DIR"/libmkldnn* "$TEST_BASE_DIR"
  ${SUDO} ln -sf "$TORCH_LIB_DIR"/libnccl* "$TEST_BASE_DIR"
  ${SUDO} ln -sf "$TORCH_LIB_DIR"/libtorch* "$TEST_BASE_DIR"

  ls "$TEST_BASE_DIR"
  aten/tools/run_tests.sh "$TEST_BASE_DIR"

  if [[ -n "$IN_WHEEL_TEST" ]]; then
    # Restore the build folder to avoid any impact on other tests
    mv "$BUILD_RENAMED_DIR" "$BUILD_DIR"
  fi

  assert_git_not_dirty
}

test_without_numpy() {
  pushd "$(dirname "${BASH_SOURCE[0]}")"
  python -c "import sys;sys.path.insert(0, 'fake_numpy');from unittest import TestCase;import torch;x=torch.randn(3,3);TestCase().assertRaises(RuntimeError, lambda: x.numpy())"
  # Regression test for https://github.com/pytorch/pytorch/issues/66353
  python -c "import sys;sys.path.insert(0, 'fake_numpy');import torch;print(torch.tensor([torch.tensor(0.), torch.tensor(1.)]))"
  # Regression test for https://github.com/pytorch/pytorch/issues/109387
  if [[ "${TEST_CONFIG}" == *dynamo_wrapped* ]]; then
    python -c "import sys;sys.path.insert(0, 'fake_numpy');import torch;torch.compile(lambda x:print(x))('Hello World')"
  fi
  popd
}

test_libtorch() {
  local SHARD="$1"

  # The slow test config corresponds to a default test config that should run
  # the libtorch tests instead.
  if [[ "$TEST_CONFIG" != "slow" ]]; then
    echo "Testing libtorch"
    ln -sf "$TORCH_LIB_DIR"/libbackend_with_compiler.so "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libjitbackend_test.so "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libcaffe2_nvrtc.so "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libshm* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libnvfuser* "$TORCH_BIN_DIR"

    export CPP_TESTS_DIR="${TORCH_BIN_DIR}"

    if [[ -z "${SHARD}" || "${SHARD}" == "1" ]]; then
      test_libtorch_api
    fi

    if [[ -z "${SHARD}" || "${SHARD}" == "2" ]]; then
      test_libtorch_jit
    fi

    assert_git_not_dirty
  fi
}

test_libtorch_jit() {
  # Prepare the model used by test_jit, the model needs to be in the test directory
  # to get picked up by run_test
  pushd test
  python cpp/jit/tests_setup.py setup
  popd

  # Run jit and lazy tensor cpp tests together to finish them faster
  if [[ "$BUILD_ENVIRONMENT" == *cuda* && "$TEST_CONFIG" != *nogpu* ]]; then
    LTC_TS_CUDA=1 python test/run_test.py --cpp --verbose -i cpp/test_jit cpp/test_lazy
  else
    # CUDA tests have already been skipped when CUDA is not available
    python test/run_test.py --cpp --verbose -i cpp/test_jit cpp/test_lazy -k "not CUDA"
  fi

  # Cleaning up test artifacts in the test folder
  pushd test
  python cpp/jit/tests_setup.py shutdown
  popd
}

test_libtorch_api() {
  # Start background download
  MNIST_DIR="${PWD}/test/cpp/api/mnist"
  python tools/download_mnist.py --quiet -d "${MNIST_DIR}"

  if [[ "$BUILD_ENVIRONMENT" == *asan* || "$BUILD_ENVIRONMENT" == *slow-gradcheck* ]]; then
    TEST_REPORTS_DIR=test/test-reports/cpp-unittest/test_libtorch
    mkdir -p $TEST_REPORTS_DIR

    OMP_NUM_THREADS=2 TORCH_CPP_TEST_MNIST_PATH="${MNIST_DIR}" "$TORCH_BIN_DIR"/test_api --gtest_filter='-IMethodTest.*' --gtest_output=xml:$TEST_REPORTS_DIR/test_api.xml
    "$TORCH_BIN_DIR"/test_tensorexpr --gtest_output=xml:$TEST_REPORTS_DIR/test_tensorexpr.xml
  else
    # Exclude IMethodTest that relies on torch::deploy, which will instead be ran in test_deploy
    OMP_NUM_THREADS=2 TORCH_CPP_TEST_MNIST_PATH="${MNIST_DIR}" python test/run_test.py --cpp --verbose -i cpp/test_api -k "not IMethodTest"
    python test/run_test.py --cpp --verbose -i cpp/test_tensorexpr
  fi

  if [[ "${BUILD_ENVIRONMENT}" != *android* && "${BUILD_ENVIRONMENT}" != *cuda* && "${BUILD_ENVIRONMENT}" != *asan* ]]; then
    # NB: This test is not under TORCH_BIN_DIR but under BUILD_BIN_DIR
    export CPP_TESTS_DIR="${BUILD_BIN_DIR}"
    python test/run_test.py --cpp --verbose -i cpp/static_runtime_test
  fi
}

test_xpu_bin(){
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  for xpu_case in "${BUILD_BIN_DIR}"/*{xpu,sycl}*; do
    if [[ "$xpu_case" != *"*"* && "$xpu_case" != *.so && "$xpu_case" != *.a ]]; then
      case_name=$(basename "$xpu_case")
      echo "Testing ${case_name} ..."
      "$xpu_case" --gtest_output=xml:"$TEST_REPORTS_DIR"/"$case_name".xml
    fi
  done
}

test_aot_compilation() {
  echo "Testing Ahead of Time compilation"
  ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"
  ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"

  if [ -f "$TORCH_BIN_DIR"/test_mobile_nnc ]; then
    CPP_TESTS_DIR="${TORCH_BIN_DIR}" python test/run_test.py --cpp --verbose -i cpp/test_mobile_nnc
  fi

  if [ -f "$TORCH_BIN_DIR"/aot_model_compiler_test ]; then
    source test/mobile/nnc/test_aot_compile.sh
  fi
}

test_vulkan() {
  if [[ "$BUILD_ENVIRONMENT" == *vulkan* ]]; then
    ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_TEST_DIR"
    ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_TEST_DIR"
    export VK_ICD_FILENAMES=/var/lib/jenkins/swiftshader/swiftshader/build/Linux/vk_swiftshader_icd.json
    CPP_TESTS_DIR="${TORCH_TEST_DIR}" LD_LIBRARY_PATH=/var/lib/jenkins/swiftshader/swiftshader/build/Linux/ python test/run_test.py --cpp --verbose -i cpp/vulkan_api_test
  fi
}

test_distributed() {
  echo "Testing distributed python tests"
  # shellcheck disable=SC2086
  time python test/run_test.py --distributed-tests --shard "$SHARD_NUMBER" "$NUM_TEST_SHARDS" $INCLUDE_CLAUSE --verbose
  assert_git_not_dirty

  if [[ ("$BUILD_ENVIRONMENT" == *cuda* || "$BUILD_ENVIRONMENT" == *rocm*) && "$SHARD_NUMBER" == 1 ]]; then
    echo "Testing distributed C++ tests"
    ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"

    export CPP_TESTS_DIR="${TORCH_BIN_DIR}"
    # These are distributed tests, so let's continue running them sequentially here to avoid
    # any surprise
    python test/run_test.py --cpp --verbose -i cpp/FileStoreTest
    python test/run_test.py --cpp --verbose -i cpp/HashStoreTest
    python test/run_test.py --cpp --verbose -i cpp/TCPStoreTest

    echo "Testing multi-GPU linalg tests"
    python test/run_test.py -i test_linalg.py -k test_matmul_offline_mgpu_tunable --verbose

    if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
      MPIEXEC=$(command -v mpiexec)
      if [[ -n "$MPIEXEC" ]]; then
        # NB: mpiexec only works directly with the C++ test binary here
        MPICMD="${MPIEXEC} -np 2 $TORCH_BIN_DIR/ProcessGroupMPITest"
        eval "$MPICMD"
      fi

      python test/run_test.py --cpp --verbose -i cpp/ProcessGroupGlooTest
      python test/run_test.py --cpp --verbose -i cpp/ProcessGroupNCCLTest
      python test/run_test.py --cpp --verbose -i cpp/ProcessGroupNCCLErrorsTest
    fi
  fi
}

test_rpc() {
  echo "Testing RPC C++ tests"
  # NB: the ending test_rpc must match the current function name for the current
  # test reporting process to function as expected.
  ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"
  ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"

  CPP_TESTS_DIR="${TORCH_BIN_DIR}" python test/run_test.py --cpp --verbose -i cpp/test_cpp_rpc
}

test_custom_backend() {
  echo "Testing custom backends"
  CUSTOM_BACKEND_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-backend-build"
  pushd test/custom_backend
  cp -a "$CUSTOM_BACKEND_BUILD" build
  # Run tests Python-side and export a lowered module.
  python test_custom_backend.py -v
  python backend.py --export-module-to=model.pt
  # Run tests C++-side and load the exported lowered module.
  build/test_custom_backend ./model.pt
  rm -f ./model.pt
  popd
  assert_git_not_dirty
}

test_custom_script_ops() {
  echo "Testing custom script operators"
  CUSTOM_OP_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-op-build"
  pushd test/custom_operator
  cp -a "$CUSTOM_OP_BUILD" build
  # Run tests Python-side and export a script module.
  python test_custom_ops.py -v
  python model.py --export-script-module=model.pt
  # Run tests C++-side and load the exported script module.
  build/test_custom_ops ./model.pt
  popd
  assert_git_not_dirty
}

test_jit_hooks() {
  echo "Testing jit hooks in cpp"
  HOOK_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/jit-hook-build"
  pushd test/jit_hooks
  cp -a "$HOOK_BUILD" build
  # Run tests Python-side and export the script modules with hooks
  python model.py --export-script-module=model
  # Run tests C++-side and load the exported script modules
  build/test_jit_hooks ./model
  popd
  assert_git_not_dirty
}

test_torch_function_benchmark() {
  echo "Testing __torch_function__ benchmarks"
  pushd benchmarks/overrides_benchmark
  python bench.py -n 1 -m 2
  python pyspybench.py Tensor -n 1
  python pyspybench.py SubTensor -n 1
  python pyspybench.py WithTorchFunction -n 1
  python pyspybench.py SubWithTorchFunction -n 1
  popd
  assert_git_not_dirty
}

build_xla() {
  # xla test needs pytorch headers in torch/include
  pushd ..
  python -c "import os, torch, shutil; shutil.copytree(os.path.join(os.path.dirname(torch.__file__), 'include'), 'workspace/torch/include', dirs_exist_ok=True)"
  popd

  # xla test needs sccache setup.
  # shellcheck source=./common-build.sh
  source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

  XLA_DIR=xla
  USE_CACHE=1
  clone_pytorch_xla
  # shellcheck disable=SC1091
  source "xla/.circleci/common.sh"

  # TODO: The torch pin #73164 is involved in the sev https://github.com/pytorch/pytorch/issues/86093
  # so this is temporarily removed until XLA fixes the weird logic in https://github.com/pytorch/xla/blob/master/scripts/apply_patches.sh#L17-L18
  rm "${XLA_DIR}/torch_patches/.torch_pin" || true

  apply_patches
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  # These functions are defined in .circleci/common.sh in pytorch/xla repo
  retry install_deps_pytorch_xla $XLA_DIR $USE_CACHE
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch:${CMAKE_PREFIX_PATH}" XLA_SANDBOX_BUILD=1 build_torch_xla $XLA_DIR
  assert_git_not_dirty
}

test_xla() {
  # xla test needs sccache setup.
  # shellcheck source=./common-build.sh
  source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

  clone_pytorch_xla
  # shellcheck disable=SC1091
  source "./xla/.circleci/common.sh"
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  # Set LD_LIBRARY_PATH for C++ tests
  export LD_LIBRARY_PATH="/opt/conda/lib/:${LD_LIBRARY_PATH}"
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch:${CMAKE_PREFIX_PATH}" XLA_SKIP_MP_OP_TESTS=1 run_torch_xla_tests "$(pwd)" "$(pwd)/xla"
  assert_git_not_dirty
}

function check_public_api_test_fails {
    test_name=$1
    invalid_item_name=$2
    invalid_item_desc=$3

    echo "Running public API test '${test_name}'..."
    test_output=$(python test/test_public_bindings.py -k "${test_name}" 2>&1) && ret=$? || ret=$?

    # Ensure test fails correctly.
    if [ "$ret" -eq 0 ]; then
        cat << EOF
Expected the public API test '${test_name}' to fail after introducing
${invalid_item_desc}, but it succeeded! Check test/test_public_bindings.py
for any changes that may have broken the test.
EOF
        return 1
    fi

    # Ensure invalid item is in the test output.
    echo "${test_output}" | grep -q "${invalid_item_name}" && ret=$? || ret=$?

    if [ $ret -ne 0 ]; then
        cat << EOF
Expected the public API test '${test_name}' to identify ${invalid_item_desc}, but
it didn't! It's possible the test may not have run. Check test/test_public_bindings.py
for any changes that may have broken the test.
EOF
        return 1
    fi

    echo "Success! '${test_name}' identified ${invalid_item_desc} ${invalid_item_name}."
    return 0
}

# Do NOT run this test before any other tests, like test_python_shard, etc.
# Because this function uninstalls the torch built from branch and installs
# the torch built on its base commit.
test_forward_backward_compatibility() {
  set -x

  # First, validate public API tests in the torch built from branch.
  # Step 1. Make sure the public API test "test_correct_module_names" fails when a new file
  # introduces an invalid public API function.
  new_filename=$(mktemp XXXXXXXX.py -p "${TORCH_INSTALL_DIR}")

  BAD_PUBLIC_FUNC=$(
  cat << 'EOF'
def new_public_func():
  pass

# valid public API functions have __module__ set correctly
new_public_func.__module__ = None
EOF
  )

  echo "${BAD_PUBLIC_FUNC}" >> "${new_filename}"
  invalid_api="torch.$(basename -s '.py' "${new_filename}").new_public_func"
  echo "Created an invalid public API function ${invalid_api}..."

  check_public_api_test_fails \
      "test_correct_module_names" \
      "${invalid_api}" \
      "an invalid public API function" && ret=$? || ret=$?

  rm -v "${new_filename}"

  if [ "$ret" -ne 0 ]; then
      exit 1
  fi

  # Step 2. Make sure that the public API test "test_correct_module_names" fails when an existing
  # file is modified to introduce an invalid public API function.
  EXISTING_FILEPATH="${TORCH_INSTALL_DIR}/nn/parameter.py"
  cp -v "${EXISTING_FILEPATH}" "${EXISTING_FILEPATH}.orig"
  echo "${BAD_PUBLIC_FUNC}" >> "${EXISTING_FILEPATH}"
  invalid_api="torch.nn.parameter.new_public_func"
  echo "Appended an invalid public API function to existing file ${EXISTING_FILEPATH}..."

  check_public_api_test_fails \
      "test_correct_module_names" \
      "${invalid_api}" \
      "an invalid public API function" && ret=$? || ret=$?

  mv -v "${EXISTING_FILEPATH}.orig" "${EXISTING_FILEPATH}"

  if [ "$ret" -ne 0 ]; then
      exit 1
  fi

  # Step 3. Make sure that the public API test "test_modules_can_be_imported" fails when a module
  # cannot be imported.
  new_module_dir=$(mktemp XXXXXXXX -d -p "${TORCH_INSTALL_DIR}")
  echo "invalid syntax garbage" > "${new_module_dir}/__init__.py"
  invalid_module_name="torch.$(basename "${new_module_dir}")"

  check_public_api_test_fails \
      "test_modules_can_be_imported" \
      "${invalid_module_name}" \
      "a non-importable module" && ret=$? || ret=$?

  rm -rv "${new_module_dir}"

  if [ "$ret" -ne 0 ]; then
      exit 1
  fi

  # Next, build torch from the merge base.
  REPO_DIR=$(pwd)
  if [[ "${BASE_SHA}" == "${SHA1}" ]]; then
    echo "On trunk, we should compare schemas with torch built from the parent commit"
    SHA_TO_COMPARE=$(git rev-parse "${SHA1}"^)
  else
    echo "On pull, we should compare schemas with torch built from the merge base"
    SHA_TO_COMPARE=$(git merge-base "${SHA1}" "${BASE_SHA}")
  fi
  export SHA_TO_COMPARE

  # create a dummy ts model at this version
  python test/create_dummy_torchscript_model.py /tmp/model_new.pt
  python -m venv venv
  # shellcheck disable=SC1091
  . venv/bin/activate

  # build torch at the base commit to generate a base function schema for comparison
  git reset --hard "${SHA_TO_COMPARE}"
  git submodule sync && git submodule update --init --recursive
  echo "::group::Installing Torch From Base Commit"
  pip3 install -r requirements.txt
  # shellcheck source=./common-build.sh
  source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"
  python setup.py bdist_wheel --bdist-dir="base_bdist_tmp" --dist-dir="base_dist"
  python -mpip install base_dist/*.whl
  echo "::endgroup::"

  pushd test/forward_backward_compatibility
  pip show torch
  python dump_all_function_schemas.py --filename nightly_schemas.txt

  git reset --hard "${SHA1}"
  git submodule sync && git submodule update --init --recursive
  # FC: verify new model can be load with old code.
  if ! python ../load_torchscript_model.py /tmp/model_new.pt; then
      echo "FC check failed: new model cannot be load in old code"
      return 1
  fi
  python ../create_dummy_torchscript_model.py /tmp/model_old.pt
  deactivate
  rm -r "${REPO_DIR}/venv" "${REPO_DIR}/base_dist"
  pip show torch
  python check_forward_backward_compatibility.py --existing-schemas nightly_schemas.txt
  # BC: verify old model can be load with new code
  if ! python ../load_torchscript_model.py /tmp/model_old.pt; then
      echo "BC check failed: old model cannot be load in new code"
      return 1
  fi
  popd
  set +x
  assert_git_not_dirty
}

test_bazel() {
  set -e -o pipefail

  # bazel test needs sccache setup.
  # shellcheck source=./common-build.sh
  source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

  get_bazel

  if [[ "$CUDA_VERSION" == "cpu" ]]; then
    # Test //c10/... without Google flags and logging libraries. The
    # :all_tests target in the subsequent Bazel invocation tests
    # //c10/... with the Google libraries.
    tools/bazel test --config=cpu-only --test_timeout=480 --test_output=all --test_tag_filters=-gpu-required --test_filter=-*CUDA \
      --no//c10:use_gflags --no//c10:use_glog //c10/...

    tools/bazel test --config=cpu-only --test_timeout=480 --test_output=all --test_tag_filters=-gpu-required --test_filter=-*CUDA :all_tests
  else
    # Increase the test timeout to 480 like CPU tests because modules_test frequently timeout
    tools/bazel test --test_timeout=480 --test_output=errors \
      //:any_test \
      //:autograd_test \
      //:dataloader_test \
      //:dispatch_test \
      //:enum_test \
      //:expanding_array_test \
      //:fft_test \
      //:functional_test \
      //:grad_mode_test \
      //:inference_mode_test \
      //:init_test \
      //:jit_test \
      //:memory_test \
      //:meta_tensor_test \
      //:misc_test \
      //:moduledict_test \
      //:modulelist_test \
      //:modules_test \
      //:namespace_test \
      //:nested_test \
      //:nn_utils_test \
      //:operations_test \
      //:ordered_dict_test \
      //:parallel_benchmark_test \
      //:parameterdict_test \
      //:parameterlist_test \
      //:sequential_test \
      //:serialize_test \
      //:special_test \
      //:static_test \
      //:support_test \
      //:tensor_flatten_test \
      //:tensor_indexing_test \
      //:tensor_options_cuda_test \
      //:tensor_options_test \
      //:tensor_test \
      //:torch_dist_autograd_test \
      //:torch_include_test \
      //:transformer_test \
      //:test_bazel \
      //c10/cuda/test:test \
      //c10/test:core_tests \
      //c10/test:typeid_test \
      //c10/test:util/ssize_test \
      //c10/test:util_base_tests
  fi
}

test_benchmarks() {
  if [[ "$BUILD_ENVIRONMENT" == *cuda* && $TEST_CONFIG != *nogpu* ]]; then
    pip_install --user "pytest-benchmark==3.2.3"
    pip_install --user "requests"
    BENCHMARK_DATA="benchmarks/.data"
    mkdir -p ${BENCHMARK_DATA}
    pytest benchmarks/fastrnns/test_bench.py --benchmark-sort=Name --benchmark-json=${BENCHMARK_DATA}/fastrnns_default.json --fuser=default --executor=default
    pytest benchmarks/fastrnns/test_bench.py --benchmark-sort=Name --benchmark-json=${BENCHMARK_DATA}/fastrnns_legacy_old.json --fuser=old --executor=legacy
    pytest benchmarks/fastrnns/test_bench.py --benchmark-sort=Name --benchmark-json=${BENCHMARK_DATA}/fastrnns_profiling_te.json --fuser=te --executor=profiling
    # TODO: Enable these for GHA once we have credentials for forked pull requests
    if [[ -z "${GITHUB_ACTIONS}" ]]; then
      python benchmarks/upload_scribe.py --pytest_bench_json ${BENCHMARK_DATA}/fastrnns_default.json
      python benchmarks/upload_scribe.py --pytest_bench_json ${BENCHMARK_DATA}/fastrnns_legacy_old.json
      python benchmarks/upload_scribe.py --pytest_bench_json ${BENCHMARK_DATA}/fastrnns_profiling_te.json
    fi
    assert_git_not_dirty
  fi
}

test_cpp_extensions() {
  # This is to test whether cpp extension build is compatible with current env. No need to test both ninja and no-ninja build
  time python test/run_test.py --include test_cpp_extensions_aot_ninja --verbose
  assert_git_not_dirty
}

test_vec256() {
  # This is to test vec256 instructions DEFAULT/AVX/AVX2 (platform dependent, some platforms might not support AVX/AVX2)
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]]; then
    echo "Testing vec256 instructions"
    mkdir -p test/test-reports/vec256
    pushd build/bin
    vec256_tests=$(find . -maxdepth 1 -executable -name 'vec256_test*')
    for vec256_exec in $vec256_tests
    do
      $vec256_exec --gtest_output=xml:test/test-reports/vec256/"$vec256_exec".xml
    done
    popd
    assert_git_not_dirty
  fi
}

test_docs_test() {
  .ci/pytorch/docs-test.sh
}

test_executorch() {
  echo "Install torchvision and torchaudio"
  install_torchvision
  install_torchaudio

  pushd /executorch

  export PYTHON_EXECUTABLE=python
  export EXECUTORCH_BUILD_PYBIND=ON
  export CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON"

  # For llama3
  bash examples/models/llama3_2_vision/install_requirements.sh
  # NB: We need to rebuild ExecuTorch runner here because it depends on PyTorch
  # from the PR
  bash .ci/scripts/setup-linux.sh cmake

  echo "Run ExecuTorch unit tests"
  pytest -v -n auto
  # shellcheck disable=SC1091
  LLVM_PROFDATA=llvm-profdata-12 LLVM_COV=llvm-cov-12 bash test/run_oss_cpp_tests.sh

  echo "Run ExecuTorch regression tests for some models"
  # TODO(huydhn): Add more coverage here using ExecuTorch's gather models script
  # shellcheck disable=SC1091
  source .ci/scripts/test_model.sh mv3 cmake xnnpack-quantization-delegation ''

  popd

  # Test torchgen generated code for Executorch.
  echo "Testing ExecuTorch op registration"
  "$BUILD_BIN_DIR"/test_edge_op_registration

  assert_git_not_dirty
}

test_linux_aarch64() {
  python test/run_test.py --include test_modules test_mkldnn test_mkldnn_fusion test_openmp test_torch test_dynamic_shapes \
        test_transformers test_multiprocessing test_numpy_interop test_autograd test_binary_ufuncs test_complex test_spectral_ops \
        test_foreach test_reductions test_unary_ufuncs \
        --shard "$SHARD_NUMBER" "$NUM_TEST_SHARDS" --verbose

  # Dynamo tests
  python test/run_test.py --include dynamo/test_compile dynamo/test_backends dynamo/test_comptime dynamo/test_config \
       dynamo/test_functions dynamo/test_fx_passes_pre_grad dynamo/test_interop dynamo/test_model_output dynamo/test_modules \
       dynamo/test_optimizers dynamo/test_recompile_ux dynamo/test_recompiles \
       --shard "$SHARD_NUMBER" "$NUM_TEST_SHARDS" --verbose

  # Inductor tests
  python test/run_test.py --include inductor/test_torchinductor inductor/test_benchmark_fusion inductor/test_codecache \
       inductor/test_config inductor/test_control_flow inductor/test_coordinate_descent_tuner inductor/test_fx_fusion \
       inductor/test_group_batch_fusion inductor/test_inductor_freezing inductor/test_inductor_utils \
       inductor/test_inplacing_pass inductor/test_kernel_benchmark inductor/test_layout_optim \
       inductor/test_max_autotune inductor/test_memory_planning inductor/test_metrics inductor/test_multi_kernel inductor/test_pad_mm \
       inductor/test_pattern_matcher inductor/test_perf inductor/test_profiler inductor/test_select_algorithm inductor/test_smoke \
       inductor/test_split_cat_fx_passes inductor/test_standalone_compile inductor/test_torchinductor \
       inductor/test_torchinductor_codegen_dynamic_shapes inductor/test_torchinductor_dynamic_shapes inductor/test_memory \
       inductor/test_triton_cpu_backend inductor/test_triton_extension_backend inductor/test_mkldnn_pattern_matcher inductor/test_cpu_cpp_wrapper \
       --shard "$SHARD_NUMBER" "$NUM_TEST_SHARDS" --verbose
}

if ! [[ "${BUILD_ENVIRONMENT}" == *libtorch* || "${BUILD_ENVIRONMENT}" == *-bazel-* ]]; then
  (cd test && python -c "import torch; print(torch.__config__.show())")
  (cd test && python -c "import torch; print(torch.__config__.parallel_info())")
fi
if [[ "${TEST_CONFIG}" == *numpy_2* ]]; then
  # Install numpy-2.0.2 and compatible scipy & numba versions
  python -mpip install --pre numpy==2.0.2 scipy==1.13.1 numba==0.60.0
  python test/run_test.py --include dynamo/test_functions.py dynamo/test_unspec.py test_binary_ufuncs.py test_fake_tensor.py test_linalg.py test_numpy_interop.py test_tensor_creation_ops.py test_torch.py torch_np/test_basic.py
elif [[ "${BUILD_ENVIRONMENT}" == *aarch64* && "${TEST_CONFIG}" != *perf_cpu_aarch64* ]]; then
  test_linux_aarch64
elif [[ "${TEST_CONFIG}" == *backward* ]]; then
  test_forward_backward_compatibility
  # Do NOT add tests after bc check tests, see its comment.
elif [[ "${TEST_CONFIG}" == *xla* ]]; then
  install_torchvision
  build_xla
  test_xla
elif [[ "${TEST_CONFIG}" == *executorch* ]]; then
  test_executorch
elif [[ "$TEST_CONFIG" == 'jit_legacy' ]]; then
  test_python_legacy_jit
elif [[ "${BUILD_ENVIRONMENT}" == *libtorch* ]]; then
  # TODO: run some C++ tests
  echo "no-op at the moment"
elif [[ "$TEST_CONFIG" == distributed ]]; then
  test_distributed
  # Only run RPC C++ tests on the first shard
  if [[ "${SHARD_NUMBER}" == 1 ]]; then
    test_rpc
  fi
elif [[ "${TEST_CONFIG}" == *inductor_distributed* ]]; then
  test_inductor_distributed
elif [[ "${TEST_CONFIG}" == *inductor-halide* ]]; then
  test_inductor_halide
elif [[ "${TEST_CONFIG}" == *inductor-triton-cpu* ]]; then
  test_inductor_triton_cpu
elif [[ "${TEST_CONFIG}" == *inductor-micro-benchmark* ]]; then
  test_inductor_micro_benchmark
elif [[ "${TEST_CONFIG}" == *huggingface* ]]; then
  install_torchvision
  id=$((SHARD_NUMBER-1))
  test_dynamo_benchmark huggingface "$id"
elif [[ "${TEST_CONFIG}" == *timm* ]]; then
  install_torchvision
  id=$((SHARD_NUMBER-1))
  test_dynamo_benchmark timm_models "$id"
elif [[ "${TEST_CONFIG}" == *torchbench* ]]; then
  if [[ "${TEST_CONFIG}" == *cpu* ]]; then
    install_torchaudio cpu
  else
    install_torchaudio cuda
  fi
  install_torchvision
  TORCH_CUDA_ARCH_LIST="8.0;8.6" pip_install git+https://github.com/pytorch/ao.git
  id=$((SHARD_NUMBER-1))
  # https://github.com/opencv/opencv-python/issues/885
  pip_install opencv-python==4.8.0.74
  if [[ "${TEST_CONFIG}" == *inductor_torchbench_smoketest_perf* ]]; then
    checkout_install_torchbench hf_Bert hf_Albert timm_vision_transformer
    PYTHONPATH=$(pwd)/torchbench test_inductor_torchbench_smoketest_perf
  elif [[ "${TEST_CONFIG}" == *inductor_torchbench_cpu_smoketest_perf* ]]; then
    checkout_install_torchbench timm_vision_transformer phlippe_densenet basic_gnn_edgecnn \
      llama_v2_7b_16h resnet50 timm_efficientnet mobilenet_v3_large timm_resnest \
      functorch_maml_omniglot yolov3 mobilenet_v2 resnext50_32x4d densenet121 mnasnet1_0
    PYTHONPATH=$(pwd)/torchbench test_inductor_torchbench_cpu_smoketest_perf
  elif [[ "${TEST_CONFIG}" == *torchbench_gcp_smoketest* ]]; then
    checkout_install_torchbench
    TORCHBENCHPATH=$(pwd)/torchbench test_torchbench_gcp_smoketest
  else
    checkout_install_torchbench
    # Do this after checkout_install_torchbench to ensure we clobber any
    # nightlies that torchbench may pull in
    if [[ "${TEST_CONFIG}" != *cpu* ]]; then
      install_torchrec_and_fbgemm
    fi
    PYTHONPATH=$(pwd)/torchbench test_dynamo_benchmark torchbench "$id"
  fi
elif [[ "${TEST_CONFIG}" == *inductor_cpp_wrapper* ]]; then
  install_torchaudio cuda
  install_torchvision
  checkout_install_torchbench hf_T5 llama moco
  PYTHONPATH=$(pwd)/torchbench test_inductor_cpp_wrapper_shard "$SHARD_NUMBER"
elif [[ "${TEST_CONFIG}" == *inductor* ]]; then
  install_torchvision
  test_inductor_shard "${SHARD_NUMBER}"
  if [[ "${SHARD_NUMBER}" == 1 ]]; then
    if [[ "${BUILD_ENVIRONMENT}" != linux-jammy-py3.9-gcc11-build ]]; then
      test_inductor_distributed
    fi
  fi
elif [[ "${TEST_CONFIG}" == *dynamo_wrapped* ]]; then
  install_torchvision
  test_dynamo_wrapped_shard "${SHARD_NUMBER}"
  if [[ "${SHARD_NUMBER}" == 1 ]]; then
    test_aten
  fi
elif [[ "${BUILD_ENVIRONMENT}" == *rocm* && -n "$TESTS_TO_INCLUDE" ]]; then
  install_torchvision
  test_python_shard "$SHARD_NUMBER"
  test_aten
elif [[ "${SHARD_NUMBER}" == 1 && $NUM_TEST_SHARDS -gt 1 ]]; then
  test_without_numpy
  install_torchvision
  test_python_shard 1
  test_aten
  test_libtorch 1
  if [[ "${BUILD_ENVIRONMENT}" == *xpu* ]]; then
    test_xpu_bin
  fi
elif [[ "${SHARD_NUMBER}" == 2 && $NUM_TEST_SHARDS -gt 1 ]]; then
  install_torchvision
  test_python_shard 2
  test_libtorch 2
  test_aot_compilation
  test_custom_script_ops
  test_custom_backend
  test_torch_function_benchmark
elif [[ "${SHARD_NUMBER}" -gt 2 ]]; then
  # Handle arbitrary number of shards
  install_torchvision
  test_python_shard "$SHARD_NUMBER"
elif [[ "${BUILD_ENVIRONMENT}" == *vulkan* ]]; then
  test_vulkan
elif [[ "${BUILD_ENVIRONMENT}" == *-bazel-* ]]; then
  test_bazel
elif [[ "${BUILD_ENVIRONMENT}" == *-mobile-lightweight-dispatch* ]]; then
  test_libtorch
elif [[ "${TEST_CONFIG}" = docs_test ]]; then
  test_docs_test
elif [[ "${BUILD_ENVIRONMENT}" == *xpu* ]]; then
  install_torchvision
  test_python
  test_aten
  test_xpu_bin
else
  install_torchvision
  install_monkeytype
  test_python
  test_aten
  test_vec256
  test_libtorch
  test_aot_compilation
  test_custom_script_ops
  test_custom_backend
  test_torch_function_benchmark
  test_benchmarks
fi

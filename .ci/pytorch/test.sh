#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

set -ex

echo "Environment variables:"
env

TORCH_INSTALL_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/torch
TORCH_BIN_DIR="$TORCH_INSTALL_DIR"/bin
TORCH_LIB_DIR="$TORCH_INSTALL_DIR"/lib
TORCH_TEST_DIR="$TORCH_INSTALL_DIR"/test

BUILD_DIR="build"
BUILD_RENAMED_DIR="build_renamed"
BUILD_BIN_DIR="$BUILD_DIR"/bin

export VALGRIND=ON
# export TORCH_INDUCTOR_INSTALL_GXX=ON
if [[ "$BUILD_ENVIRONMENT" == *clang9* ]]; then
  # clang9 appears to miscompile code involving c10::optional<c10::SymInt>,
  # such that valgrind complains along these lines:
  #
  # Conditional jump or move depends on uninitialised value(s)
  #    at 0x40303A: ~optional_base (Optional.h:281)
  #    by 0x40303A: call (Dispatcher.h:448)
  #    by 0x40303A: call(at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>) (basic.cpp:10)
  #    by 0x403700: main (basic.cpp:16)
  #  Uninitialised value was created by a stack allocation
  #    at 0x402AAA: call(at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::SymInt>) (basic.cpp:6)
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
  # Tensor call(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset) {
  #   auto op = c10::Dispatcher::singleton()
  #       .findSchemaOrThrow(at::_ops::as_strided::name, at::_ops::as_strided::overload_name)
  #       .typed<at::_ops::as_strided::schema>();
  #   return op.call(self, size, stride, storage_offset);
  # }
  #
  # int main(int argv) {
  #   Tensor b = empty({3, 4});
  #   auto z = call(b, b.sym_sizes(), b.sym_strides(), c10::nullopt);
  # }
  export VALGRIND=OFF
fi

# Get fully qualified path using realpath
if [[ "$BUILD_ENVIRONMENT" != *bazel* ]]; then
  CUSTOM_TEST_ARTIFACT_BUILD_DIR=$(realpath "${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-"build/custom_test_artifacts"}")
fi


# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

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
fi

if [[ "$TEST_CONFIG" == *crossref* ]]; then
  export PYTORCH_TEST_WITH_CROSSREF=1
fi

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # Print GPU info
  rocminfo
  rocminfo | grep -E 'Name:.*\sgfx|Marketing'
fi

if [[ "$BUILD_ENVIRONMENT" != *-bazel-* ]] ; then
  # JIT C++ extensions require ninja.
  pip_install --user "ninja==1.10.2"
  # ninja is installed in $HOME/.local/bin, e.g., /var/lib/jenkins/.local/bin for CI user jenkins
  # but this script should be runnable by any user, including root
  export PATH="$HOME/.local/bin:$PATH"
fi

# DANGER WILL ROBINSON.  The LD_PRELOAD here could cause you problems
# if you're not careful.  Check this if you made some changes and the
# ASAN test is not working
if [[ "$BUILD_ENVIRONMENT" == *asan* ]]; then
    export ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_stack_use_after_return=1:strict_init_order=true:detect_odr_violation=1:detect_container_overflow=0
    export UBSAN_OPTIONS=print_stacktrace=1
    export PYTORCH_TEST_WITH_ASAN=1
    export PYTORCH_TEST_WITH_UBSAN=1
    # TODO: Figure out how to avoid hard-coding these paths
    export ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-7/bin/llvm-symbolizer
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

    # TODO: get rid of the hardcoded path
    export LD_PRELOAD=/usr/lib/llvm-7/lib/clang/7.0.1/lib/linux/libclang_rt.asan-x86_64.so
    # Increase stack size, because ASAN red zones use more stack
    ulimit -s 81920

    (cd test && python -c "import torch; print(torch.__version__, torch.version.git_version)")
    echo "The next four invocations are expected to crash; if they don't that means ASAN/UBSAN is misconfigured"
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_asan(3)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_ubsan(0)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_vptr_ubsan()")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_aten_asan(3)")
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

  time python test/run_test.py --exclude-jit-executor --exclude-distributed-tests --shard "$1" "$NUM_TEST_SHARDS" --verbose

  assert_git_not_dirty
}

test_python() {
  time python test/run_test.py --exclude-jit-executor --exclude-distributed-tests --verbose
  assert_git_not_dirty
}


test_dynamo_shard() {
  if [[ -z "$NUM_TEST_SHARDS" ]]; then
    echo "NUM_TEST_SHARDS must be defined to run a Python test shard"
    exit 1
  fi
  python tools/dynamo/verify_dynamo.py
  # Temporarily disable test_fx for dynamo pending the investigation on TTS
  # regression in https://github.com/pytorch/torchdynamo/issues/784
  time python test/run_test.py --dynamo \
    --exclude-jit-executor \
    --exclude-distributed-tests \
    --exclude \
      test_autograd \
      test_proxy_tensor \
      test_quantization \
      test_public_bindings \
      test_dataloader \
      test_reductions \
      test_namedtensor \
      test_namedtuple_return_api \
      profiler/test_profiler \
      profiler/test_profiler_tree \
      test_overrides \
      test_python_dispatch \
      test_fx \
      test_package \
      test_legacy_vmap \
      functorch/test_dims \
      functorch/test_aotdispatch \
    --shard "$1" "$NUM_TEST_SHARDS" \
    --verbose
  assert_git_not_dirty
}

test_inductor_distributed() {
  # this runs on both single-gpu and multi-gpu instance. It should be smart about skipping tests that aren't supported
  # with if required # gpus aren't available
  python test/run_test.py --include distributed/test_dynamo_distributed distributed/test_inductor_collectives --verbose
  assert_git_not_dirty
}

test_inductor() {
  python tools/dynamo/verify_dynamo.py
  python test/run_test.py --inductor --include test_modules test_ops test_ops_gradients test_torch --verbose
  # Do not add --inductor for the following inductor unit tests, otherwise we will fail because of nested dynamo state
  python test/run_test.py --include inductor/test_torchinductor inductor/test_torchinductor_opinfo --verbose
}

# "Global" flags for inductor benchmarking controlled by TEST_CONFIG
# For example 'dynamic_aot_eager_torchbench' TEST_CONFIG means we run
# the benchmark script with '--dynamic-shapes --backend aot_eager --device cuda'
# The matrix of test options is specified in .github/workflows/periodic.yml
# and .github/workflows/inductor.yml
DYNAMO_BENCHMARK_FLAGS=()

if [[ "${TEST_CONFIG}" == *dynamo_eager* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--backend eager)
elif [[ "${TEST_CONFIG}" == *aot_eager* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--backend aot_eager)
elif [[ "${TEST_CONFIG}" == *inductor* && "${TEST_CONFIG}" != *perf* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--inductor)
fi

if [[ "${TEST_CONFIG}" == *dynamic* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--dynamic-shapes --dynamic-batch-only)
fi

if [[ "${TEST_CONFIG}" == *cpu_accuracy* ]]; then
  DYNAMO_BENCHMARK_FLAGS+=(--device cpu)
else
  DYNAMO_BENCHMARK_FLAGS+=(--device cuda)
fi

test_perf_for_dashboard() {
  TEST_REPORTS_DIR=$(pwd)/test/test-reports
  mkdir -p "$TEST_REPORTS_DIR"

  local suite="$1"
  shift

  dtype=amp
  backend=inductor
  for mode in inference training; do
    # All the accuracy tests can be skipped once the CI accuracy checking is stable enough
    # Run accuracy test for inductor with different configs
    # --disable-cudagraphs is the default inductor behavior
    # TODO: update here once cudagraphs is turned on as default
    python "benchmarks/dynamo/$suite.py" \
        --accuracy --"$mode" --"$dtype" --backend "$backend" --disable-cudagraphs "$@" \
        --output "$TEST_REPORTS_DIR/${backend}_no_cudagraphs_${suite}_${dtype}_${mode}_cuda_accuracy.csv"
    python "benchmarks/dynamo/$suite.py" \
        --accuracy --"$mode" --"$dtype" --backend "$backend" "$@" \
        --output "$TEST_REPORTS_DIR/${backend}_with_cudagraphs_${suite}_${dtype}_${mode}_cuda_accuracy.csv"
    python "benchmarks/dynamo/$suite.py" \
        --accuracy --"$mode" --"$dtype" --backend "$backend" --dynamic-shapes --dynamic-batch-only --disable-cudagraphs "$@" \
        --output "$TEST_REPORTS_DIR/${backend}_dynamic_${suite}_${dtype}_${mode}_cuda_accuracy.csv"

    # Run performance test
    # Skip dynamo-eager and aot-eager for performance test
    # Run performance test for inductor with different configs
    # TODO: add more configs here, e.g. max-autotune, etc.
    python "benchmarks/dynamo/$suite.py" \
        --performance --cold-start-latency --"$mode" --"$dtype" --backend "$backend" --disable-cudagraphs "$@" \
        --output "$TEST_REPORTS_DIR/${backend}_no_cudagraphs_${suite}_${dtype}_${mode}_cuda_performance.csv"
    python "benchmarks/dynamo/$suite.py" \
        --performance --cold-start-latency --"$mode" --"$dtype" --backend "$backend" "$@" \
        --output "$TEST_REPORTS_DIR/${backend}_with_cudagraphs_${suite}_${dtype}_${mode}_cuda_performance.csv"
    python "benchmarks/dynamo/$suite.py" \
        --performance --cold-start-latency --"$mode" --"$dtype" --backend "$backend" --dynamic-shapes \
        --dynamic-batch-only --disable-cudagraphs "$@" \
        --output "$TEST_REPORTS_DIR/${backend}_dynamic_${suite}_${dtype}_${mode}_cuda_performance.csv"
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
    python "benchmarks/dynamo/$suite.py" \
      --ci --accuracy --timing --explain \
      "${DYNAMO_BENCHMARK_FLAGS[@]}" \
      "$@" "${partition_flags[@]}" \
      --output "$TEST_REPORTS_DIR/${name}_${suite}.csv"

    if [[ "${TEST_CONFIG}" == *inductor* ]] && [[ "${TEST_CONFIG}" != *cpu_accuracy* ]]; then
      # other jobs (e.g. periodic, cpu-accuracy) may have different set of expected models.
      python benchmarks/dynamo/check_accuracy.py \
        --actual "$TEST_REPORTS_DIR/${name}_$suite.csv" \
        --expected "benchmarks/dynamo/ci_expected_accuracy/${TEST_CONFIG}_${name}.csv"
      python benchmarks/dynamo/check_graph_breaks.py \
        --actual "$TEST_REPORTS_DIR/${name}_$suite.csv" \
        --expected "benchmarks/dynamo/ci_expected_accuracy/${TEST_CONFIG}_${name}.csv"
    else
      python benchmarks/dynamo/check_csv.py \
        -f "$TEST_REPORTS_DIR/${name}_${suite}.csv"
    fi
  fi
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
    # Check inference with --float32
    test_single_dynamo_benchmark "inference" "$suite" "$shard_id" --inference --float32 "$@"
    if [[ "${TEST_CONFIG}" != *cpu_accuracy* ]]; then
      # Check training with --amp
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
  # the reference speedup value is hardcoded in check_hf_bert_perf_csv.py
  # this value needs to be actively maintained to make this check useful
  python benchmarks/dynamo/check_hf_bert_perf_csv.py -f "$TEST_REPORTS_DIR/inductor_training_smoketest.csv"

  # Check memory compression ratio for a few models
  for test in hf_Albert timm_efficientdet timm_vision_transformer; do
    python benchmarks/dynamo/torchbench.py --device cuda --performance --backend inductor --amp --training \
      --disable-cudagraphs --batch-size-file "$(realpath benchmarks/dynamo/torchbench_models_list.txt)" \
      --only $test --output "$TEST_REPORTS_DIR/inductor_training_smoketest_$test.csv"
    cat "$TEST_REPORTS_DIR/inductor_training_smoketest_$test.csv"
    python benchmarks/dynamo/check_memory_compression_ratio.py --actual \
      "$TEST_REPORTS_DIR/inductor_training_smoketest_$test.csv" \
      --expected benchmarks/dynamo/expected_ci_perf_inductor_torchbench.csv
  done
}

test_python_gloo_with_tls() {
  source "$(dirname "${BASH_SOURCE[0]}")/run_glootls_test.sh"
  assert_git_not_dirty
}


test_aten() {
  # Test ATen
  # The following test(s) of ATen have already been skipped by caffe2 in rocm environment:
  # scalar_tensor_test, basic, native_test
  if [[ "$BUILD_ENVIRONMENT" != *asan* ]] && [[ "$BUILD_ENVIRONMENT" != *rocm* ]]; then
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
    ${SUDO} ln -sf "$TORCH_LIB_DIR"/libtbb* "$TEST_BASE_DIR"

    ls "$TEST_BASE_DIR"
    aten/tools/run_tests.sh "$TEST_BASE_DIR"

    if [[ -n "$IN_WHEEL_TEST" ]]; then
      # Restore the build folder to avoid any impact on other tests
      mv "$BUILD_RENAMED_DIR" "$BUILD_DIR"
    fi

    assert_git_not_dirty
  fi
}

test_without_numpy() {
  pushd "$(dirname "${BASH_SOURCE[0]}")"
  python -c "import sys;sys.path.insert(0, 'fake_numpy');from unittest import TestCase;import torch;x=torch.randn(3,3);TestCase().assertRaises(RuntimeError, lambda: x.numpy())"
  # Regression test for https://github.com/pytorch/pytorch/issues/66353
  python -c "import sys;sys.path.insert(0, 'fake_numpy');import torch;print(torch.tensor([torch.tensor(0.), torch.tensor(1.)]))"
  popd
}

# pytorch extensions require including torch/extension.h which includes all.h
# which includes utils.h which includes Parallel.h.
# So you can call for instance parallel_for() from your extension,
# but the compilation will fail because of Parallel.h has only declarations
# and definitions are conditionally included Parallel.h(see last lines of Parallel.h).
# I tried to solve it #39612 and #39881 by including Config.h into Parallel.h
# But if Pytorch is built with TBB it provides Config.h
# that has AT_PARALLEL_NATIVE_TBB=1(see #3961 or #39881) and it means that if you include
# torch/extension.h which transitively includes Parallel.h
# which transitively includes tbb.h which is not available!
if [[ "${BUILD_ENVIRONMENT}" == *tbb* ]]; then
  sudo mkdir -p /usr/include/tbb
  sudo cp -r "$PWD"/third_party/tbb/include/tbb/* /usr/include/tbb
fi

test_libtorch() {
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]]; then
    echo "Testing libtorch"
    ln -sf "$TORCH_LIB_DIR"/libbackend_with_compiler.so "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libjitbackend_test.so "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libshm* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libtbb* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libnvfuser* "$TORCH_BIN_DIR"

    # Start background download
    python tools/download_mnist.py --quiet -d test/cpp/api/mnist &

    # Make test_reports directory
    # NB: the ending test_libtorch must match the current function name for the current
    # test reporting process to function as expected.
    TEST_REPORTS_DIR=test/test-reports/cpp-unittest/test_libtorch
    mkdir -p $TEST_REPORTS_DIR

    # Run JIT cpp tests
    python test/cpp/jit/tests_setup.py setup

    if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
      "$TORCH_BIN_DIR"/test_jit  --gtest_output=xml:$TEST_REPORTS_DIR/test_jit.xml
      "$TORCH_BIN_DIR"/nvfuser_tests --gtest_output=xml:$TEST_REPORTS_DIR/nvfuser_tests.xml
    else
      "$TORCH_BIN_DIR"/test_jit  --gtest_filter='-*CUDA' --gtest_output=xml:$TEST_REPORTS_DIR/test_jit.xml
    fi

    # Run Lazy Tensor cpp tests
    if [[ "$BUILD_ENVIRONMENT" == *cuda* && "$TEST_CONFIG" != *nogpu* ]]; then
      LTC_TS_CUDA=1 "$TORCH_BIN_DIR"/test_lazy  --gtest_output=xml:$TEST_REPORTS_DIR/test_lazy.xml
    else
      "$TORCH_BIN_DIR"/test_lazy  --gtest_output=xml:$TEST_REPORTS_DIR/test_lazy.xml
    fi

    python test/cpp/jit/tests_setup.py shutdown

    # Wait for background download to finish
    wait
    # Exclude IMethodTest that relies on torch::deploy, which will instead be ran in test_deploy.
    OMP_NUM_THREADS=2 TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" "$TORCH_BIN_DIR"/test_api --gtest_filter='-IMethodTest.*' --gtest_output=xml:$TEST_REPORTS_DIR/test_api.xml
    "$TORCH_BIN_DIR"/test_tensorexpr --gtest_output=xml:$TEST_REPORTS_DIR/test_tensorexpr.xml

    if [[ "${BUILD_ENVIRONMENT}" != *android* && "${BUILD_ENVIRONMENT}" != *cuda* && "${BUILD_ENVIRONMENT}" != *asan* ]]; then
      # TODO: Consider to run static_runtime_test from $TORCH_BIN_DIR (may need modify build script)
      "$BUILD_BIN_DIR"/static_runtime_test --gtest_output=xml:$TEST_REPORTS_DIR/static_runtime_test.xml
    fi
    assert_git_not_dirty
  fi
}

test_aot_compilation() {
  echo "Testing Ahead of Time compilation"
  ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"
  ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"

  # Make test_reports directory
  # NB: the ending test_libtorch must match the current function name for the current
  # test reporting process to function as expected.
  TEST_REPORTS_DIR=test/test-reports/cpp-unittest/test_aot_compilation
  mkdir -p $TEST_REPORTS_DIR
  if [ -f "$TORCH_BIN_DIR"/test_mobile_nnc ]; then "$TORCH_BIN_DIR"/test_mobile_nnc --gtest_output=xml:$TEST_REPORTS_DIR/test_mobile_nnc.xml; fi
  # shellcheck source=test/mobile/nnc/test_aot_compile.sh
  if [ -f "$TORCH_BIN_DIR"/aot_model_compiler_test ]; then source test/mobile/nnc/test_aot_compile.sh; fi
}

test_vulkan() {
  if [[ "$BUILD_ENVIRONMENT" == *vulkan* ]]; then
    ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_TEST_DIR"
    ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_TEST_DIR"
    export VK_ICD_FILENAMES=/var/lib/jenkins/swiftshader/swiftshader/build/Linux/vk_swiftshader_icd.json
    # NB: the ending test_vulkan must match the current function name for the current
    # test reporting process to function as expected.
    TEST_REPORTS_DIR=test/test-reports/cpp-vulkan/test_vulkan
    mkdir -p $TEST_REPORTS_DIR
    LD_LIBRARY_PATH=/var/lib/jenkins/swiftshader/swiftshader/build/Linux/ "$TORCH_TEST_DIR"/vulkan_api_test --gtest_output=xml:$TEST_REPORTS_DIR/vulkan_test.xml
  fi
}

test_distributed() {
  # Smuggle a few multi-gpu tests here so that we don't have to request another large node
  echo "Testing multi_gpu tests in test_torchinductor"
  pytest test/inductor/test_torchinductor.py -k test_multi_gpu

  echo "Testing distributed python tests"
  time python test/run_test.py --distributed-tests --shard "$SHARD_NUMBER" "$NUM_TEST_SHARDS" --verbose
  assert_git_not_dirty

  if [[ "$BUILD_ENVIRONMENT" == *cuda* && "$SHARD_NUMBER" == 1 ]]; then
    echo "Testing distributed C++ tests"
    ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"
    # NB: the ending test_distributed must match the current function name for the current
    # test reporting process to function as expected.
    TEST_REPORTS_DIR=test/test-reports/cpp-distributed/test_distributed
    mkdir -p $TEST_REPORTS_DIR
    "$TORCH_BIN_DIR"/FileStoreTest --gtest_output=xml:$TEST_REPORTS_DIR/FileStoreTest.xml
    "$TORCH_BIN_DIR"/HashStoreTest --gtest_output=xml:$TEST_REPORTS_DIR/HashStoreTest.xml
    "$TORCH_BIN_DIR"/TCPStoreTest --gtest_output=xml:$TEST_REPORTS_DIR/TCPStoreTest.xml

    MPIEXEC=$(command -v mpiexec)
    if [[ -n "$MPIEXEC" ]]; then
      MPICMD="${MPIEXEC} -np 2 $TORCH_BIN_DIR/ProcessGroupMPITest"
      eval "$MPICMD"
    fi
    "$TORCH_BIN_DIR"/ProcessGroupGlooTest --gtest_output=xml:$TEST_REPORTS_DIR/ProcessGroupGlooTest.xml
    "$TORCH_BIN_DIR"/ProcessGroupNCCLTest --gtest_output=xml:$TEST_REPORTS_DIR/ProcessGroupNCCLTest.xml
    "$TORCH_BIN_DIR"/ProcessGroupNCCLErrorsTest --gtest_output=xml:$TEST_REPORTS_DIR/ProcessGroupNCCLErrorsTest.xml
  fi
}

test_rpc() {
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]]; then
    echo "Testing RPC C++ tests"
    # NB: the ending test_rpc must match the current function name for the current
    # test reporting process to function as expected.
    ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"
    ln -sf "$TORCH_LIB_DIR"/libtbb* "$TORCH_BIN_DIR"
    TEST_REPORTS_DIR=test/test-reports/cpp-rpc/test_rpc
    mkdir -p $TEST_REPORTS_DIR
    "$TORCH_BIN_DIR"/test_cpp_rpc --gtest_output=xml:$TEST_REPORTS_DIR/test_cpp_rpc.xml
  fi
}

test_custom_backend() {
  if [[ "$BUILD_ENVIRONMENT" != *asan* ]] ; then
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
  fi
}

test_custom_script_ops() {
  if [[ "$BUILD_ENVIRONMENT" != *asan* ]] ; then
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
  fi
}

test_jit_hooks() {
  if [[ "$BUILD_ENVIRONMENT" != *asan* ]] ; then
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
  fi
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
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch:${CMAKE_PREFIX_PATH}" XLA_SKIP_MP_OP_TESTS=1 run_torch_xla_tests "$(pwd)" "$(pwd)/xla"
  assert_git_not_dirty
}

# Do NOT run this test before any other tests, like test_python_shard, etc.
# Because this function uninstalls the torch built from branch and installs
# the torch built on its base commit.
test_forward_backward_compatibility() {
  set -x
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
  pip install -r requirements.txt
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
  set -e

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
    tools/bazel test \
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
  if [[ "$BUILD_ENVIRONMENT" != *asan* ]] && [[ "$BUILD_ENVIRONMENT" != *rocm* ]]; then
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
  # Test torchgen generated code for Executorch.
  echo "Testing Executorch op registration"
  "$BUILD_BIN_DIR"/test_edge_op_registration
  assert_git_not_dirty
}

if ! [[ "${BUILD_ENVIRONMENT}" == *libtorch* || "${BUILD_ENVIRONMENT}" == *-bazel-* ]]; then
  (cd test && python -c "import torch; print(torch.__config__.show())")
  (cd test && python -c "import torch; print(torch.__config__.parallel_info())")
fi
if [[ "${TEST_CONFIG}" == *backward* ]]; then
  test_forward_backward_compatibility
  # Do NOT add tests after bc check tests, see its comment.
elif [[ "${TEST_CONFIG}" == *xla* ]]; then
  install_torchvision
  build_xla
  test_xla
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
elif [[ "$TEST_CONFIG" == deploy ]]; then
  checkout_install_torchdeploy
  test_torch_deploy
elif [[ "${TEST_CONFIG}" == *inductor_distributed* ]]; then
  install_huggingface
  test_inductor_distributed
elif [[ "${TEST_CONFIG}" == *huggingface* ]]; then
  install_torchvision
  install_huggingface
  id=$((SHARD_NUMBER-1))
  test_dynamo_benchmark huggingface "$id"
elif [[ "${TEST_CONFIG}" == *timm* ]]; then
  install_torchvision
  install_timm
  id=$((SHARD_NUMBER-1))
  test_dynamo_benchmark timm_models "$id"
elif [[ "${TEST_CONFIG}" == *torchbench* ]]; then
  if [[ "${TEST_CONFIG}" == *cpu_accuracy* ]]; then
    install_torchaudio cpu
  else
    install_torchaudio cuda
  fi
  install_torchtext
  install_torchvision
  id=$((SHARD_NUMBER-1))
  if [[ "${TEST_CONFIG}" == *inductor_torchbench_smoketest_perf* ]]; then
    checkout_install_torchbench hf_Bert hf_Albert timm_efficientdet timm_vision_transformer
    PYTHONPATH=$(pwd)/torchbench test_inductor_torchbench_smoketest_perf
  else
    checkout_install_torchbench
    PYTHONPATH=$(pwd)/torchbench test_dynamo_benchmark torchbench "$id"
  fi
elif [[ "${TEST_CONFIG}" == *inductor* && "${SHARD_NUMBER}" == 1 ]]; then
  install_torchvision
  test_inductor
  test_inductor_distributed
elif [[ "${TEST_CONFIG}" == *dynamo* && "${SHARD_NUMBER}" == 1 && $NUM_TEST_SHARDS -gt 1 ]]; then
  test_without_numpy
  install_torchvision
  test_dynamo_shard 1
  test_aten
elif [[ "${TEST_CONFIG}" == *dynamo* && "${SHARD_NUMBER}" == 2 && $NUM_TEST_SHARDS -gt 1 ]]; then
  install_torchvision
  test_dynamo_shard 2
elif [[ "${SHARD_NUMBER}" == 1 && $NUM_TEST_SHARDS -gt 1 ]]; then
  test_without_numpy
  install_torchvision
  test_python_shard 1
  test_aten
elif [[ "${SHARD_NUMBER}" == 2 && $NUM_TEST_SHARDS -gt 1 ]]; then
  install_torchvision
  test_python_shard 2
  test_libtorch
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
  test_executorch
fi

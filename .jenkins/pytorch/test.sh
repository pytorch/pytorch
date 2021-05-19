#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

# Get fully qualified path using realpath
CUSTOM_TEST_ARTIFACT_BUILD_DIR=$(realpath "${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-${PWD}/../}")

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Testing pytorch"

export LANG=C.UTF-8

if [[ "$BUILD_ENVIRONMENT" == *-slow-* ]]; then
  export PYTORCH_TEST_WITH_SLOW=1
  export PYTORCH_TEST_SKIP_FAST=1
fi

if [[ "$BUILD_ENVIRONMENT" == *old-gradcheck* ]]; then
  export PYTORCH_TEST_WITH_SLOW_GRADCHECK=ON
fi

if [[ "$BUILD_ENVIRONMENT" == *coverage* ]]; then
  export PYTORCH_COLLECT_COVERAGE=1
  export COVERAGE_RCFILE="$PWD/.coveragerc" # coverage config file needed for plug-ins and settings to work
  pip install -e tools/coverage_plugins_package # allows coverage to run with JitPlugin for JIT coverage
fi

if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
  # Used so that only cuda specific versions of tests are generated
  # mainly used so that we're not spending extra cycles testing cpu
  # devices on expensive gpu machines
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
elif [[ "$BUILD_ENVIRONMENT" == *xla* ]]; then
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"
fi

if [[ "$BUILD_ENVIRONMENT" == *cuda11* ]]; then
  export BUILD_SPLIT_CUDA=ON
fi

if [[ "$BUILD_ENVIRONMENT" == *noarch* ]]; then
  export PYTORCH_TEST_SKIP_NOARCH=0
else
  export PYTORCH_TEST_SKIP_NOARCH=1
fi

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # Print GPU info
  rocminfo | grep -E 'Name:.*\sgfx|Marketing'
fi

# --user breaks ppc64le builds and these packages are already in ppc64le docker
if [[ "$BUILD_ENVIRONMENT" != *ppc64le* ]] && [[ "$BUILD_ENVIRONMENT" != *-bazel-* ]] ; then
  # JIT C++ extensions require ninja.
  pip_install --user ninja
  # ninja is installed in $HOME/.local/bin, e.g., /var/lib/jenkins/.local/bin for CI user jenkins
  # but this script should be runnable by any user, including root
  export PATH="$HOME/.local/bin:$PATH"
fi

# DANGER WILL ROBINSON.  The LD_PRELOAD here could cause you problems
# if you're not careful.  Check this if you made some changes and the
# ASAN test is not working
if [[ "$BUILD_ENVIRONMENT" == *asan* ]]; then
    # Suppress vptr violations arising from multiple copies of pybind11
    export ASAN_OPTIONS=detect_leaks=0:symbolize=1:strict_init_order=true:detect_odr_violation=0
    export UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PWD/ubsan.supp
    export PYTORCH_TEST_WITH_ASAN=1
    export PYTORCH_TEST_WITH_UBSAN=1
    # TODO: Figure out how to avoid hard-coding these paths
    export ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-5.0/bin/llvm-symbolizer
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
    export LD_PRELOAD=/usr/lib/llvm-5.0/lib/clang/5.0.0/lib/linux/libclang_rt.asan-x86_64.so
    # Increase stack size, because ASAN red zones use more stack
    ulimit -s 81920

    (cd test && python -c "import torch; print(torch.__version__, torch.version.git_version)")
    echo "The next three invocations are expected to crash; if they don't that means ASAN/UBSAN is misconfigured"
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_asan(3)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_ubsan(0)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_aten_asan(3)")
fi

if [[ "${BUILD_ENVIRONMENT}" == *-NO_AVX-* ]]; then
  export ATEN_CPU_CAPABILITY=default
elif [[ "${BUILD_ENVIRONMENT}" == *-NO_AVX2-* ]]; then
  export ATEN_CPU_CAPABILITY=avx
fi

# Try to pull value from CIRCLE_PULL_REQUEST first then GITHUB_HEAD_REF second
# CIRCLE_PULL_REQUEST comes from CircleCI
# GITHUB_HEAD_REF comes from Github Actions
IN_PULL_REQUEST=${CIRCLE_PULL_REQUEST:-}
if [ -n "$IN_PULL_REQUEST" ] && [[ "$BUILD_ENVIRONMENT" != *coverage* ]]; then
  DETERMINE_FROM=$(mktemp)
  file_diff_from_base "$DETERMINE_FROM"
fi

test_python_legacy_jit() {
  time python test/run_test.py --include test_jit_legacy test_jit_fuser_legacy --verbose --determine-from="$DETERMINE_FROM"
  assert_git_not_dirty
}

test_python_shard1() {
  time python test/run_test.py --exclude-jit-executor --shard 1 2 --verbose --determine-from="$DETERMINE_FROM"
  assert_git_not_dirty
}

test_python_shard2() {
  time python test/run_test.py --exclude-jit-executor --shard 2 2 --verbose --determine-from="$DETERMINE_FROM"
  assert_git_not_dirty
}

test_python() {
  time python test/run_test.py --exclude-jit-executor --verbose --determine-from="$DETERMINE_FROM"
  assert_git_not_dirty
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
    TORCH_LIB_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/torch/lib
    # NB: the ATen test binaries don't have RPATH set, so it's necessary to
    # put the dynamic libraries somewhere were the dynamic linker can find them.
    # This is a bit of a hack.
    if [[ "$BUILD_ENVIRONMENT" == *ppc64le* ]]; then
      SUDO=sudo
    fi

    ${SUDO} ln -sf "$TORCH_LIB_PATH"/libc10* build/bin
    ${SUDO} ln -sf "$TORCH_LIB_PATH"/libcaffe2* build/bin
    ${SUDO} ln -sf "$TORCH_LIB_PATH"/libmkldnn* build/bin
    ${SUDO} ln -sf "$TORCH_LIB_PATH"/libnccl* build/bin

    ls build/bin
    aten/tools/run_tests.sh build/bin
    assert_git_not_dirty
  fi
}

test_without_numpy() {
  pushd "$(dirname "${BASH_SOURCE[0]}")"
  python -c "import sys;sys.path.insert(0, 'fake_numpy');from unittest import TestCase;import torch;x=torch.randn(3,3);TestCase().assertRaises(RuntimeError, lambda: x.numpy())"
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

    # Start background download
    python tools/download_mnist.py --quiet -d test/cpp/api/mnist &

    # Make test_reports directory
    # NB: the ending test_libtorch must match the current function name for the current
    # test reporting process (in print_test_stats.py) to function as expected.
    TEST_REPORTS_DIR=test/test-reports/cpp-unittest/test_libtorch
    mkdir -p $TEST_REPORTS_DIR

    # Run JIT cpp tests
    python test/cpp/jit/tests_setup.py setup
    if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
      build/bin/test_jit  --gtest_output=xml:$TEST_REPORTS_DIR/test_jit.xml
    else
      build/bin/test_jit  --gtest_filter='-*CUDA' --gtest_output=xml:$TEST_REPORTS_DIR/test_jit.xml
    fi
    python test/cpp/jit/tests_setup.py shutdown
    # Wait for background download to finish
    wait
    OMP_NUM_THREADS=2 TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" build/bin/test_api --gtest_output=xml:$TEST_REPORTS_DIR/test_api.xml
    build/bin/test_tensorexpr --gtest_output=xml:$TEST_REPORTS_DIR/test_tensorexpr.xml
    assert_git_not_dirty
  fi
}

test_vulkan() {
  if [[ "$BUILD_ENVIRONMENT" == *vulkan-linux* ]]; then
    export VK_ICD_FILENAMES=/var/lib/jenkins/swiftshader/build/Linux/vk_swiftshader_icd.json
    # NB: the ending test_vulkan must match the current function name for the current
    # test reporting process (in print_test_stats.py) to function as expected.
    TEST_REPORTS_DIR=test/test-reports/cpp-vulkan/test_vulkan
    mkdir -p $TEST_REPORTS_DIR
    build/bin/vulkan_test --gtest_output=xml:$TEST_REPORTS_DIR/vulkan_test.xml
  fi
}

test_distributed() {
  if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
    echo "Testing distributed C++ tests"
    # NB: the ending test_distributed must match the current function name for the current
    # test reporting process (in print_test_stats.py) to function as expected.
    TEST_REPORTS_DIR=test/test-reports/cpp-distributed/test_distributed
    mkdir -p $TEST_REPORTS_DIR
    build/bin/FileStoreTest --gtest_output=xml:$TEST_REPORTS_DIR/FileStoreTest.xml
    build/bin/HashStoreTest --gtest_output=xml:$TEST_REPORTS_DIR/HashStoreTest.xml
    build/bin/TCPStoreTest --gtest_output=xml:$TEST_REPORTS_DIR/TCPStoreTest.xml

    MPIEXEC=$(command -v mpiexec)
    if [[ -n "$MPIEXEC" ]]; then
      MPICMD="${MPIEXEC} -np 2 build/bin/ProcessGroupMPITest"
      eval "$MPICMD"
    fi
    build/bin/ProcessGroupGlooTest --gtest_output=xml:$TEST_REPORTS_DIR/ProcessGroupGlooTest.xml
    build/bin/ProcessGroupNCCLTest --gtest_output=xml:$TEST_REPORTS_DIR/ProcessGroupNCCLTest.xml
    build/bin/ProcessGroupNCCLErrorsTest --gtest_output=xml:$TEST_REPORTS_DIR/ProcessGroupNCCLErrorsTest.xml
  fi
}

test_rpc() {
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]]; then
    echo "Testing RPC C++ tests"
    # NB: the ending test_rpc must match the current function name for the current
    # test reporting process (in print_test_stats.py) to function as expected.
    TEST_REPORTS_DIR=test/test-reports/cpp-rpc/test_rpc
    mkdir -p $TEST_REPORTS_DIR
    build/bin/test_cpp_rpc --gtest_output=xml:$TEST_REPORTS_DIR/test_cpp_rpc.xml
  fi
}

test_custom_backend() {
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *asan* ]] ; then
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
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *asan* ]] ; then
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
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *asan* ]] ; then
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

test_xla() {
  export XLA_USE_XRT=1 XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  # Issue #30717: randomize the port of XLA/gRPC workers is listening on to reduce flaky tests.
  XLA_PORT=$(shuf -i 40701-40999 -n 1)
  export XRT_WORKERS="localservice:0;grpc://localhost:$XLA_PORT"
  pushd xla
  echo "Running Python Tests"
  ./test/run_tests.sh

  # Disabled due to MNIST download issue.
  # See https://github.com/pytorch/pytorch/issues/53267
  # echo "Running MNIST Test"
  # python test/test_train_mnist.py --tidy

  echo "Running C++ Tests"
  pushd test/cpp
  CC=clang-9 CXX=clang++-9 ./run_tests.sh
  popd
  assert_git_not_dirty
}

# Do NOT run this test before any other tests, like test_python_shard1, etc.
# Because this function uninstalls the torch built from branch, and install
# nightly version.
test_backward_compatibility() {
  set -x
  pushd test/backward_compatibility
  python -m venv venv
  # shellcheck disable=SC1091
  . venv/bin/activate
  pip_install --pre torch -f https://download.pytorch.org/whl/test/cpu/torch_test.html
  pip show torch
  python dump_all_function_schemas.py --filename nightly_schemas.txt
  deactivate
  rm -r venv
  pip show torch
  python check_backward_compatibility.py --existing-schemas nightly_schemas.txt
  popd
  set +x
  assert_git_not_dirty
}

test_bazel() {
  set -e

  get_bazel

  tools/bazel test --test_timeout=480 --test_output=all --test_tag_filters=-gpu-required --test_filter=-*CUDA :all_tests
}

test_benchmarks() {
  if [[ "$BUILD_ENVIRONMENT" == *cuda* && "$BUILD_ENVIRONMENT" != *nogpu* ]]; then
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
  time python test/run_test.py --include test_cpp_extensions_aot_ninja --verbose --determine-from="$DETERMINE_FROM"
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

test_torch_deploy() {
  python torch/csrc/deploy/example/generate_examples.py
  build/bin/test_deploy
  assert_git_not_dirty
}

if ! [[ "${BUILD_ENVIRONMENT}" == *libtorch* || "${BUILD_ENVIRONMENT}" == *-bazel-* ]]; then
  (cd test && python -c "import torch; print(torch.__config__.show())")
  (cd test && python -c "import torch; print(torch.__config__.parallel_info())")
fi

if [[ "${BUILD_ENVIRONMENT}" == *backward* ]]; then
  test_backward_compatibility
  # Do NOT add tests after bc check tests, see its comment.
elif [[ "${BUILD_ENVIRONMENT}" == *xla* || "${JOB_BASE_NAME}" == *xla* ]]; then
  install_torchvision
  test_xla
elif [[ "${BUILD_ENVIRONMENT}" == *jit_legacy-test || "${JOB_BASE_NAME}" == *jit_legacy-test ]]; then
  test_python_legacy_jit
elif [[ "${BUILD_ENVIRONMENT}" == *libtorch* ]]; then
  # TODO: run some C++ tests
  echo "no-op at the moment"
elif [[ "${BUILD_ENVIRONMENT}" == *-test1 || "${JOB_BASE_NAME}" == *-test1 ]]; then
  if [[ "${BUILD_ENVIRONMENT}" == pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7-test1 ]]; then
    test_torch_deploy
  fi
  test_without_numpy
  install_torchvision
  test_python_shard1
  test_aten
elif [[ "${BUILD_ENVIRONMENT}" == *-test2 || "${JOB_BASE_NAME}" == *-test2 ]]; then
  install_torchvision
  test_python_shard2
  test_libtorch
  test_custom_script_ops
  test_custom_backend
  test_torch_function_benchmark
elif [[ "${BUILD_ENVIRONMENT}" == *vulkan-linux* ]]; then
  test_vulkan
elif [[ "${BUILD_ENVIRONMENT}" == *-bazel-* ]]; then
  test_bazel
else
  install_torchvision
  install_monkeytype
  test_python
  test_aten
  test_vec256
  test_libtorch
  test_custom_script_ops
  test_custom_backend
  test_torch_function_benchmark
  test_distributed
  test_benchmarks
  test_rpc
  if [[ "${BUILD_ENVIRONMENT}" == pytorch-linux-xenial-py3.6-gcc7-test || "${BUILD_ENVIRONMENT}" == pytorch-linux-xenial-py3.6-gcc5.4-test ]]; then
    test_python_gloo_with_tls
  fi
fi

if [[ "$BUILD_ENVIRONMENT" == *coverage* ]]; then
  pushd test
  echo "Generating XML coverage report"
  time python -mcoverage xml
  popd
  pushd build
  echo "Generating lcov coverage report for C++ sources"
  time lcov --capture --directory . --output-file coverage.info
  popd
fi

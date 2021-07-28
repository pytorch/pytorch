#!/bin/bash

# shellcheck source=./test-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/test-common.sh"

if [[ "${BUILD_ENVIRONMENT}" == *backward* ]]; then
  test_backward_compatibility
  # Do NOT add tests after bc check tests, see its comment.
elif [[ "${BUILD_ENVIRONMENT}" == *xla* || "${JOB_BASE_NAME}" == *xla* ]]; then
  install_torchvision
  test_xla
elif [[ "${BUILD_ENVIRONMENT}" == *jit_legacy-test || "${JOB_BASE_NAME}" == *jit_legacy-test || $TEST_CONFIG == 'jit_legacy' ]]; then
  test_python_legacy_jit
elif [[ "${BUILD_ENVIRONMENT}" == *libtorch* ]]; then
  # TODO: run some C++ tests
  echo "no-op at the moment"
elif [[ "${BUILD_ENVIRONMENT}" == *-test1 || "${JOB_BASE_NAME}" == *-test1 || "${SHARD_NUMBER}" == 1 ]]; then
  if [[ "${BUILD_ENVIRONMENT}" == pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7-test1 ]]; then
    test_torch_deploy
  fi
  test_without_numpy
  install_torchvision
  test_python_shard1
  test_aten
elif [[ "${BUILD_ENVIRONMENT}" == *-test2 || "${JOB_BASE_NAME}" == *-test2 || "${SHARD_NUMBER}" == 2 ]]; then
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

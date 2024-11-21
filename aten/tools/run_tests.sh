#!/bin/bash
set -x
set -e

VALGRIND_SUP="${PWD}/`dirname $0`/valgrind.sup"
export CPP_TESTS_DIR=$1

VALGRIND=${VALGRIND:=ON}
python test/run_test.py --cpp --verbose -i \
  cpp/basic \
  cpp/atest \
  cpp/scalar_test \
  cpp/broadcast_test \
  cpp/wrapdim_test \
  cpp/apply_utils_test \
  cpp/dlconvertor_test \
  cpp/native_test \
  cpp/scalar_tensor_test \
  cpp/undefined_tensor_test \
  cpp/extension_backend_test \
  cpp/lazy_tensor_test \
  cpp/tensor_iterator_test \
  cpp/Dimname_test \
  cpp/Dict_test \
  cpp/NamedTensor_test \
  cpp/cpu_generator_test \
  cpp/legacy_vmap_test \
  cpp/operators_test

run_if_exists() {
  local test_name="$1"
  if [[ -x "${CPP_TESTS_DIR}/${test_name}" ]]; then
    python test/run_test.py --cpp --verbose -i "cpp/${test_name}"
  else
    echo "Warning: $test_name does not exist."
  fi
}

run_if_exists tensor_interop_test
run_if_exists cudnn_test
run_if_exists cuda_generator_test
run_if_exists apply_test
run_if_exists stream_test
run_if_exists cuda_half_test
run_if_exists cuda_vectorized_test
run_if_exists cuda_distributions_test
run_if_exists cuda_optional_test
run_if_exists cuda_tensor_interop_test
run_if_exists cuda_complex_test
run_if_exists cuda_complex_math_test
run_if_exists cuda_cub_test
run_if_exists cuda_atomic_ops_test

if [ "$VALGRIND" == "ON" ]; then
  # NB: As these tests are invoked by valgrind, let's leave them for now as it's
  # unclear if valgrind -> python -> gtest would work
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 "${CPP_TESTS_DIR}/basic" --gtest_filter='-*CUDA'
  if [[ -x ${CPP_TESTS_DIR}/tensor_interop_test ]]; then
    valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 "${CPP_TESTS_DIR}/tensor_interop_test"
  fi
fi

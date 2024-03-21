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

if [[ -x ./tensor_interop_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/tensor_interop_test
fi
if [[ -x ./cudnn_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cudnn_test
fi
if [[ -x ./cuda_generator_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_generator_test
fi
if [[ -x ./apply_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/apply_test
fi
if [[ -x ./stream_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/stream_test
fi
if [[ -x ./cuda_half_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_half_test
fi
if [[ -x ./cuda_vectorized_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_vectorized_test
fi
if [[ -x ./cuda_distributions_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_distributions_test
fi
if [[ -x ./cuda_optional_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_optional_test
fi
if [[ -x ./cuda_tensor_interop_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_tensor_interop_test
fi
if [[ -x ./cuda_complex_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_complex_test
fi
if [[ -x ./cuda_complex_math_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_complex_math_test
fi
if [[ -x ./cuda_cub_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_cub_test
fi
if [[ -x ./cuda_atomic_ops_test ]]; then
  python test/run_test.py --cpp --verbose -i cpp/cuda_atomic_ops_test
fi

if [ "$VALGRIND" == "ON" ]; then
  # NB: As these tests are invoked by valgrind, let's leave them for now as it's
  # unclear if valgrind -> python -> gtest would work
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 "${CPP_TESTS_DIR}/basic" --gtest_filter='-*CUDA'
  if [[ -x ./tensor_interop_test ]]; then
    valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 "${CPP_TESTS_DIR}/tensor_interop_test"
  fi
fi

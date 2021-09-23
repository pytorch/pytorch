#!/bin/bash
set -x
set -e

VALGRIND_SUP="${PWD}/`dirname $0`/valgrind.sup"
pushd $1

VALGRIND=${VALGRIND:=ON}
python ../../test/test_gtest.py GTest.test_basic
python ../../test/test_gtest.py GTest.test_atest
python ../../test/test_gtest.py GTest.test_scalar_test
python ../../test/test_gtest.py GTest.test_broadcast_test
python ../../test/test_gtest.py GTest.test_wrapdim_test
python ../../test/test_gtest.py GTest.test_apply_utils_test
python ../../test/test_gtest.py GTest.test_dlconvertor_test
python ../../test/test_gtest.py GTest.test_native_test
python ../../test/test_gtest.py GTest.test_scalar_tensor_test
python ../../test/test_gtest.py GTest.test_tensor_interop_test
python ../../test/test_gtest.py GTest.test_undefined_tensor_test
python ../../test/test_gtest.py GTest.test_extension_backend_test
python ../../test/test_gtest.py GTest.test_lazy_tensor_test
python ../../test/test_gtest.py GTest.test_tensor_iterator_test
python ../../test/test_gtest.py GTest.test_Dimname_test
python ../../test/test_gtest.py GTest.test_Dict_test
python ../../test/test_gtest.py GTest.test_NamedTensor_test
python ../../test/test_gtest.py GTest.test_cpu_generator_test
python ../../test/test_gtest.py GTest.test_vmap_test
python ../../test/test_gtest.py GTest.test_operators_test

if [[ -x ./cudnn_test ]]; then
  ./cudnn_test
fi
if [[ -x ./cuda_generator_test ]]; then
  ./cuda_generator_test
fi
if [[ -x ./apply_test ]]; then
  ./apply_test
fi
if [[ -x ./stream_test ]]; then
  ./stream_test
fi
if [[ -x ./cuda_half_test ]]; then
  ./cuda_half_test
fi
if [[ -x ./cuda_vectorized_test ]]; then
  ./cuda_vectorized_test
fi
if [[ -x ./cuda_distributions_test ]]; then
  ./cuda_distributions_test
fi
if [[ -x ./cuda_optional_test ]]; then
  ./cuda_optional_test
fi
if [[ -x ./cuda_tensor_interop_test ]]; then
  ./cuda_tensor_interop_test
fi
if [[ -x ./cuda_complex_test ]]; then
  ./cuda_complex_test
fi
if [[ -x ./cuda_complex_math_test ]]; then
  ./cuda_complex_math_test
fi
if [[ -x ./cuda_cub_test ]]; then
  ./cuda_cub_test
fi
if [ "$VALGRIND" == "ON" ]
then
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 ./basic --gtest_filter='-*CUDA'
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 ./tensor_interop_test
fi

popd

#!/bin/bash
set -x
set -e

VALGRIND_SUP="${PWD}/`dirname $0`/valgrind.sup"
pushd $1

VALGRIND=${VALGRIND:=ON}
./basic
./atest
./scalar_test
./broadcast_test
./wrapdim_test
./apply_utils_test
./dlconvertor_test
./native_test
./scalar_tensor_test
./tensor_interop_test
./undefined_tensor_test
./extension_backend_test
./xla_tensor_test
./tensor_iterator_test
./Dimname_test
./Dict_test
./NamedTensor_test
./cpu_generator_test
./vmap_test
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
if [ "$VALGRIND" == "ON" ]
then
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 ./basic --gtest_filter='-*CUDA'
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 ./tensor_interop_test
fi

popd

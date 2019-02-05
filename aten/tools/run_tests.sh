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
if [[ -x ./cudnn_test ]]; then
  ./cudnn_test
fi
if [[ -x ./cuda_rng_test ]]; then
  ./cuda_rng_test
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
if [[ -x ./cuda_optional_test ]]; then
  ./cuda_optional_test
fi
if [[ -x ./cuda_tensor_interop_test ]]; then
  ./cuda_tensor_interop_test
fi
if [ "$VALGRIND" == "ON" ]
then
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 ./basic "[cpu]"
  valgrind --suppressions="$VALGRIND_SUP" --error-exitcode=1 ./tensor_interop_test
fi

popd

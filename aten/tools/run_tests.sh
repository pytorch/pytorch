#!/bin/bash
set -x
set -e

pushd $BUILD_ROOT

VALGRIND=${VALGRIND:=ON}
BUILD_ROOT=$1
./basic
./atest
./scalar_test
./broadcast_test
./wrapdim_test
./apply_utils_test
./dlconvertor_test
./native_test
./scalar_tensor_test
./undefined_tensor_test
if [[ -x ./cudnn_test ]]; then
  ./cudnn_test
fi
if [[ -x ./cuda_rng_test ]]; then
  ./cuda_rng_test
fi
if [[ -x ./apply_test ]]; then
  ./apply_test
fi
if [ "$VALGRIND" == "ON" ]
then
  valgrind --suppressions=`dirname $0`/valgrind.sup --error-exitcode=1 ./basic "[cpu]"
fi

popd

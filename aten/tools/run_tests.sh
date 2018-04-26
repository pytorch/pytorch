#!/bin/bash
set -x
set -e

VALGRIND=${VALGRIND:=ON}
BUILD_ROOT=$1
$BUILD_ROOT/src/ATen/test/basic
$BUILD_ROOT/src/ATen/test/atest
$BUILD_ROOT/src/ATen/test/scalar_test
$BUILD_ROOT/src/ATen/test/broadcast_test
$BUILD_ROOT/src/ATen/test/wrapdim_test
$BUILD_ROOT/src/ATen/test/dlconvertor_test
$BUILD_ROOT/src/ATen/test/native_test
$BUILD_ROOT/src/ATen/test/scalar_tensor_test
$BUILD_ROOT/src/ATen/test/undefined_tensor_test
if [[ -x $BUILD_ROOT/src/ATen/test/cudnn_test ]]; then
  $BUILD_ROOT/src/ATen/test/cudnn_test
fi
if [[ -x $BUILD_ROOT/src/ATen/test/cuda_rng_test ]]; then
  $BUILD_ROOT/src/ATen/test/cuda_rng_test
fi
if [ "$VALGRIND" == "ON" ]
then
  valgrind --suppressions=`dirname $0`/valgrind.sup --error-exitcode=1 $BUILD_ROOT/src/ATen/test/basic "[cpu]"
fi

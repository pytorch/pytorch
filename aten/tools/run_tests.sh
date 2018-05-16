#!/bin/bash
set -x
set -e

VALGRIND=${VALGRIND:=ON}
BUILD_ROOT=$1
$BUILD_ROOT/basic
$BUILD_ROOT/atest
$BUILD_ROOT/scalar_test
$BUILD_ROOT/broadcast_test
$BUILD_ROOT/wrapdim_test
$BUILD_ROOT/apply_utils_test
$BUILD_ROOT/dlconvertor_test
$BUILD_ROOT/native_test
$BUILD_ROOT/scalar_tensor_test
$BUILD_ROOT/undefined_tensor_test
if [[ -x $BUILD_ROOT/cudnn_test ]]; then
  $BUILD_ROOT/cudnn_test
fi
if [[ -x $BUILD_ROOT/cuda_rng_test ]]; then
  $BUILD_ROOT/cuda_rng_test
fi
if [[ -x $BUILD_ROOT/apply_test ]]; then
  $BUILD_ROOT/apply_test
fi
if [ "$VALGRIND" == "ON" ]
then
  valgrind --suppressions=`dirname $0`/valgrind.sup --error-exitcode=1 $BUILD_ROOT/basic "[cpu]"
fi

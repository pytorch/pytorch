#!/bin/bash
set -x
set -e
BUILD_ROOT=$1
$BUILD_ROOT/src/ATen/test/basic
$BUILD_ROOT/src/ATen/test/atest
$BUILD_ROOT/src/ATen/test/scalar_test
$BUILD_ROOT/src/ATen/test/broadcast_test
valgrind --suppressions=`dirname $0`/valgrind.sup --error-exitcode=1 $BUILD_ROOT/src/ATen/test/basic -n

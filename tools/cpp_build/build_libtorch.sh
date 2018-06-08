#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pushd $SCRIPTPATH
source ./build_common.sh

echo "Building Torch"

mkdir -p $LIBTORCH_BUILDPATH
pushd $LIBTORCH_BUILDPATH

cmake -DUSE_CUDA:BOOL=$USE_CUDA \
      -DNO_API:BOOL=${NO_API:0} \
      -DCAFFE2_PATH=$PYTORCHPATH/ \
      -DCAFFE2_BUILD_PATH=$CAFFE2_BUILDPATH \
      -DNANOPB_BUILD_PATH=$NANOPB_BUILDPATH \
      -DINSTALL_PREFIX=$INSTALL_PREFIX \
      -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX:STRING=$INSTALL_PREFIX \
      -DCMAKE_INSTALL_MESSAGE=NEVER \
      -DVERBOSE:BOOL=${VERBOSE:-0} \
      -G "$GENERATE" \
      $SCRIPTPATH/libtorch
$MAKE -j "$JOBS"

popd
popd

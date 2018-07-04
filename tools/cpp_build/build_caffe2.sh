#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pushd $SCRIPTPATH
source ./build_common.sh

echo "Building Caffe2"

mkdir -p $CAFFE2_BUILDPATH
pushd $CAFFE2_BUILDPATH

cmake -DUSE_CUDA=$USE_CUDA \
      -DBUILD_CAFFE2=OFF \
      -DBUILD_ATEN=ON \
      -DBUILD_PYTHON=OFF \
      -DBUILD_BINARY=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DONNX_NAMESPACE=$ONNX_NAMESPACE \
      -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX:STRING=$INSTALL_PREFIX \
      -DCMAKE_INSTALL_MESSAGE=NEVER \
      -G "$GENERATE" \
      $PYTORCHPATH/
$MAKE -j "$JOBS" install

popd
popd

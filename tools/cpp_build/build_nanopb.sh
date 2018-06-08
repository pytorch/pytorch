#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pushd $SCRIPTPATH
source ./build_common.sh

echo "Building nanopb"

mkdir -p $NANOPB_BUILDPATH
pushd $NANOPB_BUILDPATH

cmake -DCMAKE_BUILD_TYPE:STRING=Release \
      -DCMAKE_INSTALL_PREFIX:STRING=$INSTALL_PREFIX \
      -DCMAKE_INSTALL_MESSAGE=NEVER \
      -Dnanopb_BUILD_GENERATOR:BOOL=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
      -G "$GENERATE" \
      $PYTORCHPATH/third_party/nanopb
$MAKE -j "$JOBS"

popd
popd

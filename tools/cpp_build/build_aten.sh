#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pushd $SCRIPTPATH
source ./build_common.sh

echo "Building ATen"

mkdir -p $ATEN_BUILDPATH
pushd $ATEN_BUILDPATH

cmake -DNO_CUDA:BOOL=$NO_CUDA \
      -DAT_LINK_STYLE:STRING=SHARED \
      -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX:STRING=$INSTALL_PREFIX \
      -DCMAKE_INSTALL_MESSAGE=NEVER \
      -G "$GENERATE" \
      $PYTORCHPATH/aten
$MAKE -j "$JOBS"

popd
popd

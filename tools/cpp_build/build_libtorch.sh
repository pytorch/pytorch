#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pushd $SCRIPTPATH
source ./build_common.sh

echo "Generating code"

pushd "../.."
cp aten/src/ATen/common_with_cwrap.py tools/shared/cwrap_common.py
python tools/setup_helpers/generate_code.py \
  --declarations-path "$ATEN_BUILDPATH/src/ATen/ATen/Declarations.yaml" \
  --nn-path "aten/src/"
popd

echo "Building Torch"

mkdir -p $LIBTORCH_BUILDPATH
pushd $LIBTORCH_BUILDPATH

cmake -DNO_CUDA:BOOL=$NO_CUDA \
      -DATEN_PATH=$PYTORCHPATH/aten/ \
      -DATEN_BUILD_PATH=$ATEN_BUILDPATH \
      -DNANOPB_BUILD_PATH=$NANOPB_BUILDPATH \
      -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX:STRING=$INSTALL_PREFIX \
      -DCMAKE_INSTALL_MESSAGE=NEVER \
      -DVERBOSE=${VERBOSE:-0} \
      -G "$GENERATE" \
      $SCRIPTPATH/libtorch
$MAKE -j "$JOBS"

popd
popd

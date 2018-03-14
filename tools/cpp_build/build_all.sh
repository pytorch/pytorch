#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

INSTALL_PREFIX="$1"

if [[ -z $INSTALL_PREFIX ]]; then
  INSTALL_PREFIX=$SCRIPTPATH/build/Install
fi

cd $SCRIPTPATH

PYTORCHPATH="$SCRIPTPATH/../../"

if [ ! -d "$PYTORCHPATH/torch/csrc/autograd/generated" ]; then
  echo "Generated files are not present.\nRun the generators through `python setup.py build` or copy the generated files over."
  exit 1
fi

if [ -x "$(command -v nvcc)" ]; then
  NO_CUDA=OFF
else
  NO_CUDA=ON
fi

ATEN_BUILDPATH="$SCRIPTPATH/build/aten-build"
LIBTORCH_BUILDPATH="$SCRIPTPATH/build/libtorch-build"

mkdir -p $ATEN_BUILDPATH && cd $ATEN_BUILDPATH
cmake -DNO_CUDA:BOOL=$NO_CUDA -DAT_LINK_STYLE:STRING=SHARED -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX -DCMAKE_BUILD_TYPE:STRING=Release $PYTORCHPATH/aten
make -j4 && make install

mkdir -p $LIBTORCH_BUILDPATH && cd $LIBTORCH_BUILDPATH
cmake -DNO_CUDA:BOOL=$NO_CUDA -DCMAKE_BUILD_TYPE:STRING=RELEASE -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX $SCRIPTPATH/libtorch
make -j4
make install

cd $SCRIPTPATH

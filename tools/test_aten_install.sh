#!/bin/sh
set -xe
mkdir aten_build aten_install
cd aten_build
# Tests only buildable via root right now, so disable here
cmake ../aten -DNO_CUDA=1 -DATEN_NO_TEST=ON -DCMAKE_INSTALL_PREFIX=../aten_install
NUM_JOBS="$(getconf _NPROCESSORS_ONLN)"
make -j"$NUM_JOBS" install
cd ..
aten/tools/test_install.sh $(pwd)/aten_install $(pwd)/aten

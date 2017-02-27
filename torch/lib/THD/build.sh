#!/bin/bash

if [[ $(basename $(pwd)) != "THD" ]]; then
  echo "The build script has to be executed from the root directory of THD!"
  exit 2
fi

cd ..
lib_dir="$(pwd)"
cd THD

mkdir -p build
cd build

LD_POSTFIX=".so.1"
if [[ $(uname) == 'Darwin' ]]; then
    LD_POSTFIX=".1.dylib"
fi

cmake .. -DCMAKE_CXX_FLAGS=" -I${lib_dir}/tmp_install/include -pthread \
                             -I${lib_dir}/THPP " \
         -DCMAKE_SHARED_LINKER_FLAGS="-L${lib_dir}/tmp_install/lib " \
         -DCMAKE_EXE_LINKER_FLAGS="-L${lib_dir}/tmp_install/lib -pthread " \
         -DTH_LIBRARIES="${lib_dir}/libTH$LD_POSTFIX" \
         -DTHPP_LIBRARIES="${lib_dir}/libTHPP$LD_POSTFIX" \
         -DTorch_FOUND="1"
make

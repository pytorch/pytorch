#!/bin/bash

export CXX=$COMPILER

# just to make sure
git submodule update --init --recursive

mkdir build
cd build

if [[ $BUILD_TARGET == 'android' ]]; then
#***************#
# Android build #
#***************#
  sh ../scripts/build_android.sh
elif [[ $BUILD_TARGET == 'ios' ]]; then
#***************#
# iOS build     #
#***************#
  sh ../scripts/build_ios.sh
elif [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#************#
# OS X build #
#************#
  cmake .. \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DUSE_OPENCV=OFF \
      -DUSE_NNPACK=ON \
      && make
else
#*************#
# Linux build #
#*************#
  if [[ $BLAS == 'MKL' ]]; then
    cmake .. \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DBLAS=MKL \
        -DUSE_NNPACK=ON \
        && make
  else
    cmake .. \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DUSE_NNPACK=ON && make
  fi
fi

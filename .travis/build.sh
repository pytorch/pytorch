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
  # Note: we will only build arm64 in the travis case for faster compilation.
  # You might want to build a fat binary containng armv7, armv7s and arm64.
  # This can be done by simply not passing in the CMAKE_OSX_ARCHITECTURES flag.
  sh ../scripts/build_ios.sh -DCMAKE_OSX_ARCHITECTURES=arm64
elif [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#************#
# OS X build #
#************#
  cmake .. \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DUSE_OPENCV=OFF \
  && make
else
#*************#
# Linux build #
#*************#
  if [[ $BLAS == 'MKL' ]]; then
    cmake .. \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DBLAS=MKL \
        -DUSE_CUDA=OFF \
    && make
  else
    cmake .. \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
    && make
  fi
fi

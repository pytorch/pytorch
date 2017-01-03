#!/bin/bash

mkdir build
cd build

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#************#
# OS X build #
#************#
  cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON -DBLAS=OpenBLAS -DUSE_OPENCV=off && make
else
#*************#
# Linux build #
#*************#
  cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON && make
fi

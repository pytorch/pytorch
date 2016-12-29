#!/bin/bash

mkdir build
cd build

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#************#
# OS X build #
#************#
  export CXX=$COMPILER && cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON -DBLAS=OpenBLAS -DUSE_OPENCV=off && make CXX=$COMPILER
else
#*************#
# Linux build #
#*************#
  export CXX=$COMPILER && cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON && make CXX=$COMPILER
fi

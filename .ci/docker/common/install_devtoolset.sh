#!/bin/bash

set -ex

[ -n "$DEVTOOLSET_VERSION" ]

yum install -y \
  gcc-toolset-$DEVTOOLSET_VERSION-gcc \
  gcc-toolset-$DEVTOOLSET_VERSION-gcc-c++ \
  gcc-toolset-$DEVTOOLSET_VERSION-gcc-gfortran

echo "source scl_source enable gcc-toolset-$DEVTOOLSET_VERSION" > "/etc/profile.d/gcc-toolset-$DEVTOOLSET_VERSION.sh"

#!/bin/bash

set -ex

# MKL
MKL_VERSION=2022.2.1
MKL_BUILD=16993
mkdir -p /opt/intel/lib
pushd /tmp
curl -fsSL https://anaconda.org/intel/mkl-static/${MKL_VERSION}/download/linux-64/mkl-static-${MKL_VERSION}-intel_${MKL_BUILD}.tar.bz2 | tar xjv
mv lib/* /opt/intel/lib/
curl -fsSL https://anaconda.org/intel/mkl-include/${MKL_VERSION}/download/linux-64/mkl-include-${MKL_VERSION}-intel_${MKL_BUILD}.tar.bz2 | tar xjv
mv include /opt/intel/

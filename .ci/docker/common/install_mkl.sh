#!/bin/bash
set -ex

# MKL
MKL_VERSION=2024.2.0

MKLROOT=/opt/intel
mkdir -p ${MKLROOT}
pushd /tmp

python3 -mpip install wheel
python3 -mpip download -d . mkl-static==${MKL_VERSION}
python3 -m wheel unpack mkl_static-${MKL_VERSION}-py2.py3-none-manylinux1_x86_64.whl
python3 -m wheel unpack mkl_include-${MKL_VERSION}-py2.py3-none-manylinux1_x86_64.whl
mv mkl_static-${MKL_VERSION}/mkl_static-${MKL_VERSION}.data/data/lib ${MKLROOT}
mv mkl_include-${MKL_VERSION}/mkl_include-${MKL_VERSION}.data/data/include ${MKLROOT}

#!/bin/bash
set -ex

# MKL
MKL_VERSION=2026.2.0

MKLROOT=/opt/intel
mkdir -p ${MKLROOT}
pushd /tmp

# The 2026.x wheels are tagged manylinux_2_28_x86_64, which pip < 20.3 does not
# understand (it only sees MKL up to 2024.2.2). The AlmaLinux 8 images ship pip 9,
# so upgrade it first. Only these two wheels are needed, not their dependencies.
python3 -mpip install --upgrade pip wheel
python3 -mpip download -d . --no-deps mkl-static==${MKL_VERSION} mkl-include==${MKL_VERSION}
python3 -m wheel unpack mkl_static-${MKL_VERSION}-py2.py3-none-manylinux_2_28_x86_64.whl
python3 -m wheel unpack mkl_include-${MKL_VERSION}-py2.py3-none-manylinux_2_28_x86_64.whl
mv mkl_static-${MKL_VERSION}/mkl_static-${MKL_VERSION}.data/data/lib ${MKLROOT}
mv mkl_include-${MKL_VERSION}/mkl_include-${MKL_VERSION}.data/data/include ${MKLROOT}

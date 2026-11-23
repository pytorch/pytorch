#!/bin/bash
set -ex

# MKL
MKL_VERSION=2026.2.0

MKLROOT=/opt/intel
mkdir -p ${MKLROOT}
pushd /tmp

# The 2026.x wheels are tagged manylinux_2_28_x86_64, which pip < 20.3 does not
# understand (it only sees MKL up to 2024.2.2). The system Python 3.6 of the
# manywheel `intel` stage has pip 9, so make sure pip is new enough; an
# environment that already has pip >= 20.3 (conda, the ROCm stage, a wheel build
# env) is left alone. Only these two wheels are needed, not their dependencies.
# Keep MKL_VERSION in sync with requirements-ci.txt, build_pytorch.bat and
# .ci/wheel/windows/build_install_deps.py.
python3 -mpip install "pip>=20.3" wheel
python3 -mpip download -d . --no-deps mkl-static==${MKL_VERSION} mkl-include==${MKL_VERSION}
python3 -m wheel unpack mkl_static-${MKL_VERSION}-*.whl
python3 -m wheel unpack mkl_include-${MKL_VERSION}-*.whl
mv mkl_static-${MKL_VERSION}/mkl_static-${MKL_VERSION}.data/data/lib ${MKLROOT}
mv mkl_include-${MKL_VERSION}/mkl_include-${MKL_VERSION}.data/data/include ${MKLROOT}

#! /bin/bash

export USE_SYSTEM_PYBIND11=ON
export USE_XNNPACK=OFF
export USE_DISTRIBUTED=OFF
export USE_KINETO=OFF
export CMAKE_CXX_FLAGS="-I/usr/include/python3.12/"
export BUILD_TEST=OFF
export USE_CUDA=OFF
export USE_XPU=OFF
export CMAKE_POLICY_VERSION_MINIMUM=3.5 # not sure why :/

export USE_MPS=ON
export USE_MPS_REMOTING_FRONTEND=ON
export MAX_JOBS=6

export CMAKE_INSTALL_COMPONENT=true

sed 's/# Set the install prefix/set(CMAKE_INSTALL_COMPONENT true)/' -i build/third_party/protobuf/cmake/cmake_install.cmake
python setup.py develop && python -c 'import torch; print("Python working!")'

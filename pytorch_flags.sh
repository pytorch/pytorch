#!/bin/sh

export DEBUG=1
export REL_WITH_DEB_INFO="-g"
export USE_CUDA=1
export USE_CUDNN=1
export BUILD_BINARY=1
export USE_MKLDNN=0
export CXXFLAGS="-g -O0"
export CFLAGS="-g -O0"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

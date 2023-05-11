#!/bin/bash
export CMAKE_GENERATOR=Ninja
export TORCH_PACKAGE_NAME=torch
export USE_PYTORCH_METAL_EXPORT=1
export CMAKE_BUILD_TYPE=Release
export USE_COREML_DELEGATE=1
export MAX_JOBS=10
export DEBUG=0
export WERROR=0
LLVM='/opt/homebrew/opt/llvm@15'
export CFLAGS="-Wno-unused-command-line-argument -Wno-cast-function-type -Werror=incompatible-function-pointer-types  -Wno-incompatible-function-pointer-types -Wno-cast-function-type -march=native -mfpu=native -mcpu=native -Ofast -I${LLVM}/include -I/opt/homebrew/opt/libomp/include"
export LDFLAGS="-L${LLVM}/lib -L/opt/homebrew/opt/libomp/lib -L${LLVM}/lib/c++ -Wl,-rpath,${LLVM}/lib/c++"
export USE_CUDA=0
export CC="${LLVM}/bin/clang"
export CXX="${LLVM}/bin/clang++"
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CMAKE_INCLUDE_PATH="${LLVM}/include /opt/homebrew/opt/libomp/include"
export LIB="${LLVM}/lib /opt/homebrew/opt/libomp/lib"
export BUILD_CAFFE2=0
export MACOSX_DEPLOYMENT_TARGET=13.3
export BUILD_CAFFE2_OPS=0
export USE_DISTRIBUTED=0
export USE_ONNX=0
export TORCH_SHOW_CPP_STACKTRACES=0

# python setup.py install
python setup.py develop

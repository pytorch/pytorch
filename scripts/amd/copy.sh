#!/bin/bash
set -ex

BUILD_DIR=/tmp/pytorch

cp_to_build_dir() {
    local CUR_FILE=$1
    chmod -R 777 $CUR_FILE
    cp -rf --parents $CUR_FILE $BUILD_DIR
}

# "c10/macros/Macros.h"
# "aten/src/ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh"
# "aten/src/ATen/native/cuda/block_reduce.cuh"
# "aten/src/ATen/native/cuda/RangeFactories.cu"
# "aten/src/ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh"
# "aten/src/ATen/native/cuda/MemoryAccess.cuh"
# "aten/src/ATen/native/cuda/ROCmLoops.cuh"
# "aten/src/ATen/native/cuda/Loops.cuh"
# "aten/src/ATen/native/hip/Loops.cuh"
# "aten/src/ATen/native/cuda/ScatterGatherKernel.cu"
# "aten/src/ATen/native/cuda/LinearAlgebra.cu"
# "test/test_hang.py"
# "test/test_cuda.py"

FILE_LIST=(
    "test/test_nn.py"
)
for FILE in "${FILE_LIST[@]}"; do
    cp_to_build_dir $FILE
done
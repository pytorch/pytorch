#include "caffe2/operators/batch_matmul_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {
namespace {

REGISTER_CUDA_OPERATOR(BatchMatMul, BatchMatMulOp<float, CUDAContext>);
}
}

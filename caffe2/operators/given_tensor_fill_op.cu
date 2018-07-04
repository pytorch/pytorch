#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/given_tensor_fill_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    GivenTensorDoubleFill,
    GivenTensorFillOp<double, CUDAContext>);
REGISTER_CUDA_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    GivenTensorInt64Fill,
    GivenTensorFillOp<int64_t, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    GivenTensorBoolFill,
    GivenTensorFillOp<bool, CUDAContext>);
}

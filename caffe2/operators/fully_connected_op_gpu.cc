#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(FC, FullyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(FCGradient,
                       FullyConnectedGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2

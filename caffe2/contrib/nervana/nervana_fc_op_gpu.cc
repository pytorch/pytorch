#include "nervana.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    FC, NERVANA, FullyConnectedOp<float, CUDAContext, NervanaEngine>);
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    FCGradient, NERVANA,
    FullyConnectedGradientOp<float, CUDAContext, NervanaEngine>);
}  // namespace caffe2

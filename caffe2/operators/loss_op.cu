#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/loss_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(AveragedLoss, AveragedLoss<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    AveragedLossGradient,
    AveragedLossGradient<float, CUDAContext>);
}  // namespace
}  // namespace caffe2

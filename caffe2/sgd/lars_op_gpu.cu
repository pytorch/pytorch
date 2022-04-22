#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/lars_op.h"

namespace caffe2 {
__global__ void ComputeLearningRateKernel(
    const float* wd,
    const float* trust,
    const float* lr_max,
    float offset,
    float lr_min,
    float* X_norm,
    float* dX_norm,
    float* lr_rescaled) {
  float val = 1.0;

  if (*X_norm > 0) {
    val = (*trust) / (*dX_norm / *X_norm + (*wd) + offset);
  }
  *lr_rescaled = fmaxf(fminf(val, *lr_max), lr_min);
}

template <>
void LarsOp<float, CUDAContext>::ComputeLearningRate(
    const float* wd,
    const float* trust,
    const float* lr_max,
    float offset,
    float lr_min,
    float* X_norm,
    float* dX_norm,
    float* lr_rescaled) {
  ComputeLearningRateKernel<<<1, 1, 0, context_.cuda_stream()>>>(
      wd, trust, lr_max, offset, lr_min, X_norm, dX_norm, lr_rescaled);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

REGISTER_CUDA_OPERATOR(Lars, LarsOp<float, CUDAContext>);
} // namespace caffe2

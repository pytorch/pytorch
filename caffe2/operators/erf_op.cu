#include "caffe2/operators/erf_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

__global__ void ErfGradientCUDAKernel(
    const int N,
    const float* dY,
    const float* X,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = 2.0f / sqrtf(PI) * expf(-powf(__ldg(X+i), 2.0f)) * __ldg(dY + i);
#else
    dX[i] = 2.0f / sqrtf(PI) * expf(-powf(X[i], 2.0f)) * dY[i];
#endif
  }
}

} // namespace

template <>
template <typename T>
bool ErfGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  ErfGradientCUDAKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Erf,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        ErfFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ErfGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        ErfGradientFunctor<CUDAContext>>);

} // namespace caffe2

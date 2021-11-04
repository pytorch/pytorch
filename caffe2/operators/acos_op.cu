#include "caffe2/operators/acos_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

__global__ void AcosGradientCUDAKernel(
    const int N,
    const float* dY,
    const float* X,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = -__ldg(dY + i) * rsqrtf(1.0f - __ldg(X + i) * __ldg(X + i));
#else
    dX[i] = -dY[i] * rsqrtf(1.0f - X[i] * X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool AcosGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  AcosGradientCUDAKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Acos,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        AcosFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    AcosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        AcosGradientFunctor<CUDAContext>>);

} // namespace caffe2

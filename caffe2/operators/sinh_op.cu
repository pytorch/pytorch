#include "caffe2/operators/sinh_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

__global__ void SinhGradientCUDAKernel(
    const int N,
    const float* dY,
    const float* X,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * coshf(__ldg(X + i));
#else
    dX[i] = dY[i] * coshf(X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SinhGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& /* dY_dims */,
    const std::vector<int>& X_dims,
    const T* dY,
    const T* X,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  SinhGradientCUDAKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Sinh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SinhFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    SinhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SinhGradientFunctor<CUDAContext>>);

} // namespace caffe2

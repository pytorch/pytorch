#include "caffe2/operators/cos_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
CosGradientCUDAKernel(const int N, const T* dY, const T* X, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = -__ldg(dY + i) * sin(__ldg(X + i));
#else
    dX[i] = -dY[i] * sin(X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CosGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  CosGradientCUDAKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Cos,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        CosFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    CosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        CosGradientFunctor<CUDAContext>>);

} // namespace caffe2

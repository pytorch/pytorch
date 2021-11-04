#include "caffe2/operators/cosh_op.h"

#include <c10/util/accumulate.h>
#include "caffe2/core/context_gpu.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

namespace {

__global__ void CoshGradientCUDAKernel(
    const int N,
    const float* dY,
    const float* X,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * sinhf(__ldg(X + i));
#else
    dX[i] = dY[i] * sinhf(X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CoshGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& /* dY_dims */,
    const std::vector<int>& X_dims,
    const T* dY,
    const T* X,
    T* dX,
    CUDAContext* context) const {
  const auto size = c10::multiply_integers(X_dims.cbegin(), X_dims.cend());
  CoshGradientCUDAKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Cosh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        CoshFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    CoshGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        CoshGradientFunctor<CUDAContext>>);

} // namespace caffe2

#include "caffe2/operators/abs_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
AbsGradientCUDAKernel(const int N, const T* dY, const T* X, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(X + i) == T(0)
        ? T(0)
        : (__ldg(X + i) > T(0) ? __ldg(dY + i) : -__ldg(dY + i));
#else
    dX[i] = X[i] == T(0) ? T(0) : (X[i] > T(0) ? dY[i] : -dY[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool AbsGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  AbsGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Abs,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        AbsFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        AbsGradientFunctor<CUDAContext>>);

} // namespace caffe2

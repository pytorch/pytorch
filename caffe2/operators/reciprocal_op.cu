#include "caffe2/operators/reciprocal_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
ReciprocalGradientCUDAKernel(const int N, const T* dY, const T* Y, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * (-__ldg(Y + i) * __ldg(Y + i));
#else
    dX[i] = dY[i] * (-Y[i] * Y[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool ReciprocalGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  ReciprocalGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Reciprocal,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        ReciprocalFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ReciprocalGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        ReciprocalGradientFunctor<CUDAContext>>);

} // namespace caffe2

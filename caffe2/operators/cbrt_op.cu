#include "caffe2/operators/cbrt_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
CbrtGradientCUDAKernel(const int N, const T* dY, const T* Y, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) / (__ldg(Y + i) * __ldg(Y + i) * T(3));
#else
    dX[i] = dY[i] / (Y[i] * Y[i] * T(3));
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CbrtGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* Y_dims */,
    const T* dY,
    const T* Y,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  CbrtGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Cbrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        CbrtFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    CbrtGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        CbrtGradientFunctor<CUDAContext>>);

} // namespace caffe2

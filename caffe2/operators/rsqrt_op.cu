#include "caffe2/operators/rsqrt_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
RsqrtGradientCUDAKernel(const int size, const T* dY, const T* Y, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, size) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * math::utils::Cube<T>(__ldg(Y + i)) *
        static_cast<T>(-0.5);
#else
    dX[i] = dY[i] * math::utils::Cube<T>(Y[i]) * static_cast<T>(-0.5);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool RsqrtGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* Y_dims */,
    const T* dY,
    const T* Y,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  RsqrtGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Rsqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        RsqrtFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    RsqrtGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        RsqrtGradientFunctor<CUDAContext>>);

} // namespace caffe2

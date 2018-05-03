#include "caffe2/operators/rsqrt_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T CubeCUDA(const T x) {
  return x * x * x;
}

template <typename T>
__global__ void
RSqrtGradientCUDAKernel(const int size, const T* dY, const T* Y, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, size) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * CubeCUDA(__ldg(Y + i)) * static_cast<T>(-0.5);
#else
    dX[i] = dY[i] * CubeCUDA(Y[i]) * static_cast<T>(-0.5);
#endif
  }
}

} // namespace

template <>
template <typename T>
void RSqrtGradientFunctor<CUDAContext>::Run<T>(
    const int size,
    const T* dY,
    const T* Y,
    T* dX,
    CUDAContext* context) const {
  RSqrtGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, Y, dX);
}

REGISTER_CUDA_OPERATOR(
    RSqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        RSqrtFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    RSqrtGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        WithoutBroadcast<RSqrtGradientFunctor<CUDAContext>>>);

} // namespace caffe2

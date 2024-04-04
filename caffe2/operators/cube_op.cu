#include "caffe2/operators/cube_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
CubeGradientCUDAKernel(const int N, const T* dY, const T* X, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * __ldg(X + i) * __ldg(X + i) * T(3);
#else
    dX[i] = dY[i] * X[i] * X[i] * T(3);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CubeGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* X_dims */,
    const T* dY,
    const T* X,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  CubeGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Cube,
    UnaryElementwiseOp<NumericTypes, CUDAContext, CubeFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    CubeGradient,
    BinaryElementwiseOp<
        NumericTypes,
        CUDAContext,
        CubeGradientFunctor<CUDAContext>>);

} // namespace caffe2

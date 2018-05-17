#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void AcosKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = acos(X[i]);
  }
}

template <typename T>
__global__ void AcosGradientKernel(const int N, const T* X, const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = -dY[i] / sqrt(1 - X[i] * X[i]);
  }
}

struct AcosCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    AcosKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

struct AcosGradientCUDAFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* x,
      const T* dy,
      T* dx,
      CUDAContext* device_context) {
    AcosGradientKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, dy, dx);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Acos,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, AcosCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    AcosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        WithoutBroadcast<AcosGradientCUDAFunctor>>);
} // namespace caffe2

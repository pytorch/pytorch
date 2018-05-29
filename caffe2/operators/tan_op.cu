#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void TanKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = tan(X[i]);
  }
}

template <typename T>
__global__ void TanGradientKernel(const int N, const T* X, const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] / pow(cos(X[i]), 2);
  }
}

struct TanCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    TanKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

struct TanGradientCUDAFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* x,
      const T* dy,
      T* dx,
      CUDAContext* device_context) {
    TanGradientKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, dy, dx);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Tan,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, TanCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    TanGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        WithoutBroadcast<TanGradientCUDAFunctor>>);
} // namespace caffe2

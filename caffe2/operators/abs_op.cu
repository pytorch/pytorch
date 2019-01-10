#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void AbsKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = fabs(X[i]);
  }
}

template <typename T>
__global__ void AbsGradientKernel(const int N, const T* X, const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = X[i] == T(0) ? T(0) : (X[i] > T(0) ? dY[i] : -dY[i]);
  }
}

struct AbsCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    AbsKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

struct AbsGradientCUDAFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* x,
      const T* dy,
      T* dx,
      CUDAContext* device_context) {
    AbsGradientKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, dy, dx);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Abs,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, AbsCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        WithoutBroadcast<AbsGradientCUDAFunctor>>);
} // namespace caffe2

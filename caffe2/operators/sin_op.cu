#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void SinKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = sin(X[i]);
  }
}

template <typename T>
__global__ void SinGradientKernel(const int N, const T* X, const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] * cos(X[i]);
  }
}

struct SinCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    SinKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

struct SinGradientCUDAFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* x,
      const T* dy,
      T* dx,
      CUDAContext* device_context) {
    SinGradientKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, dy, dx);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Sin,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, SinCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    SinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        WithoutBroadcast<SinGradientCUDAFunctor>>);
} // namespace caffe2

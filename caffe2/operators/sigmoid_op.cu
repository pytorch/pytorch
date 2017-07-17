#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void SigmoidKernel(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = 1. / (1. + exp(-x[i]));
  }
}

template <typename T>
__global__ void SigmoidGradientKernel(const int N, const T* y, const T* dy,
                              T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * y[i] * (1. - y[i]);
  }
}

struct SigmoidCUDAFunctor {
  template <typename T>
  inline void operator()(const int n, const T* x,
                         T* y, CUDAContext* device_context) {
    SigmoidKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                    0, device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

struct SigmoidGradientCUDAFunctor {
  template <typename T>
  inline void Run(const int n, const T* y, const T* dy,
                  T* dx, CUDAContext* device_context) {
    SigmoidGradientKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                            0, device_context->cuda_stream()>>>(n, y, dy, dx);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Sigmoid,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, SigmoidCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    SigmoidGradient, BinaryElementwiseOp<
        TensorTypes<float>, CUDAContext,
        WithoutBroadcast<SigmoidGradientCUDAFunctor>>);
}  // namespace caffe2

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

template <typename T>
struct SigmoidCUDAFunctor {
  inline void operator()(const int n, const float* x,
                         float* y, CUDAContext* device_context) {
    SigmoidKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                    0, device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

template <typename T>
struct SigmoidGradientCUDAFunctor {
  inline void operator()(const int n, const T* y, const T* dy,
                         T* dx, CUDAContext* device_context) {
    SigmoidGradientKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                            0, device_context->cuda_stream()>>>(n, y, dy, dx);
    return;
  }
};

namespace {
REGISTER_CUDA_OPERATOR(
    Sigmoid,
    UnaryElementwiseOp<float, CUDAContext, SigmoidCUDAFunctor<float> >);
REGISTER_CUDA_OPERATOR(
    SigmoidGradient, BinaryElementwiseOp<float, CUDAContext,
                                     SigmoidGradientCUDAFunctor<float> >);
}  // namespace
}  // namespace caffe2

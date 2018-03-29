#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/math_ops.h"

namespace caffe2 {

struct SqrCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    math::Sqr<T, CUDAContext>(n, x, y, device_context);
  }
};

template <typename T>
__global__ void SignKernel(int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = (-T(1) * (x[i] < 0)) + (x[i] > 0);
  }
}

struct SignCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    SignKernel<<<
        CAFFE_GET_BLOCKS(n),
        CAFFE_CUDA_NUM_THREADS,
        0,
        device_context->cuda_stream()>>>(n, x, y);
  }
};

REGISTER_CUDA_OPERATOR(
    Sqr,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, SqrCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    Sign,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, SignCUDAFunctor>);
}

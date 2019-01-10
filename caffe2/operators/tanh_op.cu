#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void TanhKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = tanh(X[i]);
  }
}

template <typename T>
__global__ void TanhGradientKernel(const int N, const T* Y, const T* dY,
                              T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i]*(1 - Y[i]*Y[i]);
  }
}

struct TanhCUDAFunctor {
  template <typename T>
  inline void operator()(const int n, const T* x,
                         T* y, CUDAContext* device_context) {
    TanhKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                    0, device_context->cuda_stream()>>>(n, x, y);
    return;
  }
  inline bool InplaceAllowed() {
    return true;
  }
};

struct TanhGradientCUDAFunctor {
  template <typename T>
  inline void Run(const int n, const T* y, const T* dy,
                  T* dx, CUDAContext* device_context) {
    TanhGradientKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                            0, device_context->cuda_stream()>>>(n, y, dy, dx);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Tanh, UnaryElementwiseOp<TensorTypes<float>, CUDAContext, TanhCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    TanhGradient, BinaryElementwiseOp<
        TensorTypes<float>, CUDAContext,
        WithoutBroadcast<TanhGradientCUDAFunctor>>);
}  // namespace caffe2

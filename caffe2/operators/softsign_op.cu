#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void SoftsignKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] / (1 + abs(X[i]));
  }
}

struct SoftsignCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    SoftsignKernel<T><<<
        CAFFE_GET_BLOCKS(n),
        CAFFE_CUDA_NUM_THREADS,
        0,
        device_context->cuda_stream()>>>(n, x, y);
    return;
  }
  inline bool InplaceAllowed() {
    return true;
  }
};

namespace {
REGISTER_CUDA_OPERATOR(
    Softsign,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, SoftsignCUDAFunctor>);
} // namespace
} // namespace caffe2

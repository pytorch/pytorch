#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void ExpKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __expf(X[i]);
  }
}

struct ExpCUDAFunctor {
  template <typename T>
  inline void operator()(const int n, const T* x,
                         T* y, CUDAContext* device_context) {
    ExpKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                    0, device_context->cuda_stream()>>>(n, x, y);
    return;
  }
  inline bool InplaceAllowed() {
    return true;
  }
};

REGISTER_CUDA_OPERATOR(
    Exp, UnaryElementwiseOp<TensorTypes<float>, CUDAContext, ExpCUDAFunctor>);
}  // namespace caffe2

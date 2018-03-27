#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void NegativeKernel(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = -x[i];
  }
}

struct NegativeCUDAFunctor {
  template <typename T>
  inline void operator()(const int n, const T* x,
                         T* y, CUDAContext* device_context) {
    NegativeKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                    0, device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Negative, UnaryElementwiseOp<
        TensorTypes<float, double, int, long>, CUDAContext,
        NegativeCUDAFunctor>);
}  // namespace caffe2

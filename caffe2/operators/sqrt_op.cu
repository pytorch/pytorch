#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/operators/math_ops.h"

namespace caffe2 {

template <typename T>
__global__ void SqrtKernel(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = sqrt(x[i]);
  }
}

struct SqrtCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    SqrtKernel<T>
        <<<CAFFE_GET_BLOCKS(n),
           CAFFE_CUDA_NUM_THREADS,
           0,
           device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Sqrt,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, SqrtCUDAFunctor>);
} // namespace caffe2

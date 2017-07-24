#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/math_ops.h"

namespace caffe2 {

struct LogCUDAFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CUDAContext* device_context) {
    math::Log<T, CUDAContext>(n, x, y, device_context);
  }
};

REGISTER_CUDA_OPERATOR(
    Log,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, LogCUDAFunctor>);
}

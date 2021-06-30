#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/replace_nan_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void
replace_nan_kernel(const T value, const int64_t size, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    if (isnan(X[i])) {
      Y[i] = value;
    } else {
      Y[i] = X[i];
    }
  }
}
} // namespace

template <>
template <typename T>
void ReplaceNaNOp<CUDAContext>::ReplaceNaN(
    const T& value,
    const int64_t size,
    const T* X,
    T* Y) {
  replace_nan_kernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(value, size, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
REGISTER_CUDA_OPERATOR(ReplaceNaN, ReplaceNaNOp<CUDAContext>);
} // namespace caffe2

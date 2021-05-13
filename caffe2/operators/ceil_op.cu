#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/ceil_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T>
__global__ void CeilKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = std::ceil(X[i]);
  }
}

template <>
bool CeilOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  CAFFE_ENFORCE_GT(X.numel(), 0);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  CeilKernel<<<
      CAFFE_GET_BLOCKS(X.numel()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.numel(), X.data<float>(), Y->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(Ceil, CeilOp<float, CUDAContext>);
} // namespace caffe2

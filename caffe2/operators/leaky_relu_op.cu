#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/leaky_relu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void LeakyReluKernel(const int N, const T alpha, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] >= 0 ? X[i] : X[i] * alpha;
  }
}
} // namespace

template <>
bool LeakyReluOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  LeakyReluKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(), alpha_, X.data<float>(), Y->mutable_data<float>());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(LeakyRelu, LeakyReluOp<float, CUDAContext>);
} // namespace
} // namespace caffe2

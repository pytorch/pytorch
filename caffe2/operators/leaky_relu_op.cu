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

template <typename T>
__global__ void LeakyReluGradientKernel(
    const int N,
    const T alpha,
    const T* Y,
    const T* dY,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = Y[i] >= 0 ? dY[i] : dY[i] * alpha;
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

template <>
bool LeakyReluGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& Y = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(Y);
  CAFFE_ENFORCE_EQ(Y.size(), dY.size());
  LeakyReluGradientKernel<<<
      CAFFE_GET_BLOCKS(Y.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      Y.size(),
      alpha_,
      Y.data<float>(),
      dY.data<float>(),
      dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(LeakyRelu, LeakyReluOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LeakyReluGradient,
    LeakyReluGradientOp<float, CUDAContext>);
} // namespace caffe2

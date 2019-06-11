#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/selu_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void SeluKernel(const int N, const T* X, T* Y, T alpha_, T lambda_) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = lambda_ * (X[i] > 0 ? X[i] : alpha_ * __expf(X[i]) - alpha_);
  }
}

template <typename T>
__global__ void SeluGradientKernel(
    const int N,
    const T* Y,
    const T* dY,
    T* dX,
    T alpha_,
    T lambda_) {
  const T c = lambda_ * alpha_;
  CUDA_1D_KERNEL_LOOP(i, N) {
    // Reuse Y[i] to avoid computing exp(X[i])
    dX[i] = Y[i] > 0 ? lambda_ * dY[i] : dY[i] * (Y[i] + c);
  }
}
} // namespace

template <>
bool SeluOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  CAFFE_ENFORCE_GT(X.numel(), 0);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  SeluKernel<float>
      <<<CAFFE_GET_BLOCKS(X.numel()),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          X.numel(),
          X.data<float>(),
          Y->template mutable_data<float>(),
          alpha_,
          lambda_);
  return true;
}

template <>
bool SeluGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  CAFFE_ENFORCE_GT(Y.numel(), 0);
  CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());
  SeluGradientKernel<float>
      <<<CAFFE_GET_BLOCKS(Y.numel()),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          Y.numel(),
          Y.data<float>(),
          dY.data<float>(),
          dX->template mutable_data<float>(),
          alpha_,
          lambda_);
  return true;
}

REGISTER_CUDA_OPERATOR(Selu, SeluOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SeluGradient, SeluGradientOp<float, CUDAContext>);
} // namespace caffe2

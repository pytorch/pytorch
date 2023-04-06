#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/thresholded_relu_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void ThresholdedReluKernel(const int N, const T* X, T* Y, T alpha_) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] > alpha_ ? X[i] : 0;
  }
}

template <typename T>
__global__ void
ThresholdedReluGradientKernel(const int N, const T* Y, const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = Y[i] > 0 ? dY[i] : 0;
  }
}
} // namespace

template <>
bool ThresholdedReluOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);

  CAFFE_ENFORCE_GT(X.numel(), 0);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  ThresholdedReluKernel<<<
      CAFFE_GET_BLOCKS(X.numel()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.numel(), X.data<float>(), Y->template mutable_data<float>(), alpha_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool ThresholdedReluGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  CAFFE_ENFORCE_GT(Y.numel(), 0);
  CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());
  ThresholdedReluGradientKernel<<<
      CAFFE_GET_BLOCKS(Y.numel()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      Y.numel(),
      Y.data<float>(),
      dY.data<float>(),
      dX->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(ThresholdedRelu, ThresholdedReluOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ThresholdedReluGradient,
    ThresholdedReluGradientOp<float, CUDAContext>);
} // namespace caffe2

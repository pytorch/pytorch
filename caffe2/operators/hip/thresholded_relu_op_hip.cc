#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/thresholded_relu_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void ThresholdedReluKernel(const int N, const T* X, T* Y, T alpha_) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] > alpha_ ? X[i] : 0;
  }
}

template <typename T>
__global__ void
ThresholdedReluGradientKernel(const int N, const T* Y, const T* dY, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = Y[i] > 0 ? dY[i] : 0;
  }
}
} // namespace

template <>
bool ThresholdedReluOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  hipLaunchKernelGGL((ThresholdedReluKernel), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(X.size()), X.data<float>(), Y->mutable_data<float>(), alpha_);
  return true;
}

template <>
bool ThresholdedReluGradientOp<float, HIPContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.size(), 0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  hipLaunchKernelGGL((ThresholdedReluGradientKernel), dim3(CAFFE_GET_BLOCKS(Y.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(Y.size()), Y.data<float>(), dY.data<float>(), dX->mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(ThresholdedRelu, ThresholdedReluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    ThresholdedReluGradient,
    ThresholdedReluGradientOp<float, HIPContext>);
} // namespace caffe2

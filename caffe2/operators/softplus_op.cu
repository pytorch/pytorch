#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/softplus_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void SoftplusKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = log(exp(X[i]) + 1.0f);
  }
}

template <typename T>
__global__ void
SoftplusGradientKernel(const int N, const T* Y, const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float nexpY = exp(-Y[i]);
    dX[i] = dY[i] * (1 - nexpY);
  }
}
} // namespace

template <>
bool SoftplusOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_GT(X.size(), 0);
  Y->ResizeLike(X);
  SoftplusKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(), X.data<float>(), Y->mutable_data<float>());
  return true;
}

template <>
bool SoftplusGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(Y.size(), 0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  SoftplusGradientKernel<<<
      CAFFE_GET_BLOCKS(Y.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      Y.size(), Y.data<float>(), dY.data<float>(), dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Softplus, SoftplusOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SoftplusGradient,
    SoftplusGradientOp<float, CUDAContext>);
} // namespace caffe2

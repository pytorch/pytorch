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

  TORCH_DCHECK_GT(X.numel(), 0);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  SoftplusKernel<float>
      <<<CAFFE_GET_BLOCKS(X.numel()),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          X.numel(), X.data<float>(), Y->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool SoftplusGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);

  TORCH_DCHECK_GT(Y.numel(), 0);
  TORCH_DCHECK_EQ(dY.numel(), Y.numel());
  auto* dX = Output(0, Y.sizes(), at::dtype<float>());
  SoftplusGradientKernel<float>
      <<<CAFFE_GET_BLOCKS(Y.numel()),
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

REGISTER_CUDA_OPERATOR(Softplus, SoftplusOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SoftplusGradient,
    SoftplusGradientOp<float, CUDAContext>);
} // namespace caffe2

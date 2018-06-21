#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/hardtanh_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void HardtanhKernel(const int N, const T* X, T* Y, T min_val_, T max_val_) {
  // Utilize naive implementation of Hardtanh
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = fminf(max_val_, fmaxf(min_val_, X[i]));
  }
}

template <typename T>
__global__ void HardtanhGradientKernel(
    const int N,
    const T* Y,
    const T* dY,
    T* dX,
    T min_val_,
    T max_val_) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = Y[i] >= max_val_ ? 0 : (Y[i] <= min_val_ ? 0 : dY[i]);
  }
}
} // namespace

template <>
bool HardtanhOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  HardtanhKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(), X.data<float>(), Y->mutable_data<float>(), min_val_, max_val_);
  return true;
}

template <>
bool HardtanhGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(1);
  auto& dY = Input(0);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.size(), 0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  HardtanhGradientKernel<<<
      CAFFE_GET_BLOCKS(Y.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      Y.size(),
      Y.data<float>(),
      dY.data<float>(),
      dX->mutable_data<float>(),
      min_val_,
      max_val_);
  return true;
}

REGISTER_CUDA_OPERATOR(Hardtanh, HardtanhOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(HardtanhGradient, HardtanhGradientOp<float, CUDAContext>);
} // namespace caffe2

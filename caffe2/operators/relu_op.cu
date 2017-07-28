#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void ReluKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] > 0 ? X[i] : 0;
  }
}

template <typename T>
__global__ void ReluGradientKernel(const int N, const T* Y, const T* dY,
                              T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = Y[i] > 0 ? dY[i] : 0;
  }
}
}  // namespace

template <>
bool ReluOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  ReluKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
               0, context_.cuda_stream()>>>(
      X.size(), X.data<float>(), Y->mutable_data<float>());
  return true;
}

template <>
bool ReluGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.size(), 0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  ReluGradientKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS,
                       0, context_.cuda_stream()>>>(
      Y.size(), Y.data<float>(), dY.data<float>(), dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Relu, ReluOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReluGradient, ReluGradientOp<float, CUDAContext>);
}  // namespace caffe2

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {
namespace {
template <typename dtype>
__global__ void ReluKernel(const int N, const dtype* X, dtype* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] > 0 ? X[i] : 0;
  }
}

template <typename dtype>
__global__ void ReluGradientKernel(const int N, const dtype* X, const dtype* dY,
                              dtype* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] * (X[i] > 0);
  }
}
}  // namespace

template <>
bool ReluOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  ReluKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
               0, device_context_.cuda_stream()>>>(
      X.size(), X.data(), Y->mutable_data());
  return true;
}

template <>
bool ReluGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(X.size(), 0);
  DCHECK_EQ(dY.size(), X.size());
  dX->ReshapeLike(X);
  ReluGradientKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                       0, device_context_.cuda_stream()>>>(
      X.size(), X.data(), dY.data(), dX->mutable_data());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(Relu, ReluOp<float, CUDAContext>)
REGISTER_CUDA_OPERATOR(ReluGradient, ReluGradientOp<float, CUDAContext>)
}  // namespace
}  // namespace caffe2

#include <cuda_fp16.h>

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

template <>
__global__ void ReluKernel<half>(const int N, const half* X, half* Y) {
  const half kZero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    Y[i] = __hgt(X[i], kZero) ? X[i] : kZero;
#else
    Y[i] = (__half2float(X[i]) > 0) ? X[i] : kZero;
#endif
  }
}

template <>
__global__ void ReluGradientKernel(const int N, const half* Y, const half* dY,
                                   half* dX) {
  const half kZero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    dX[i] = __hgt(Y[i], kZero) ? dY[i] : kZero;
#else
    dX[i] = (__half2float(Y[i]) > 0) ? dY[i] : kZero;
#endif
  }
}

}  // namespace

template <>
bool ReluOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  ReluKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
               0, device_context_.cuda_stream()>>>(
      X.size(), X.data<float>(), Y->mutable_data<float>());
  return true;
}

template <>
bool ReluGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_GT(Y.size(), 0);
  CAFFE_DCHECK_EQ(dY.size(), Y.size());
  dX->ReshapeLike(Y);
  ReluGradientKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS,
                       0, device_context_.cuda_stream()>>>(
      Y.size(), Y.data<float>(), dY.data<float>(), dX->mutable_data<float>());
  return true;
}

template <>
bool ReluOp<float16, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  ReluKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
               0, device_context_.cuda_stream()>>>(
      X.size(), reinterpret_cast<const half*>(X.data<float16>()),
      reinterpret_cast<half*>(Y->mutable_data<float16>()));
  return true;
}

template <>
bool ReluGradientOp<float16, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_GT(Y.size(), 0);
  CAFFE_DCHECK_EQ(dY.size(), Y.size());
  dX->ReshapeLike(Y);
  ReluGradientKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS,
                       0, device_context_.cuda_stream()>>>(
      Y.size(), reinterpret_cast<const half*>(Y.data<float16>()),
      reinterpret_cast<const half*>(dY.data<float16>()),
      reinterpret_cast<half*>(dX->mutable_data<float16>()));
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(Relu, ReluOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReluGradient, ReluGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReluFp16, ReluOp<float16, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReluGradientFp16, ReluGradientOp<float16, CUDAContext>);
}  // namespace
}  // namespace caffe2

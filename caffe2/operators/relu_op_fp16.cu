#include "caffe2/core/common_gpu.h"
#ifdef CAFFE_HAS_CUDA_FP16

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {
namespace {
__global__ void ReluKernelHalf(const int N, const half* X, half* Y) {
  const half kZero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    Y[i] = __hgt(X[i], kZero) ? X[i] : kZero;
#else
    Y[i] = (__half2float(X[i]) > 0) ? X[i] : kZero;
#endif
  }
}

__global__ void ReluKernelHalf2(const int N, const half2* X, half2* Y) {
  const half2 kZero = __float2half2_rn(0.0);
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    Y[i] = __hmul2(__hgt2(X[i], kZero), X[i]);
#else
    float2 xx = __half22float2(X[i]);
    Y[i] = __floats2half2_rn(xx.x > 0 ? xx.x : 0.f,
                             xx.y > 0 ? xx.y : 0.f);
#endif
  }
}

__global__ void ReluGradientKernelHalf(
    const int N, const half* Y, const half* dY, half* dX) {
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
bool ReluOp<float16, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  if (X.size() % 2 == 0) {
    ReluKernelHalf2<<<CAFFE_GET_BLOCKS(X.size() / 2), CAFFE_CUDA_NUM_THREADS,
                      0, context_.cuda_stream()>>>(
        X.size() / 2, reinterpret_cast<const half2*>(X.data<float16>()),
        reinterpret_cast<half2*>(Y->mutable_data<float16>()));
    return true;
  } else {
    ReluKernelHalf<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                     0, context_.cuda_stream()>>>(
        X.size(), reinterpret_cast<const half*>(X.data<float16>()),
        reinterpret_cast<half*>(Y->mutable_data<float16>()));
    return true;
  }
}

template <>
bool ReluGradientOp<float16, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.size(), 0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  ReluGradientKernelHalf<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS,
                           0, context_.cuda_stream()>>>(
      Y.size(), reinterpret_cast<const half*>(Y.data<float16>()),
      reinterpret_cast<const half*>(dY.data<float16>()),
      reinterpret_cast<half*>(dX->mutable_data<float16>()));
  return true;
}

OPERATOR_SCHEMA(ReluFp16);
OPERATOR_SCHEMA(ReluFp16Gradient);

REGISTER_CUDA_OPERATOR(ReluFp16, ReluOp<float16, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReluFp16Gradient, ReluGradientOp<float16, CUDAContext>);
}  // namespace caffe2

#endif  // CAFFE_HAS_CUDA_FP16

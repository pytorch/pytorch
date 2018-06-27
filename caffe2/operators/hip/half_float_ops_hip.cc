#include "hip/hip_runtime.h"
#include "caffe2/operators/half_float_ops.h"

#include "caffe2/core/hip/context_hip.h"

#ifdef CAFFE_HAS_HIP_FP16

namespace caffe2 {
namespace {
__global__ void FloatToHalfKernel(const int N, const float* X, half* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = __float2half(X[i]);
  }
}

__global__ void HalfToFloatKernel(const int N, const half* X, float* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = __half2float(X[i]);
  }
}
}

template <>
bool FloatToHalfOp<HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  hipLaunchKernelGGL((FloatToHalfKernel), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(X.size()),
      X.data<float>(),
      reinterpret_cast<half*>(Y->mutable_data<float16>()));
  return true;
}

template <>
bool HalfToFloatOp<HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  hipLaunchKernelGGL((HalfToFloatKernel), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(X.size()),
      reinterpret_cast<const half*>(X.data<float16>()),
      Y->mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(FloatToHalf, FloatToHalfOp<HIPContext>);
REGISTER_HIP_OPERATOR(HalfToFloat, HalfToFloatOp<HIPContext>);
} // namespace caffe2

#endif // CAFFE_HAS_HIP_FP16

#include "caffe2/operators/minmax_ops.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SelectGradientCUDAKernel(
    const int N,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dX[i] = __ldg(X + i) == __ldg(Y + i) ? __ldg(dY + i) : T(0);
#else
    dX[i] = X[i] == Y[i] ? dY[i] : T(0);
#endif
  }
}

} // namespace

template <>
bool SelectGradientOpBase<float, CUDAContext>::RunOnDevice() {
  const auto& Y = Input(0);
  const auto& dY = Input(1);
  const int N = Y.numel();
  const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
  const float* dY_data = dY.data<float>();
  const float* Y_data = Y.data<float>();
  for (int i = 0; i < OutputSize(); i++) {
    const auto& Xi = Input(i + 2);
    auto* dXi = Output(i, Xi.sizes(), at::dtype<float>());
    const float* Xi_data = Xi.data<float>();
    float* dXi_data = dXi->mutable_data<float>();
    if (N > 0) {
      SelectGradientCUDAKernel<float>
          <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
              N, dY_data, Xi_data, Y_data, dXi_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Min, MinOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MinGradient, MinGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Max, MaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MaxGradient, MaxGradientOp<float, CUDAContext>);

} // namespace caffe2

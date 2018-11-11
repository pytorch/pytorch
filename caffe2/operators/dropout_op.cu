#include "caffe2/operators/dropout_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void DropoutFowardCUDAKernel(
    const int N,
    const T ratio,
    const T* X,
    const T* uniform,
    T* Y,
    bool* mask) {
  const T scale = T(1) / (T(1) - ratio);
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    const bool cur_mask = (__ldg(uniform + i) > ratio);
    Y[i] = __ldg(X + i) * static_cast<T>(cur_mask) * scale;
#else
    const bool cur_mask = (uniform[i] > ratio);
    Y[i] = X[i] * static_cast<T>(cur_mask) * scale;
#endif
    mask[i] = cur_mask;
  }
}

template <typename T>
__global__ void DropoutBackwardCUDAKenel(
    const int N,
    const float scale,
    const T* dY,
    const bool* mask,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * static_cast<T>(__ldg(mask + i)) * scale;
#else
    dX[i] = dY[i] * static_cast<T>(mask[i]) * scale;
#endif
  }
}

} // namespace

template <>
void DropoutOp<float, CUDAContext>::DropoutForward(
    const int N,
    const float* X,
    float* Y,
    bool* mask) {
  float* uniform_data = Y;
  if (Y == X) {
    uniform_.Resize(N);
    uniform_data = uniform_.mutable_data<float>();
  }
  CURAND_ENFORCE(
      curandGenerateUniform(context_.curand_generator(), uniform_data, N));
  DropoutFowardCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, ratio_, X, uniform, Y, mask);
}

template <>
void DropoutGradientOp<float, CUDAContext>::DropoutBackward(
    const int N,
    const float* dY,
    const bool* mask,
    float* dX) {
  const float scale = 1.0f / (1.0f - ratio_);
  DropoutBackwardCUDAKenel<float>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, scale, dY, mask, dX);
}

REGISTER_CUDA_OPERATOR(Dropout, DropoutOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(DropoutGrad, DropoutGradientOp<float, CUDAContext>);

} // namespace caffe2

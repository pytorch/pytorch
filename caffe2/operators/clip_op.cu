#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/clip_op.h"

namespace caffe2 {
namespace {

template <typename T>
__device__ T cuda_min(T x, T y);
template <typename T>
__device__ T cuda_max(T x, T y);
template <>
__device__ float cuda_min(float x, float y) { return fminf(x, y); }
template <>
__device__ float cuda_max(float x, float y) { return fmaxf(x, y); }

// Disabled since we don't use it right now.
/*
template <>
__device__ double cuda_min(double x, double y) { return fmin(x, y); }
template <>
__device__ double cuda_max(double x, double y) { return fmax(x, y); }
*/


template <typename T>
__global__ void ClipKernel(const int N, const T minval, const T maxval,
                           const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = cuda_min<T>(cuda_max<T>(X[i], minval), maxval);
  }
}

template <typename T>
__global__ void ClipGradientKernel(const int N,  const T minval,
                                   const T maxval, const T* Y,
                                   const T* dY, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] * (Y[i] > minval && Y[i] < maxval);
  }
}
}  // namespace

template <>
bool ClipOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  ClipKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
               0, context_.cuda_stream()>>>(
      X.size(), min_, max_, X.data<float>(), Y->mutable_data<float>());
  return true;
}

template <>
bool ClipGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.size(), 0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  ClipGradientKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS,
                       0, context_.cuda_stream()>>>(
      Y.size(), min_, max_, Y.data<float>(), dY.data<float>(),
      dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Clip, ClipOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ClipGradient, ClipGradientOp<float, CUDAContext>);
}  // namespace caffe2

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/clip_op.h"

namespace caffe2 {
namespace {

template <typename dtype>
__device__ dtype cuda_min(dtype x, dtype y);
template <typename dtype>
__device__ dtype cuda_max(dtype x, dtype y);
template <>
__device__ float cuda_min(float x, float y) { return fminf(x, y); }
template <>
__device__ float cuda_max(float x, float y) { return fmaxf(x, y); }
template <>
__device__ double cuda_min(double x, double y) { return fmin(x, y); }
template <>
__device__ double cuda_max(double x, double y) { return fmax(x, y); }



template <typename dtype>
__global__ void ClipKernel(const int N, const dtype minval, const dtype maxval,
                           const dtype* X, dtype* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = cuda_min<dtype>(cuda_max<dtype>(X[i], minval), maxval);
  }
}

template <typename dtype>
__global__ void ClipGradientKernel(const int N,  const dtype minval,
                                   const dtype maxval, const dtype* X,
                                   const dtype* dY, dtype* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] * (X[i] > minval && X[i] < maxval);
  }
}
}  // namespace

template <>
bool ClipOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  ClipKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
               0, device_context_.cuda_stream()>>>(
      X.size(), min_, max_, X.data(), Y->mutable_data());
  return true;
}

template <>
bool ClipGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(X.size(), 0);
  DCHECK_EQ(dY.size(), X.size());
  dX->ReshapeLike(X);
  ClipGradientKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
                       0, device_context_.cuda_stream()>>>(
      X.size(), min_, max_, X.data(), dY.data(), dX->mutable_data());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(Clip, ClipOp<float, CUDAContext>)
REGISTER_CUDA_OPERATOR(ClipGradient, ClipGradientOp<float, CUDAContext>)
}  // namespace
}  // namespace caffe2

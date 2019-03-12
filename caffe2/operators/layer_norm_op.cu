#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/operators/layer_norm_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/reduce.cuh"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ComputeStdDevAndFusedParamsCUDAKernel(
    const int N,
    const T epsilon,
    const T* mean,
    const T* var,
    T* stddev,
    T* scale,
    T* bias);

template <>
__global__ void ComputeStdDevAndFusedParamsCUDAKernel<float>(
    const int N,
    const float epsilon,
    const float* mean,
    const float* var,
    float* stddev,
    float* scale,
    float* bias) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N) {
#if __CUDA_ARCH__ >= 350
    const float rstd = rsqrtf(__ldg(var + index) + epsilon);
    stddev[index] = rstd * (__ldg(var + index) + epsilon);
    scale[index] = rstd;
    bias[index] = -rstd * __ldg(mean + index);
#else
    const float rstd = rsqrtf(var[index] + epsilon);
    stddev[index] = rstd * (var[index] + epsilon);
    scale[index] = rstd;
    bias[index] = -rstd * mean[index];
#endif
  }
}

template <typename T>
__global__ void LayerNormForwardCUDAKernel(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <>
__global__ void LayerNormForwardCUDAKernel<float>(
    const int M,
    const int N,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int size = M * N;
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < size) {
    const int i = index / N;
#if __CUDA_ARCH__ >= 350
    Y[index] = fmaf(__ldg(X + index), __ldg(scale + i), __ldg(bias + i));
#else
    Y[index] = fmaf(X[index], scale[i], bias[i]);
#endif
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    T* ds,
    T* db) {
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  const int i = blockIdx.x;
  T ds_val = 0;
  T db_val = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
#if __CUDA_ARCH__ >= 350
    ds_val += __ldg(dY + index) * __ldg(X + index);
    db_val += __ldg(dY + index);
#else
    ds_val += dY[index] * X[index];
    db_val += dY[index];
#endif
  }
  ds_val = BlockReduce<T>(ds_storage).Sum(ds_val);
  db_val = BlockReduce<T>(db_storage).Sum(db_val);
  if (threadIdx.x == 0) {
    ds[i] = ds_val;
    db[i] = db_val;
  }
}

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    const int M,
    const int N,
    const T* mean,
    const T* sig,
    const T* ds,
    const T* db,
    T* dY_scale,
    T* X_scale,
    T* bias);

template <>
__global__ void ComputeFusedParamsCUDAKernel<float>(
    const int M,
    const int N,
    const float* mean,
    const float* sig,
    const float* ds,
    const float* db,
    float* dY_scale,
    float* X_scale,
    float* bias) {
  const float scale = 1.0f / static_cast<float>(N);
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < M) {
#if __CUDA_ARCH__ >= 350
    const float rsig = 1.0f / __ldg(sig + index);
    const float X_scale_val =
        fmaf(__ldg(db + index), __ldg(mean + index), -__ldg(ds + index)) *
        math::utils::Cube<float>(rsig) * scale;
    dY_scale[index] = rsig;
    X_scale[index] = X_scale_val;
    bias[index] = -fmaf(
        X_scale_val, __ldg(mean + index), __ldg(db + index) * rsig * scale);
#else
    const float rsig = 1.0f / sig[index];
    const float X_scale_val = fmaf(db[index], mean[index], -ds[index]) *
        math::utils::Cube<float>(rsig) * scale;
    dY_scale[index] = rsig;
    X_scale[index] = X_scale_val;
    bias[index] = -fmaf(X_scale_val, mean[index], db[index] * rsig * scale);
#endif
  }
}

template <typename T>
__global__ void LayerNormBackwardCUDAKenrel(
    const int M,
    const int N,
    const T* dY_scale,
    const T* dY,
    const T* X_scale,
    const T* X,
    const T* bias,
    T* dX);

template <>
__global__ void LayerNormBackwardCUDAKenrel<float>(
    const int M,
    const int N,
    const float* dY_scale,
    const float* dY,
    const float* X_scale,
    const float* X,
    const float* bias,
    float* dX) {
  const int size = M * N;
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < size) {
    const int i = index / N;
#if __CUDA_ARCH__ >= 350
    dX[index] = fmaf(
        __ldg(dY + index),
        __ldg(dY_scale + i),
        fmaf(__ldg(X + index), __ldg(X_scale + i), __ldg(bias + i)));
#else
    dX[index] =
        fmaf(dY[index], dY_scale[i], fmaf(X[index], X_scale[i], bias[i]));
#endif
  }
}

} //  namespace

template <>
template <typename T>
void LayerNormOp<CUDAContext>::ComputeStdDevAndFusedParams(
    const int N,
    const T* mean,
    const T* var,
    T* stddev,
    T* scale,
    T* bias,
    float epsilon,
    CUDAContext* context) {
  const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
  ComputeStdDevAndFusedParamsCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          N, static_cast<T>(epsilon), mean, var, stddev, scale, bias);
}

template <>
template <typename T>
void LayerNormOp<CUDAContext>::LayerNormForward(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y,
    CUDAContext* context) {
  const int K = math::DivUp(M * N, CAFFE_CUDA_NUM_THREADS);
  LayerNormForwardCUDAKernel<T>
      <<<K, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          M, N, X, scale, bias, Y);
}

REGISTER_CUDA_OPERATOR(LayerNorm, LayerNormOp<CUDAContext>);

template <>
template <typename T>
void LayerNormGradientOp<CUDAContext>::ComputeInternalGradients(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    T* ds,
    T* db) {
  ComputeInternalGradientsCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          M, N, dY, X, ds, db);
}

template <>
template <typename T>
void LayerNormGradientOp<CUDAContext>::ComputeFusedParams(
    const int M,
    const int N,
    const T* mean,
    const T* sig,
    const T* ds,
    const T* db,
    T* dY_scale,
    T* X_scale,
    T* bias) {
  const int K = math::DivUp(M, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamsCUDAKernel<T>
      <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          M, N, mean, sig, ds, db, dY_scale, X_scale, bias);
}

template <>
template <typename T>
void LayerNormGradientOp<CUDAContext>::LayerNormBackward(
    const int M,
    const int N,
    const T* dY_scale,
    const T* dY,
    const T* X_scale,
    const T* X,
    const T* bias,
    T* dX) {
  const int K = math::DivUp(M * N, CAFFE_CUDA_NUM_THREADS);
  LayerNormBackwardCUDAKenrel<T>
      <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          M, N, dY_scale, dY, X_scale, X, bias, dX);
}

REGISTER_CUDA_OPERATOR(LayerNormGradient, LayerNormGradientOp<CUDAContext>);

} // namespace caffe2

C10_REGISTER_CAFFE2_OPERATOR_CUDA(
    LayerNorm,
    caffe2::LayerNormOp<caffe2::CUDAContext>)

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CUDA(
    _c10_ops::LayerNorm(),
    C10LayerNorm_DontUseThisOpYet);
}

#include "caffe2/operators/layer_norm_op.h"

#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

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
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    const float rstd = rsqrtf(__ldg(var + i) + epsilon);
    stddev[i] = rstd * (__ldg(var + i) + epsilon);
    scale[i] = rstd;
    bias[i] = -rstd * __ldg(mean + i);
#else
    const float rstd = rsqrtf(var[i] + epsilon);
    stddev[i] = rstd * (var[i] + epsilon);
    scale[i] = rstd;
    bias[i] = -rstd * mean[i];
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
    T* Y) {
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
#if __CUDA_ARCH__ >= 350
    const float scale_val = __ldg(scale + i);
    const float bias_val = __ldg(bias + i);
#else
    const float scale_val = scale[i];
    const float bias_val = bias[i];
#endif
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const int index = i * N + j;
#if __CUDA_ARCH__ >= 350
      Y[index] = __ldg(X + index) * scale_val + bias_val;
#else
      Y[index] = X[index] * scale_val + bias_val;
#endif
    }
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
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
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
    __syncthreads();
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
    T* bias) {
  const T scale = T(1) / static_cast<T>(N);
  CUDA_1D_KERNEL_LOOP(i, M) {
#if __CUDA_ARCH__ >= 350
    const T rsig = T(1) / __ldg(sig + i);
    const T X_scale_val = (__ldg(db + i) * __ldg(mean + i) - __ldg(ds + i)) *
        math::utils::Cube<T>(rsig) * scale;
    dY_scale[i] = rsig;
    X_scale[i] = X_scale_val;
    bias[i] = -X_scale_val * __ldg(mean + i) - __ldg(db + i) * rsig * scale;
#else
    const T rsig = T(1) / sig[i];
    const T X_scale_val =
        (db[i] * mean[i] - ds[i]) * math::utils::Cube<T>(rsig) * scale;
    dY_scale[i] = rsig;
    X_scale[i] = X_scale_val;
    bias[i] = -X_scale_val * mean[i] - db[i] * rsig * scale;
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
    T* dX) {
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
#if __CUDA_ARCH__ >= 350
    const float dY_scale_val = __ldg(dY_scale + i);
    const float X_scale_val = __ldg(X_scale + i);
    const float bias_val = __ldg(bias + i);
#else
    const float dY_scale_val = dY_scale[i];
    const float X_scale_val = X_scale[i];
    const float bias_val = bias[i];
#endif
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const int index = i * N + j;
#if __CUDA_ARCH__ >= 350
      dX[index] = __ldg(dY + index) * dY_scale_val +
          __ldg(X + index) * X_scale_val + bias_val;
#else
      dX[index] = dY[index] * dY_scale_val + X[index] * X_scale_val + bias_val;
#endif
    }
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
  ComputeStdDevAndFusedParamsCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
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
  LayerNormForwardCUDAKernel<T>
      <<<std::min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(M, N, X, scale, bias, Y);
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
      <<<std::min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(M, N, dY, X, ds, db);
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
  ComputeFusedParamsCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(M),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
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
  LayerNormBackwardCUDAKenrel<T>
      <<<std::min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(M, N, dY_scale, dY, X_scale, X, bias, dX);
}

REGISTER_CUDA_OPERATOR(LayerNormGradient, LayerNormGradientOp<CUDAContext>);

} // namespace caffe2

C10_REGISTER_CAFFE2_OPERATOR_CUDA(
    LayerNorm,
    caffe2::LayerNormOp<caffe2::CUDAContext>)

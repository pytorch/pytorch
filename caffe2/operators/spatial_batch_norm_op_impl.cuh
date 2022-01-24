#ifndef CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_IMPL_CUH_
#define CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_IMPL_CUH_

#include "caffe2/operators/spatial_batch_norm_op.h"

#include <limits>

#include "caffe2/utils/cub_namespace.cuh"
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/reduce.cuh"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ComputeFusedParamCUDAKernel(
    const int C,
    const T epsilon,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* var,
    T* alpha,
    T* beta);

template <>
__global__ void ComputeFusedParamCUDAKernel<float>(
    const int C,
    const float epsilon,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* var,
    float* alpha,
    float* beta) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const float scale_x_rstd = scale[c] * rsqrtf(var[c] + epsilon);
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * mean[c] + bias[c];
  }
}

template <typename T>
__global__ void ComputeBatchMomentsCUDAKernel(
    const int C,
    const T scale,
    const T* batch_mean_sum,
    const T* batch_var_sum,
    T* mean,
    T* var) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const T mu = scale * batch_mean_sum[c];
    mean[c] = mu;
    var[c] = scale * batch_var_sum[c] - mu * mu;
  }
}

template <typename T>
__global__ void ComputeRunningMomentsAndFusedParamCUDAKernel(
    const int C,
    const int reduce_size,
    const T momentum,
    const T epsilon,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* var,
    T* running_mean,
    T* running_var,
    T* rstd,
    T* alpha,
    T* beta);

template <>
__global__ void ComputeRunningMomentsAndFusedParamCUDAKernel<float>(
    const int C,
    const int reduce_size,
    const float momentum,
    const float epsilon,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* var,
    float* running_mean,
    float* running_var,
    float* rstd,
    float* alpha,
    float* beta) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  const float a = 1.0f - momentum;
  const float b = momentum;
  const float unbias_scale = reduce_size == 1
      ? std::numeric_limits<float>::infinity()
      : static_cast<float>(reduce_size) / static_cast<float>(reduce_size - 1);
  if (c < C) {
    const float rstd_val = rsqrtf(var[c] + epsilon);
    const float scale_x_rstd = scale[c] * rstd_val;
    running_mean[c] = a * mean[c] + b * running_mean[c];
    running_var[c] = a * unbias_scale * var[c] + b * running_var[c];
    rstd[c] = rstd_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * mean[c] + bias[c];
  }
}

template <typename T>
__global__ void ComputeMultiBatchScaleBiasGradientsAndFusedParamsCUDAKernel(
    const int C,
    const T batch_scale,
    const T mean_scale,
    const T* scale,
    const T* mean,
    const T* rstd,
    const T* dscale_sum,
    const T* dbias_sum,
    T* dscale,
    T* dbias,
    T* alpha,
    T* beta,
    T* gamma) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const T dscale_val = dscale_sum[c] * batch_scale;
    const T dbias_val = dbias_sum[c] * batch_scale;
    const T scale_x_rstd = scale[c] * rstd[c];
    const T dscale_x_rstd = dscale_val * rstd[c];
    dscale[c] = dscale_val;
    dbias[c] = dbias_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] =
        scale_x_rstd * (mean[c] * dscale_x_rstd - dbias_val) * mean_scale;
  }
}

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void ComputeScaleBiasGradientsAndFusedParamsNCHWCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* scale,
    const T* mean,
    const T* rstd,
    T* dscale,
    T* dbias,
    T* alpha,
    T* beta,
    T* gamma) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage ds_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage db_storage;
  const int c = blockIdx.x;
  const T mean_scale = T(1) / static_cast<T>(N * HxW);
  T ds_val = 0;
  T db_val = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    for (int j = threadIdx.y; j < HxW; j += blockDim.y) {
      const int index = (i * C + c) * HxW + j;
      ds_val += dY[index] * X[index];
      db_val += dY[index];
    }
  }
  ds_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(ds_storage).Sum(ds_val);
  db_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(db_storage).Sum(db_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    ds_val = (ds_val - mean[c] * db_val) * rstd[c];
    const T scale_x_rstd = scale[c] * rstd[c];
    const T dscale_x_rstd = ds_val * rstd[c];
    dscale[c] = ds_val;
    dbias[c] = db_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] = scale_x_rstd * (mean[c] * dscale_x_rstd - db_val) * mean_scale;
  }
}

template <typename T>
__global__ void ComputeScaleGradientAndFusedParamsNHWCCUDAKernel(
    const int C,
    const T mean_scale,
    const T* dYxX,
    const T* dbias,
    const T* scale,
    const T* mean,
    const T* rstd,
    T* dscale,
    T* alpha,
    T* beta,
    T* gamma) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const T ds = (dYxX[c] - dbias[c] * mean[c]) * rstd[c];
    dscale[c] = ds;
    const T scale_x_rstd = scale[c] * rstd[c];
    const T dscale_x_rstd = ds * rstd[c];
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] = scale_x_rstd * (mean[c] * dscale_x_rstd - dbias[c]) * mean_scale;
  }
}

template <typename T>
__global__ void ComputeXGradientNCHWCUDAKernel(
    const int C,
    const int M,
    const int HxW,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX) {
  const int nc = blockIdx.x / M;
  const int c = nc % C;
  const int x = blockIdx.x % M * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (x < HxW) {
    const int index = nc * HxW + x;
    dX[index] = alpha[c] * dY[index] + beta[c] * X[index] + gamma[c];
  }
}

template <typename T>
__global__ void ComputeXGradientNHWCCUDAKernel(
    const int C,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX) {
  const int c = blockIdx.y * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const int index = blockIdx.x * C + c;
    dX[index] = alpha[c] * dY[index] + beta[c] * X[index] + gamma[c];
  }
}

} // namespace

template <>
template <typename T>
void SpatialBNOp<CUDAContext>::ComputeFusedParam(
    const int C,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* var,
    T* alpha,
    T* beta) {
  const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C, static_cast<T>(epsilon_), scale, bias, mean, var, alpha, beta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
template <typename T>
void SpatialBNOp<CUDAContext>::ComputeBatchMoments(
    const int N,
    const int C,
    const int HxW,
    const T* batch_mean_sum,
    const T* batch_var_sum,
    T* mean,
    T* var) {
  const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  const T scale = T(1) / static_cast<T>(num_batches_ * N * HxW);
  ComputeBatchMomentsCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C, scale, batch_mean_sum, batch_var_sum, mean, var);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
template <typename T>
void SpatialBNOp<CUDAContext>::ComputeRunningMomentsAndFusedParam(
    const int C,
    const int reduce_size,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* var,
    T* running_mean,
    T* running_var,
    T* rstd,
    T* alpha,
    T* beta) {
  const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  ComputeRunningMomentsAndFusedParamCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C,
          reduce_size,
          static_cast<T>(momentum_),
          static_cast<T>(epsilon_),
          scale,
          bias,
          mean,
          var,
          running_mean,
          running_var,
          rstd,
          alpha,
          beta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
template <typename T>
void SpatialBNGradientOp<CUDAContext>::
    ComputeMultiBatchScaleBiasGradientsAndFusedParams(
        const int N,
        const int C,
        const int HxW,
        const T* scale,
        const T* mean,
        const T* rstd,
        const T* dscale_sum,
        const T* dbias_sum,
        T* dscale,
        T* dbias,
        T* alpha,
        T* beta,
        T* gamma) {
  const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  const T batch_scale = T(1) / static_cast<T>(num_batches_);
  const T mean_scale = T(1) / static_cast<T>(N * HxW);
  ComputeMultiBatchScaleBiasGradientsAndFusedParamsCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C,
          batch_scale,
          mean_scale,
          scale,
          mean,
          rstd,
          dscale_sum,
          dbias_sum,
          dscale,
          dbias,
          alpha,
          beta,
          gamma);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
template <typename T>
void SpatialBNGradientOp<CUDAContext>::ComputeScaleBiasGradientsAndFusedParams(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* scale,
    const T* mean,
    const T* rstd,
    T* dscale,
    T* dbias,
    T* alpha,
    T* beta,
    T* gamma,
    T* scratch) {
  if (order_ == StorageOrder::NCHW) {
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE_1(
        HxW,
        ComputeScaleBiasGradientsAndFusedParamsNCHWCUDAKernel,
        T,
        C,
        context_.cuda_stream(),
        N,
        C,
        HxW,
        dY,
        X,
        scale,
        mean,
        rstd,
        dscale,
        dbias,
        alpha,
        beta,
        gamma);
  } else {
    ReinitializeTensor(&ones_, N * HxW, at::dtype<T>().device(CUDA));
    math::Set<T, CUDAContext>(
        N * HxW, T(1), ones_.template mutable_data<T>(), &context_);
    const T* ones_data = ones_.template data<T>();
    math::Mul<T, CUDAContext>(N * C * HxW, dY, X, scratch, &context_);
    math::Gemm<T, CUDAContext>(
        CblasTrans,
        CblasNoTrans,
        C,
        1,
        N * HxW,
        1.0f,
        scratch,
        ones_data,
        0.0f,
        dscale,
        &context_);
    math::Gemm<T, CUDAContext>(
        CblasTrans,
        CblasNoTrans,
        C,
        1,
        N * HxW,
        1.0f,
        dY,
        ones_data,
        0.0f,
        dbias,
        &context_);
    const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
    ComputeScaleGradientAndFusedParamsNHWCCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            C,
            T(1) / static_cast<T>(N * HxW),
            dscale,
            dbias,
            scale,
            mean,
            rstd,
            dscale,
            alpha,
            beta,
            gamma);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <>
template <typename T>
void SpatialBNGradientOp<CUDAContext>::ComputeXGradient(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX) {
  if (order_ == StorageOrder::NCHW) {
    const int M = math::DivUp(HxW, CAFFE_CUDA_NUM_THREADS);
    ComputeXGradientNCHWCUDAKernel<T>
        <<<N * C * M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            C, M, HxW, dY, X, alpha, beta, gamma, dX);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
    ComputeXGradientNHWCCUDAKernel<T>
        <<<dim3(N * HxW, M),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(C, dY, X, alpha, beta, gamma, dX);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_IMPL_CUH_

#ifndef CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_IMPL_CUH_
#define CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_IMPL_CUH_

#include "caffe2/operators/spatial_batch_norm_op.h"

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

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
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const float scale_x_rstd =
        __ldg(scale + c) * rsqrtf(__ldg(var + c) + epsilon);
    alpha[c] = scale_x_rstd;
    beta[c] = fmaf(-scale_x_rstd, __ldg(mean + c), __ldg(bias + c));
#else
    const float scale_x_rstd = scale[c] * rsqrtf(var[c] + epsilon);
    alpha[c] = scale_x_rstd;
    beta[c] = fmaf(-scale_x_rstd, mean[c], bias[c]);
#endif
  }
}

template <typename T>
__global__ void ComputeBatchMomentsCUDAKernel(
    const int C,
    const T scale,
    const T* batch_mean_sum,
    const T* batch_var_sum,
    T* mean,
    T* var);

template <>
__global__ void ComputeBatchMomentsCUDAKernel<float>(
    const int C,
    const float scale,
    const float* batch_mean_sum,
    const float* batch_var_sum,
    float* mean,
    float* var) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const float mu = scale * __ldg(batch_mean_sum + c);
    mean[c] = mu;
    var[c] = fmaf(scale, __ldg(batch_var_sum + c), -mu * mu);
#else
    const float mu = scale * batch_mean_sum[c];
    mean[c] = mu;
    var[c] = fmaf(scale, batch_var_sum[c], -mu * mu);
#endif
  }
}

template <typename T>
__global__ void ComputeRunningMomentsAndFusedParamCUDAKernel(
    const int C,
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
  if (c < C) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const float rstd_val = rsqrtf(__ldg(var + c) + epsilon);
    const float scale_x_rstd = __ldg(scale + c) * rstd_val;
    running_mean[c] = fmaf(a, __ldg(mean + c), b * __ldg(running_mean + c));
    running_var[c] = fmaf(a, __ldg(var + c), b * __ldg(running_var + c));
    rstd[c] = rstd_val;
    alpha[c] = scale_x_rstd;
    beta[c] = fmaf(-scale_x_rstd, __ldg(mean + c), __ldg(bias + c));
#else
    const float rstd_val = rsqrtf(var[c] + epsilon);
    const float scale_x_rstd = scale[c] * rstd_val;
    running_mean[c] = fmaf(a, mean[c], b * running_mean[c]);
    running_var[c] = fmaf(a, var[c], b * running_var[c]);
    rstd[c] = rstd_val;
    alpha[c] = scale_x_rstd;
    beta[c] = fmaf(-scale_x_rstd, mean[c], bias[c]);
#endif
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
    T* gamma);

template <>
__global__ void
ComputeMultiBatchScaleBiasGradientsAndFusedParamsCUDAKernel<float>(
    const int C,
    const float batch_scale,
    const float mean_scale,
    const float* scale,
    const float* mean,
    const float* rstd,
    const float* dscale_sum,
    const float* dbias_sum,
    float* dscale,
    float* dbias,
    float* alpha,
    float* beta,
    float* gamma) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const float dscale_val = __ldg(dscale_sum + c) * batch_scale;
    const float dbias_val = __ldg(dbias_sum + c) * batch_scale;
    const float scale_x_rstd = __ldg(scale + c) * __ldg(rstd + c);
    const float dscale_x_rstd = dscale_val * __ldg(rstd + c);
    dscale[c] = dscale_val;
    dbias[c] = dbias_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] = scale_x_rstd * fmaf(__ldg(mean + c), dscale_x_rstd, -dbias_val) *
        mean_scale;
#else
    const float dscale_val = dscale_sum[c] * batch_scale;
    const float dbias_val = dbias_sum[c] * batch_scale;
    const float scale_x_rstd = scale[c] * rstd[c];
    const float dscale_x_rstd = dscale_val * rstd[c];
    dscale[c] = dscale_val;
    dbias[c] = dbias_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] =
        scale_x_rstd * fmaf(mean[c], dscale_x_rstd, -dbias_val) * mean_scale;
#endif
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
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
      ds_val += __ldg(dY + index) * __ldg(X + index);
      db_val += __ldg(dY + index);
#else
      ds_val += dY[index] * X[index];
      db_val += dY[index];
#endif
    }
  }
  ds_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(ds_storage).Sum(ds_val);
  db_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(db_storage).Sum(db_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    ds_val = (ds_val - __ldg(mean + c) * db_val) * __ldg(rstd + c);
    const T scale_x_rstd = __ldg(scale + c) * __ldg(rstd + c);
    const T dscale_x_rstd = ds_val * __ldg(rstd + c);
    dscale[c] = ds_val;
    dbias[c] = db_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] =
        scale_x_rstd * (__ldg(mean + c) * dscale_x_rstd - db_val) * mean_scale;
#else
    ds_val = (ds_val - mean[c] * db_val) * rstd[c];
    const T scale_x_rstd = scale[c] * rstd[c];
    const T dscale_x_rstd = ds_val * rstd[c];
    dscale[c] = ds_val;
    dbias[c] = db_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] = scale_x_rstd * (mean[c] * dscale_x_rstd - db_val) * mean_scale;
#endif
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
    T* gamma);

template <>
__global__ void ComputeScaleGradientAndFusedParamsNHWCCUDAKernel<float>(
    const int C,
    const float mean_scale,
    const float* dYxX,
    const float* dbias,
    const float* scale,
    const float* mean,
    const float* rstd,
    float* dscale,
    float* alpha,
    float* beta,
    float* gamma) {
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const float ds = fmaf(-__ldg(dbias + c), __ldg(mean + c), __ldg(dYxX + c)) *
        __ldg(rstd + c);
    dscale[c] = ds;
    const float scale_x_rstd = __ldg(scale + c) * __ldg(rstd + c);
    const float dscale_x_rstd = ds * __ldg(rstd + c);
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] = scale_x_rstd *
        fmaf(__ldg(mean + c), dscale_x_rstd, -__ldg(dbias + c)) * mean_scale;
#else
    const float ds = fmaf(-dbias[c], mean[c], dYxX[c]) * rstd[c];
    dscale[c] = ds;
    const float scale_x_rstd = scale[c] * rstd[c];
    const float dscale_x_rstd = ds * rstd[c];
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * mean_scale;
    gamma[c] =
        scale_x_rstd * fmaf(mean[c], dscale_x_rstd, -dbias[c]) * mean_scale;
#endif
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
    T* dX);

template <>
__global__ void ComputeXGradientNCHWCUDAKernel<float>(
    const int C,
    const int M,
    const int HxW,
    const float* dY,
    const float* X,
    const float* alpha,
    const float* beta,
    const float* gamma,
    float* dX) {
  const int nc = blockIdx.x / M;
  const int c = nc % C;
  const int x = blockIdx.x % M * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (x < HxW) {
    const int index = nc * HxW + x;
#if __CUDA_ARCH__ >= 350
    dX[index] = fmaf(
        __ldg(alpha + c),
        __ldg(dY + index),
        fmaf(__ldg(beta + c), __ldg(X + index), __ldg(gamma + c)));
#else
    dX[index] = fmaf(alpha[c], dY[index], fmaf(beta[c], X[index], gamma[c]));
#endif
  }
}

template <typename T>
__global__ void ComputeXGradientNHWCCUDAKernel(
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX);

template <>
__global__ void ComputeXGradientNHWCCUDAKernel(
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    const float* alpha,
    const float* beta,
    const float* gamma,
    float* dX) {
  const int c = blockIdx.y * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const int index = blockIdx.x * C + c;
#if __CUDA_ARCH__ >= 350
    dX[index] = fmaf(
        __ldg(alpha + c),
        __ldg(dY + index),
        fmaf(__ldg(beta + c), __ldg(X + index), __ldg(gamma + c)));
#else
    dX[index] = fmaf(alpha[c], dY[index], fmaf(beta[c], X[index], gamma[c]));
#endif
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
}

template <>
template <typename T>
void SpatialBNOp<CUDAContext>::ComputeRunningMomentsAndFusedParam(
    const int C,
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
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK(
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
  } else {
    const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
    ComputeXGradientNHWCCUDAKernel<T>
        <<<dim3(N * HxW, M),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(C, HxW, dY, X, alpha, beta, gamma, dX);
  }
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_IMPL_CUH_

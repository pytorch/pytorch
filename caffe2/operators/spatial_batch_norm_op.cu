#include "caffe2/operators/spatial_batch_norm_op.h"

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T, int kBlockDimX, int kBlockDimY>
using BlockReduce2D = cub::
    BlockReduce<T, kBlockDimX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, kBlockDimY>;

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
    const float mu = __ldg(batch_mean_sum + c) * scale;
    mean[c] = mu;
    var[c] = fmaf(__ldg(batch_var_sum + c), scale, -mu * mu);
#else
    const float mu = batch_mean_sum[c] * scale;
    mean[c] = mu;
    var[c] = fmaf(batch_var_sum[c], scale, -mu * mu);
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
  const float a = 1.0f - momentum;
  const float b = momentum;
  const int c = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    running_mean[c] = fmaf(a, __ldg(mean + c), b * __ldg(running_mean + c));
    running_var[c] = fmaf(a, __ldg(var + c), b * __ldg(running_var + c));
    const float rstd_val = rsqrtf(__ldg(var + c) + epsilon);
    const float scale_x_rstd = __ldg(scale + c) * rstd_val;
    rstd[c] = rstd_val;
    alpha[c] = scale_x_rstd;
    beta[c] = fmaf(-scale_x_rstd, __ldg(mean + c), __ldg(bias + c));
#else
    running_mean[c] = fmaf(a, mean[c], b * running_mean[c]);
    running_var[c] = fmaf(a, var[c], b * running_var[c]);
    const float rstd_val = rsqrtf(var[c] + epsilon);
    const float scale_x_rstd = scale[c] * rstd_val;
    rstd[c] = rstd_val;
    alpha[c] = scale_x_rstd;
    beta[c] = fmaf(-scale_x_rstd, mean[c], bias[c]);
#endif
  }
}

template <typename T>
__global__ void ComputeMultiBatchScaleBiasGradientsAndFusedParamsCUDAKernel(
    const int C,
    const T inv_num_batches,
    const T inv_nhw,
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
    const float inv_num_batches,
    const float inv_nhw,
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
    const float dscale_val = __ldg(dscale_sum + c) * inv_num_batches;
    const float dbias_val = __ldg(dbias_sum + c) * inv_num_batches;
    const float scale_x_rstd = __ldg(scale + c) * __ldg(rstd + c);
    const float dscale_x_rstd = dscale_val * __ldg(rstd + c);
    dscale[c] = dscale_val;
    dbias[c] = dbias_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[c] = scale_x_rstd * fmaf(__ldg(mean + c), dscale_x_rstd, -dbias_val) *
        inv_nhw;
#else
    const float dscale_val = dscale_sum[c] * inv_num_batches;
    const float dbias_val = dbias_sum[c] * inv_num_batches;
    const float scale_x_rstd = scale[c] * rstd[c];
    const float dscale_x_rstd = dscale_val * rstd[c];
    dscale[c] = dscale_val;
    dbias[c] = dbias_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[c] =
        scale_x_rstd * fmaf(mean[c], dscale_x_rstd, -dbias_val) * inv_nhw;
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
  const T inv_nhw = T(1) / static_cast<T>(N * HxW);
  const int c = blockIdx.x;
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
    ds_val = (ds_val - db_val * __ldg(mean + c)) * __ldg(rstd + c);
    const T scale_x_rstd = __ldg(scale + c) * __ldg(rstd + c);
    const T dscale_x_rstd = ds_val * __ldg(rstd + c);
    dscale[c] = ds_val;
    dbias[c] = db_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[c] =
        scale_x_rstd * (__ldg(mean + c) * dscale_x_rstd - db_val) * inv_nhw;
#else
    ds_val = (ds_val - db_val * mean[c]) * rstd[c];
    const T scale_x_rstd = scale[c] * rstd[c];
    const T dscale_x_rstd = ds_val * rstd[c];
    dscale[c] = ds_val;
    dbias[c] = db_val;
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[c] = scale_x_rstd * (mean[c] * dscale_x_rstd - db_val) * inv_nhw;
#endif
  }
}

template <typename T>
__global__ void ComputeScaleGradientAndFusedParamsNHWCCUDAKernel(
    const int C,
    const T inv_nhw,
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
    const float inv_nhw,
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
    beta[c] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[c] = scale_x_rstd *
        fmaf(__ldg(mean + c), dscale_x_rstd, -__ldg(dbias + c)) * inv_nhw;
#else
    const float ds = fmaf(-dbias[c], mean[c], dYxX[c]) * rstd[c];
    dscale[c] = ds;
    const float scale_x_rstd = scale[c] * rstd[c];
    const float dscale_x_rstd = ds * rstd[c];
    alpha[c] = scale_x_rstd;
    beta[c] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[c] = scale_x_rstd * fmaf(mean[c], dscale_x_rstd, -dbias[c]) * inv_nhw;
#endif
  }
}

template <typename T>
__global__ void ComputeXGradientNCHWCUDAKernel(
    const int C,
    const int HxW,
    const int K,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX);

template <>
__global__ void ComputeXGradientNCHWCUDAKernel<float>(
    const int C,
    const int HxW,
    const int K,
    const float* dY,
    const float* X,
    const float* alpha,
    const float* beta,
    const float* gamma,
    float* dX) {
  const int nc = blockIdx.x / K;
  const int block = blockIdx.x % K;
  const int c = nc % C;
  const int w = block * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (w < HxW) {
    const int index = nc * HxW + w;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dX[index] = fmaf(
        __ldg(alpha + c),
        __ldg(dY + index),
        fmaf(__ldg(beta + c), __ldg(X + index), __ldg(gamma + c)));
#else
    dX[index] = fmaf(alpha[c], dY[index], fma(beta[c], X[index], gamma[c]));
#endif
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
    T* dX);

template <>
__global__ void ComputeXGradientNHWCCUDAKernel<float>(
    const int C,
    const float* dY,
    const float* X,
    const float* alpha,
    const float* beta,
    const float* gamma,
    float* dX) {
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const int index = blockIdx.x * C + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dX[index] = fmaf(
        __ldg(alpha + c),
        __ldg(dY + index),
        fmaf(__ldg(beta + c), __ldg(X + index), __ldg(gamma + c)));
#else
    dX[index] = fmaf(alpha[c], dY[index], fma(beta[c], X[index], gamma[c]));
#endif
  }
}

} // namespace

template <>
template <>
void SpatialBNOp<CUDAContext>::ComputeFusedParam<float>(
    const int C,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* var,
    float* alpha,
    float* beta) {
  const int K = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamCUDAKernel<float>
      <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C, static_cast<float>(epsilon_), scale, bias, mean, var, alpha, beta);
}

template <>
template <>
void SpatialBNOp<CUDAContext>::ComputeBatchMoments<float>(
    const int N,
    const int C,
    const int HxW,
    const float* batch_mean_sum,
    const float* batch_var_sum,
    float* mean,
    float* var) {
  const int K = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  const float scale = 1.0f / static_cast<float>(num_batches_ * N * HxW);
  ComputeBatchMomentsCUDAKernel<float>
      <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C, scale, batch_mean_sum, batch_var_sum, mean, var);
}

template <>
template <>
void SpatialBNOp<CUDAContext>::ComputeRunningMomentsAndFusedParam<float>(
    const int C,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* var,
    float* running_mean,
    float* running_var,
    float* rstd,
    float* alpha,
    float* beta) {
  const int K = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  ComputeRunningMomentsAndFusedParamCUDAKernel<float>
      <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C,
          static_cast<float>(momentum_),
          static_cast<float>(epsilon_),
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
template <>
void SpatialBNGradientOp<CUDAContext>::
    ComputeMultiBatchScaleBiasGradientsAndFusedParams<float>(
        const int N,
        const int C,
        const int HxW,
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
  const float inv_num_batches = 1.0f / static_cast<float>(num_batches_);
  const float inv_nhw = 1.0f / static_cast<float>(N * HxW);
  const int K = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  ComputeMultiBatchScaleBiasGradientsAndFusedParamsCUDAKernel<float>
      <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C,
          inv_num_batches,
          inv_nhw,
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
template <>
void SpatialBNGradientOp<CUDAContext>::ComputeScaleBiasGradientsAndFusedParams<
    float>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    const float* scale,
    const float* mean,
    const float* rstd,
    float* dscale,
    float* dbias,
    float* alpha,
    float* beta,
    float* gamma,
    float* scratch) {
  if (order_ == StorageOrder::NCHW) {
    if (HxW >= 128) {
      ComputeScaleBiasGradientsAndFusedParamsNCHWCUDAKernel<float, 1, 128>
          <<<C, dim3(1, 128), 0, context_.cuda_stream()>>>(
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
    } else if (HxW >= 64) {
      ComputeScaleBiasGradientsAndFusedParamsNCHWCUDAKernel<float, 2, 64>
          <<<C, dim3(2, 64), 0, context_.cuda_stream()>>>(
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
    } else if (HxW >= 32) {
      ComputeScaleBiasGradientsAndFusedParamsNCHWCUDAKernel<float, 4, 32>
          <<<C, dim3(4, 32), 0, context_.cuda_stream()>>>(
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
      ComputeScaleBiasGradientsAndFusedParamsNCHWCUDAKernel<float, 8, 16>
          <<<C, dim3(8, 16), 0, context_.cuda_stream()>>>(
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
    }
  } else {
    ReinitializeTensor(&ones_, {N * HxW}, at::dtype<float>().device(CUDA));
    math::Set<float, CUDAContext>(
        N * HxW, 1.0f, ones_.mutable_data<float>(), &context_);
    const float* ones_data = ones_.data<float>();
    math::Mul<float, CUDAContext>(N * C * HxW, dY, X, scratch, &context_);
    math::Gemm<float, CUDAContext>(
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
    math::Gemm<float, CUDAContext>(
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
    const int K = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
    ComputeScaleGradientAndFusedParamsNHWCCUDAKernel<float>
        <<<K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            C,
            1.0f / static_cast<float>(N * HxW),
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
template <>
void SpatialBNGradientOp<CUDAContext>::ComputeXGradient<float>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    const float* alpha,
    const float* beta,
    const float* gamma,
    float* dX) {
  if (order_ == StorageOrder::NCHW) {
    const int K = math::DivUp(HxW, CAFFE_CUDA_NUM_THREADS);
    ComputeXGradientNCHWCUDAKernel<float>
        <<<N * C * K, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            C, HxW, K, dY, X, alpha, beta, gamma, dX);
  } else {
    ComputeXGradientNHWCCUDAKernel<float>
        <<<N * HxW, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            C, dY, X, alpha, beta, gamma, dX);
  }
}

REGISTER_CUDA_OPERATOR(SpatialBN, SpatialBNOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SpatialBNGradient, SpatialBNGradientOp<CUDAContext>);

} // namespace caffe2

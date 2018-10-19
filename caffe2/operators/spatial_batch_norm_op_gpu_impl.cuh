#include "caffe2/operators/spatial_batch_norm_op.h"

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

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
  CUDA_1D_KERNEL_LOOP(i, C) {
#if __CUDA_ARCH__ >= 350
    const float scale_x_rstd =
        __ldg(scale + i) * rsqrtf(__ldg(var + i) + epsilon);
    alpha[i] = scale_x_rstd;
    beta[i] = __ldg(bias + i) - scale_x_rstd * __ldg(mean + i);
#else
    const float scale_x_rstd = scale[i] * rsqrtf(var[i] + epsilon);
    alpha[i] = scale_x_rstd;
    beta[i] = bias[i] - scale_x_rstd * mean[i];
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
    T* var) {
  CUDA_1D_KERNEL_LOOP(i, C) {
#if __CUDA_ARCH__ >= 350
    const T mu = __ldg(batch_mean_sum + i) * scale;
    mean[i] = mu;
    var[i] = __ldg(batch_var_sum + i) * scale - mu * mu;
#else
    const T mu = batch_mean_sum[i] * scale;
    mean[i] = mu;
    var[i] = batch_var_sum[i] * scale - mu * mu;
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
  CUDA_1D_KERNEL_LOOP(i, C) {
#if __CUDA_ARCH__ >= 350
    running_mean[i] = a * __ldg(mean + i) + b * __ldg(running_mean + i);
    running_var[i] = a * __ldg(var + i) + b * __ldg(running_var + i);
    const float rstd_val = rsqrtf(__ldg(var + i) + epsilon);
    const float scale_x_rstd = __ldg(scale + i) * rstd_val;
    rstd[i] = rstd_val;
    alpha[i] = scale_x_rstd;
    beta[i] = bias[i] - scale_x_rstd * __ldg(mean + i);
#else
    running_mean[i] = a * mean[i] + b * running_mean[i];
    running_var[i] = a * var[i] + b * running_var[i];
    const float rstd_val = rsqrtf(var[i] + epsilon);
    const float scale_x_rstd = scale[i] * rstd_val;
    rstd[i] = rstd_val;
    alpha[i] = scale_x_rstd;
    beta[i] = bias[i] - scale_x_rstd * mean[i];
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
    T* gamma) {
  CUDA_1D_KERNEL_LOOP(i, C) {
#if __CUDA_ARCH__ >= 350
    const T dscale_val = __ldg(dscale_sum + i) * inv_num_batches;
    const T dbias_val = __ldg(dbias_sum + i) * inv_num_batches;
    const T scale_x_rstd = __ldg(scale + i) * __ldg(rstd + i);
    const T dscale_x_rstd = dscale_val * __ldg(rstd + i);
    dscale[i] = dscale_val;
    dbias[i] = dbias_val;
    alpha[i] = scale_x_rstd;
    beta[i] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[i] =
        scale_x_rstd * (__ldg(mean + i) * dscale_x_rstd - dbias_val) * inv_nhw;
#else
    const T dscale_val = dscale_sum[i] * inv_num_batches;
    const T dbias_val = dbias_sum[i] * inv_num_batches;
    const T scale_x_rstd = scale[i] * rstd[i];
    const T dscale_x_rstd = dscale_val * rstd[i];
    dscale[i] = dscale_val;
    dbias[i] = dbias_val;
    alpha[i] = scale_x_rstd;
    beta[i] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
    gamma[i] = scale_x_rstd * (mean[i] * dscale_x_rstd - dbias_val) * inv_nhw;
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void ComputeScaleBiasGradientsAndFusedParamsCUDAKernel(
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
  const int outer_size = C;
  const int inner_size = N * HxW;
  const T inv_nhw = T(1) / static_cast<T>(N * HxW);
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
#if __CUDA_ARCH__ >= 350
    const T scale_val = __ldg(scale + i);
    const T mean_val = __ldg(mean + i);
    const T rstd_val = __ldg(rstd + i);
#else
    const T scale_val = scale[i];
    const T mean_val = mean[i];
    const T rstd_val = rstd[i];
#endif
    T ds_val = 0;
    T db_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = kOrder == StorageOrder::NCHW
          ? (j / HxW * C + i) * HxW + j % HxW
          : j * C + i;
#if __CUDA_ARCH__ >= 350
      ds_val += __ldg(dY + index) * (__ldg(X + index) - mean_val) * rstd_val;
      db_val += __ldg(dY + index);
#else
      ds_val += dY[index] * (X[index] - mean_val) * rstd_val;
      db_val += dY[index];
#endif
    }
    ds_val = BlockReduce<T>(ds_storage).Sum(ds_val);
    db_val = BlockReduce<T>(db_storage).Sum(db_val);
    if (threadIdx.x == 0) {
      const T scale_x_rstd = scale_val * rstd_val;
      const T dscale_x_rstd = ds_val * rstd_val;
      dscale[i] = ds_val;
      dbias[i] = db_val;
      alpha[i] = scale_x_rstd;
      beta[i] = -scale_x_rstd * dscale_x_rstd * inv_nhw;
      gamma[i] = scale_x_rstd * (mean_val * dscale_x_rstd - db_val) * inv_nhw;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void ComputeXGradientNCHWCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX) {
  const int outer_size = N * C;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    const int c = i % C;
#if __CUDA_ARCH__ >= 350
    const T alpha_val = __ldg(alpha + c);
    const T beta_val = __ldg(beta + c);
    const T gamma_val = __ldg(gamma + c);
#else
    const T alpha_val = alpha[c];
    const T beta_val = beta[c];
    const T gamma_val = gamma[c];
#endif
    for (int j = threadIdx.x; j < HxW; j += blockDim.x) {
      const int index = i * HxW + j;
#if __CUDA_ARCH__ >= 350
      dX[index] = __ldg(dY + index) * alpha_val + __ldg(X + index) * beta_val +
          gamma_val;
#else
      dX[index] = dY[index] * alpha_val + X[index] * beta_val + gamma_val;
#endif
    }
  }
}

template <typename T>
__global__ void ComputeXGradientNHWCCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* alpha,
    const T* beta,
    const T* gamma,
    T* dX) {
  const int outer_size = N * HxW;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
      const int index = i * C + j;
#if __CUDA_ARCH__ >= 350
      dX[index] = __ldg(dY + index) * __ldg(alpha + j) +
          __ldg(X + index) * __ldg(beta + j) + __ldg(gamma + j);
#else
      dX[index] = dY[index] * alpha[j] + X[index] * beta[j] + gamma[j];
#endif
    }
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
  ComputeFusedParamCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(C),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
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
  const T scale = T(1) / static_cast<T>(num_batches_ * N * HxW);
  ComputeBatchMomentsCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(C),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
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
  ComputeRunningMomentsAndFusedParamCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(C),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
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
  const T inv_num_batches = T(1) / static_cast<T>(num_batches_);
  const T inv_nhw = T(1) / static_cast<T>(N * HxW);
  ComputeMultiBatchScaleBiasGradientsAndFusedParamsCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(C),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
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
    T* gamma) {
  if (order_ == StorageOrder::NCHW) {
    ComputeScaleBiasGradientsAndFusedParamsCUDAKernel<T, StorageOrder::NCHW>
        <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
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
    ComputeScaleBiasGradientsAndFusedParamsCUDAKernel<T, StorageOrder::NHWC>
        <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
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
    ComputeXGradientNCHWCUDAKernel<T>
        <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(N, C, HxW, dY, X, alpha, beta, gamma, dX);
  } else {
    ComputeXGradientNHWCCUDAKernel<T>
        <<<std::min(N * HxW, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(N, C, HxW, dY, X, alpha, beta, gamma, dX);
  }
}

} // namespace caffe2

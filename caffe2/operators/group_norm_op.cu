// ------------------------------------------------------------------
// GroupNorm op in Caffe2 for GPU
// Written by Kaiming He
// Improved by Xiaomeng Yang
// see https://arxiv.org/abs/1803.08494
// This is a stand-alone op: Y = gamma * (X - mu) / sig + beta
// ------------------------------------------------------------------

#include "caffe2/operators/group_norm_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/reduce.cuh"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    const int N,
    const int G,
    const int K,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias);

template <>
__global__ void ComputeFusedParamsCUDAKernel<float>(
    const int N,
    const int G,
    const int K,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float* beta,
    float* scale,
    float* bias) {
  const int C = G * K;
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C) {
    const int ng = index / K;
    const int c = index % C;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const float scale_val = __ldg(gamma + c) * __ldg(rsig + ng);
    scale[index] = scale_val;
    bias[index] = fmaf(-scale_val, __ldg(mu + ng), __ldg(beta + c));
#else
    const float scale_val = gamma[c] * rsig[ng];
    scale[index] = scale_val;
    bias[index] = fmaf(-scale_val, mu[ng], beta[c]);
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void GroupNormForwardCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <>
__global__ void GroupNormForwardCUDAKernel<float, StorageOrder::NCHW>(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C * HxW) {
    const int nc = index / HxW;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[index] = fmaf(__ldg(X + index), __ldg(scale + nc), __ldg(bias + nc));
#else
    Y[index] = fmaf(X[index], scale[nc], bias[nc]);
#endif
  }
}

template <>
__global__ void GroupNormForwardCUDAKernel<float, StorageOrder::NHWC>(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C * HxW) {
    const int nc = index / (HxW * C) * C + index % C;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[index] = fmaf(__ldg(X + index), __ldg(scale + nc), __ldg(bias + nc));
#else
    Y[index] = fmaf(X[index], scale[nc], bias[nc]);
#endif
  }
}

template <typename T>
__global__ void ComputeInternalGradientsNCHWCUDAKernel(
    const int HxW,
    const T* dY,
    const T* X,
    T* ds,
    T* db) {
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  const int nc = blockIdx.x;
  T ds_sum = 0;
  T db_sum = 0;
  for (int i = threadIdx.x; i < HxW; i += blockDim.x) {
    const int index = nc * HxW + i;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    ds_sum += __ldg(dY + index) * __ldg(X + index);
    db_sum += __ldg(dY + index);
#else
    ds_sum += dY[index] * X[index];
    db_sum += dY[index];
#endif
  }
  ds_sum = BlockReduce<T>(ds_storage).Sum(ds_sum);
  db_sum = BlockReduce<T>(db_storage).Sum(db_sum);
  if (threadIdx.x == 0) {
    ds[nc] = ds_sum;
    db[nc] = db_sum;
  }
}

// Math:
// Y = gamma * (X - mu) * rsig + beta
// let s = gamma * rsig
// let b = beta - gamma * mu * rsig
// Y = s * X + b
// let n = K * HxW
// dL/dX = dL/dY * dY/dX = dL/dY * (d(s * X)/dX + db/dX)
// d(s * X)/dX = s + X * ds/dX = s + gamma * X * drsig/dX
// db/dX = -gamma * u * drsig/dX - gamma * rsig * dmu/dX
// drsig/dX = -rsig^3 * (X - mu) / n
// dmu/dX = 1 / n
template <typename T>
__global__ void ComputeYGradientScaleCUDAKernel(
    const int N,
    const int G,
    const int K,
    const T* rsig,
    const T* gamma,
    T* dY_scale) {
  const int C = G * K;
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C) {
    const int ng = index / K;
    const int c = index % C;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dY_scale[index] = __ldg(gamma + c) * __ldg(rsig + ng);
#else
    dY_scale[index] = gamma[c] * rsig[ng];
#endif
  }
}

template <typename T>
__global__ void ComputeXScaleAndBiasCUDAKernel(
    const int G,
    const int K,
    const T alpha,
    const T* ds,
    const T* db,
    const T* mu,
    const T* rsig,
    const T* gamma,
    T* X_scale,
    T* bias);

template <>
__global__ void ComputeXScaleAndBiasCUDAKernel<float>(
    const int G,
    const int K,
    const float alpha,
    const float* ds,
    const float* db,
    const float* mu,
    const float* rsig,
    const float* gamma,
    float* X_scale,
    float* bias) {
  __shared__ typename BlockReduce<float>::TempStorage ds_storage;
  __shared__ typename BlockReduce<float>::TempStorage db_storage;
  const int n = blockIdx.x;
  const int g = blockIdx.y;
  const int ng = n * G + g;
  float ds_sum = 0;
  float db_sum = 0;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    const int index = ng * K + i;
    const int c = g * K + i;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    ds_sum += __ldg(ds + index) * __ldg(gamma + c);
    db_sum += __ldg(db + index) * __ldg(gamma + c);
#else
    ds_sum += ds[index] * gamma[c];
    db_sum += db[index] * gamma[c];
#endif
  }
  ds_sum = BlockReduce<float>(ds_storage).Sum(ds_sum);
  db_sum = BlockReduce<float>(db_storage).Sum(db_sum);
  if (threadIdx.x == 0) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const float x = fmaf(db_sum, __ldg(mu + ng), -ds_sum) *
        math::utils::Cube<float>(__ldg(rsig + ng)) * alpha;
    X_scale[ng] = x;
    bias[ng] = -fmaf(x, __ldg(mu + ng), db_sum * __ldg(rsig + ng) * alpha);
#else
    const float x = fmaf(db_sum, mu[ng], -ds_sum) *
        math::utils::Cube<float>(rsig[ng]) * alpha;
    X_scale[ng] = x;
    bias[ng] = -fmaf(x, mu[ng], db_sum * rsig[ng] * alpha);
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void GroupNormBackwardCUDAKernel(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* dY_scale,
    const T* dY,
    const T* X_scale,
    const T* X,
    const T* bias,
    T* dX);

template <>
__global__ void GroupNormBackwardCUDAKernel<float, StorageOrder::NCHW>(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* dY_scale,
    const float* dY,
    const float* X_scale,
    const float* X,
    const float* bias,
    float* dX) {
  const int C = G * K;
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C * HxW) {
    const int nc = index / HxW;
    const int ng = nc / K;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dX[index] = fmaf(
        __ldg(dY_scale + nc),
        __ldg(dY + index),
        fmaf(__ldg(X_scale + ng), __ldg(X + index), __ldg(bias + ng)));
#else
    dX[index] =
        fmaf(dY_scale[nc], dY[index], fmaf(X_scale[ng], X[index], bias[ng]));
#endif
  }
}

template <>
__global__ void GroupNormBackwardCUDAKernel<float, StorageOrder::NHWC>(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* dY_scale,
    const float* dY,
    const float* X_scale,
    const float* X,
    const float* bias,
    float* dX) {
  const int C = G * K;
  const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C * HxW) {
    const int nc = index / (HxW * C) * C + index % C;
    const int ng = nc / K;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dX[index] = fmaf(
        __ldg(dY_scale + nc),
        __ldg(dY + index),
        fmaf(__ldg(X_scale + ng), __ldg(X + index), __ldg(bias + ng)));
#else
    dX[index] =
        fmaf(dY_scale[nc], dY[index], fmaf(X_scale[ng], X[index], bias[ng]));
#endif
  }
}

template <typename T>
__global__ void GammaBetaBackwardCUDAKernel(
    const int N,
    const int G,
    const int K,
    const T* ds,
    const T* db,
    const T* mu,
    const T* rsig,
    T* dgamma,
    T* dbeta);

template <>
__global__ void GammaBetaBackwardCUDAKernel<float>(
    const int N,
    const int G,
    const int K,
    const float* ds,
    const float* db,
    const float* mu,
    const float* rsig,
    float* dgamma,
    float* dbeta) {
  __shared__ typename BlockReduce<float>::TempStorage dg_storage;
  __shared__ typename BlockReduce<float>::TempStorage db_storage;
  const int C = G * K;
  const int g = blockIdx.x;
  const int k = blockIdx.y;
  const int c = g * K + k;
  float dg_sum = 0;
  float db_sum = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    const int nc = i * C + c;
    const int ng = i * G + g;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dg_sum += fmaf(-__ldg(db + nc), __ldg(mu + ng), __ldg(ds + nc)) *
        __ldg(rsig + ng);
    db_sum += __ldg(db + nc);
#else
    dg_sum += fmaf(-db[nc], mu[ng], ds[nc]) * rsig[ng];
    db_sum += db[nc];
#endif
  }
  dg_sum = BlockReduce<float>(dg_storage).Sum(dg_sum);
  db_sum = BlockReduce<float>(db_storage).Sum(db_sum);
  if (threadIdx.x == 0) {
    dgamma[c] = dg_sum;
    dbeta[c] = db_sum;
  }
}

} // namespace

template <>
void GroupNormOp<float, CUDAContext>::ComputeFusedParams(
    const int N,
    const int G,
    const int K,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float* beta,
    float* scale,
    float* bias) {
  const int M = math::DivUp(N * G * K, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamsCUDAKernel<float>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, G, K, mu, rsig, gamma, beta, scale, bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
void GroupNormOp<float, CUDAContext>::GroupNormForwardNCHW(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int M = math::DivUp(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  GroupNormForwardCUDAKernel<float, StorageOrder::NCHW>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, X, scale, bias, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
void GroupNormOp<float, CUDAContext>::GroupNormForwardNHWC(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int M = math::DivUp(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  GroupNormForwardCUDAKernel<float, StorageOrder::NHWC>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, X, scale, bias, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Math:
// let: s = gamma * rsig
// let: b = beta - mu * gamma * rsig
// then: Y = s * X + b
template <>
bool GroupNormGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* dY_data,
    const float* X_data,
    const float* mu_data,
    const float* rsig_data,
    const float* gamma_data,
    float* dX_data,
    float* dgamma_data,
    float* dbeta_data) {
  const int C = G * K;
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CUDA));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  float* dY_scale_data = dY_scale_.mutable_data<float>();
  float* X_scale_data = X_scale_.mutable_data<float>();
  float* bias_data = bias_.mutable_data<float>();

  ComputeInternalGradientsNCHWCUDAKernel<float>
      <<<N * C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          HxW, dY_data, X_data, ds_data, db_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Computes dL/dX.
  int M = math::DivUp(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeYGradientScaleCUDAKernel<float>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, G, K, rsig_data, gamma_data, dY_scale_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ComputeXScaleAndBiasCUDAKernel<float>
      <<<dim3(N, G), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          G,
          K,
          1.0f / static_cast<float>(K * HxW),
          ds_data,
          db_data,
          mu_data,
          rsig_data,
          gamma_data,
          X_scale_data,
          bias_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  M = math::DivUp(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  GroupNormBackwardCUDAKernel<float, StorageOrder::NCHW>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          G,
          K,
          HxW,
          dY_scale_data,
          dY_data,
          X_scale_data,
          X_data,
          bias_data,
          dX_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Computes dL/dgamma and dL/dbeta.
  GammaBetaBackwardCUDAKernel<
      float><<<dim3(G, K), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool GroupNormGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* dY_data,
    const float* X_data,
    const float* mu_data,
    const float* rsig_data,
    const float* gamma_data,
    float* dX_data,
    float* dgamma_data,
    float* dbeta_data) {
  const int C = G * K;
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&ones_, {HxW}, at::dtype<float>().device(CUDA));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  float* dY_scale_data = dY_scale_.mutable_data<float>();
  float* X_scale_data = X_scale_.mutable_data<float>();
  float* bias_data = bias_.mutable_data<float>();
  float* ones_data = ones_.mutable_data<float>();

  math::Set<float, CUDAContext>(HxW, 1.0f, ones_data, &context_);
  math::Mul<float, CUDAContext>(
      N * C * HxW, dY_data, X_data, dX_data, &context_);
  math::GemmStridedBatched<float, CUDAContext>(
      CblasTrans,
      CblasNoTrans,
      N,
      C,
      1,
      HxW,
      1.0f,
      dX_data,
      C * HxW,
      ones_data,
      0,
      0.0f,
      ds_data,
      C,
      &context_);
  math::GemmStridedBatched<float, CUDAContext>(
      CblasTrans,
      CblasNoTrans,
      N,
      C,
      1,
      HxW,
      1.0f,
      dY_data,
      C * HxW,
      ones_data,
      0,
      0.0f,
      db_data,
      C,
      &context_);

  // Computes dL/dX.
  int M = math::DivUp(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeYGradientScaleCUDAKernel<float>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, G, K, rsig_data, gamma_data, dY_scale_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ComputeXScaleAndBiasCUDAKernel<float>
      <<<dim3(N, G), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          G,
          K,
          1.0f / static_cast<float>(K * HxW),
          ds_data,
          db_data,
          mu_data,
          rsig_data,
          gamma_data,
          X_scale_data,
          bias_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  M = math::DivUp(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  GroupNormBackwardCUDAKernel<float, StorageOrder::NHWC>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          G,
          K,
          HxW,
          dY_scale_data,
          dY_data,
          X_scale_data,
          X_data,
          bias_data,
          dX_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Computes dL/dgamma and dL/dbeta.
  GammaBetaBackwardCUDAKernel<
      float><<<dim3(G, K), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(GroupNorm, GroupNormOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    GroupNormGradient,
    GroupNormGradientOp<float, CUDAContext>);

} // namespace caffe2

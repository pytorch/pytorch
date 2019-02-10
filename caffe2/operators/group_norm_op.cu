// ------------------------------------------------------------------
// GroupNorm op in Caffe2 for GPU
// Written by Kaiming He
// Improved by Xiaomeng Yang
// see https://arxiv.org/abs/1803.08494
// This is a stand-alone op: Y = gamma * (X - mu) / sig + beta
// ------------------------------------------------------------------

#include "caffe2/operators/group_norm_op.h"

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/reduce.cuh"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    const int G,
    const int K,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  const int n = blockIdx.x;
  const int g = blockIdx.y;
  const int i_mu = n * G + g;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    const int index = i_mu * K + i;
    const int i_gamma = g * K + i;
#if __CUDA_ARCH__ >= 350
    const T scale_val = __ldg(gamma + i_gamma) * __ldg(rsig + i_mu);
    scale[index] = scale_val;
    bias[index] = __ldg(beta + i_gamma) - scale_val * __ldg(mu + i_mu);
#else
    const T scale_val = gamma[i_gamma] * rsig[i_mu];
    scale[index] = scale_val;
    bias[index] = beta[i_gamma] - scale_val * mu[i_mu];
#endif
  }
}

template <typename T>
__global__ void GroupNormForwardNCHWCUDAKernel(
    const int M,
    const int HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <>
__global__ void GroupNormForwardNCHWCUDAKernel<float>(
    const int M,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int nc = blockIdx.x / M;
  const int hw = blockIdx.x % M * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (hw < HxW) {
    const int index = nc * HxW + hw;
#if __CUDA_ARCH__ >= 350
    Y[index] = fmaf(__ldg(X + index), __ldg(scale + nc), __ldg(bias + nc));
#else
    Y[index] = fmaf(X[index], scale[nc], bias[nc]);
#endif
  }
}

template <typename T>
__global__ void GroupNormForwardNHWCCUDAKernel(
    const int C,
    const int HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <>
__global__ void GroupNormForwardNHWCCUDAKernel<float>(
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int n = blockIdx.x / HxW;
  const int c = blockIdx.y * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const int index = blockIdx.x * C + c;
    const int nc = n * C + c;
#if __CUDA_ARCH__ >= 350
    Y[index] = fmaf(__ldg(X + index), __ldg(scale + nc), __ldg(bias + nc));
#else
    Y[index] = fmaf(X[index], scale[nc], bias[nc]);
#endif
  }
}

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void ComputeInternalGradientsNCHWCUDAKernel(
    const int G,
    const int K,
    const int HxW,
    const T* dY,
    const T* X,
    const T* gamma,
    T* ds,
    T* db) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage ds_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage db_storage;
  const int n = blockIdx.x;
  const int g = blockIdx.y;
  const int ng = n * G + g;
  T ds_val = 0;
  T db_val = 0;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    const int c = g * K + i;
    for (int j = threadIdx.y; j < HxW; j += blockDim.y) {
      const int index = (ng * K + i) * HxW + j;
#if __CUDA_ARCH__ >= 350
      ds_val += __ldg(gamma + c) * __ldg(dY + index) * __ldg(X + index);
      db_val += __ldg(gamma + c) * __ldg(dY + index);
#else
      ds_val += gamma[c] * dY[index] * X[index];
      db_val += gamma[c] * dY[index];
#endif
    }
  }
  ds_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(ds_storage).Sum(ds_val);
  db_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(db_storage).Sum(db_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    ds[ng] = ds_val;
    db[ng] = db_val;
  }
}

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void ComputeInternalGradientsNHWCCUDAKernel(
    const int G,
    const int K,
    const int HxW,
    const T* dY,
    const T* X,
    const T* gamma,
    T* ds,
    T* db) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage ds_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage db_storage;
  const int C = G * K;
  const int n = blockIdx.x;
  const int g = blockIdx.y;
  const int ng = n * G + g;
  T ds_val = 0;
  T db_val = 0;
  for (int i = threadIdx.x; i < HxW; i += blockDim.x) {
    for (int j = threadIdx.y; j < K; j += blockDim.y) {
      const int c = g * K + j;
      const int index = (n * HxW + i) * C + c;
#if __CUDA_ARCH__ >= 350
      ds_val += __ldg(gamma + c) * __ldg(dY + index) * __ldg(X + index);
      db_val += __ldg(gamma + c) * __ldg(dY + index);
#else
      ds_val += gamma[c] * dY[index] * X[index];
      db_val += gamma[c] * dY[index];
#endif
    }
  }
  ds_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(ds_storage).Sum(ds_val);
  db_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(db_storage).Sum(db_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    ds[ng] = ds_val;
    db[ng] = db_val;
  }
}

// Math:
// Y = gamma * (X - mu) * rsig + beta
// let s = gamma * rsig
// let b = beta - mu * rsig
// Y = s * X + b
// let n = D * HxW
// dL/dX = dL/dY * dY/dX = dL/dY * (d(s * X)/dX + db/dX)
// d(s * X)/dX = s + X * ds/dX = s + gamma * X * drsig/dX
// db/dX = -u * drsig/dX - rsig * dmu/dX
// drsig/dX = -rsig^3 * (X - mu) / n
// dmu/dX = 1 / n
template <typename T>
__global__ void GroupNormBackwardNCHWCUDAKernel(
    const int G,
    const int K,
    const int M,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* ds,
    const T* db,
    T* dX) {
  const int C = G * K;
  const T denom = T(1) / static_cast<T>(K * HxW);
  const int nc = blockIdx.x / M;
  const int n = nc / C;
  const int c = nc % C;
  const int g = c / K;
  const int ng = n * G + g;
  const int hw = blockIdx.x % M * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  const int index = nc * HxW + hw;
  if (hw < HxW) {
#if __CUDA_ARCH__ >= 350
    const T u = (__ldg(db + ng) * __ldg(mu + ng) - __ldg(ds + ng)) *
        (__ldg(X + index) - __ldg(mu + ng)) *
        math::utils::Cube<T>(__ldg(rsig + ng));
    const T v = __ldg(db + ng) * __ldg(rsig + ng);
    dX[index] = __ldg(gamma + c) * __ldg(dY + index) * __ldg(rsig + ng) +
        (u - v) * denom;
#else
    const T u = (db[ng] * mu[ng] - ds[ng]) * (X[index] - mu[ng]) *
        math::utils::Cube<T>(rsig[ng]);
    const T v = db[ng] * rsig[ng];
    dX[index] = gamma[c] * dY[index] * rsig[ng] + (u - v) * denom;
#endif
  }
}

template <typename T>
__global__ void GroupNormBackwardNHWCCUDAKernel(
    const int G,
    const int K,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* ds,
    const T* db,
    T* dX) {
  const int C = G * K;
  const T denom = T(1) / static_cast<T>(K * HxW);
  const int x = blockIdx.x;
  const int g = blockIdx.y;
  const int n = x / HxW;
  const int ng = n * G + g;
  const int i = blockIdx.z * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < K) {
    const int c = g * K + i;
    const int index = x * C + c;
#if __CUDA_ARCH__ >= 350
    const T u = (__ldg(db + ng) * __ldg(mu + ng) - __ldg(ds + ng)) *
        (__ldg(X + index) - __ldg(mu + ng)) *
        math::utils::Cube<T>(__ldg(rsig + ng));
    const T v = __ldg(db + ng) * __ldg(rsig + ng);
    dX[index] = __ldg(gamma + c) * __ldg(dY + index) * __ldg(rsig + ng) +
        (u - v) * denom;
#else
    const T u = (db[ng] * mu[ng] - ds[ng]) * (X[index] - mu[ng]) *
        math::utils::Cube<T>(rsig[ng]);
    const T v = db[ng] * rsig[ng];
    dX[index] = gamma[c] * dY[index] * rsig[ng] + (u - v) * denom;
#endif
  }
}

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void GammaBetaBackwardNCHWCUDAKernel(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mu,
    const T* rsig,
    T* dgamma,
    T* dbeta) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage dg_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage db_storage;
  const int C = G * K;
  const int c = blockIdx.x;
  const int g = c / K;
  T dg_val = 0;
  T db_val = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    for (int j = threadIdx.y; j < HxW; j += blockDim.y) {
      const int index = (i * C + c) * HxW + j;
      const int ng = i * G + g;
#if __CUDA_ARCH__ >= 350
      dg_val += __ldg(dY + index) * (__ldg(X + index) - __ldg(mu + ng)) *
          __ldg(rsig + ng);
      db_val += __ldg(dY + index);
#else
      dg_val += dY[index] * (X[index] - mu[ng]) * rsig[ng];
      db_val += dY[index];
#endif
    }
  }
  dg_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(dg_storage).Sum(dg_val);
  db_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(db_storage).Sum(db_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    dgamma[c] = dg_val;
    dbeta[c] = db_val;
  }
}

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void GammaBetaBackwardNHWCCUDAKernel(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mu,
    const T* rsig,
    T* dgamma,
    T* dbeta) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage dg_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage db_storage;
  const int C = G * K;
  const int c = blockIdx.x;
  const int g = c / K;
  T dg_val = 0;
  T db_val = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    for (int j = threadIdx.y; j < HxW; j += blockDim.y) {
      const int index = (i * HxW + j) * C + c;
      const int ng = i * G + g;
#if __CUDA_ARCH__ >= 350
      dg_val += __ldg(dY + index) * (__ldg(X + index) - __ldg(mu + ng)) *
          __ldg(rsig + ng);
      db_val += __ldg(dY + index);
#else
      dg_val += dY[index] * (X[index] - mu[ng]) * rsig[ng];
      db_val += dY[index];
#endif
    }
  }
  dg_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(dg_storage).Sum(dg_val);
  db_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(db_storage).Sum(db_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    dgamma[c] = dg_val;
    dbeta[c] = db_val;
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
  ComputeFusedParamsCUDAKernel<float>
      <<<dim3(N, G), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          G, K, mu, rsig, gamma, beta, scale, bias);
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
  const int M = math::DivUp(HxW, CAFFE_CUDA_NUM_THREADS);
  GroupNormForwardNCHWCUDAKernel<float>
      <<<N * C * M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          M, HxW, X, scale, bias, Y);
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
  const int M = math::DivUp(C, CAFFE_CUDA_NUM_THREADS);
  GroupNormForwardNHWCCUDAKernel<float>
      <<<dim3(N * HxW, M), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          C, HxW, X, scale, bias, Y);
}

// Math:
// let: s = gamma * rsig
// let: b = beta - mu * gamma * rsig
// then: Y = s * X + b
template <>
bool GroupNormGradientOp<float, CUDAContext>::RunOnDeviceImpl(
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
  ReinitializeTensor(&ds_, {N, G}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&db_, {N, G}, at::dtype<float>().device(CUDA));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  if (order_ == StorageOrder::NCHW) {
    // Computes dL/ds and dL/db.
    // dL/ds = Sum(dL/dY * gamma * X)
    // dL/db = Sum(dL/dY * gamma)
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK(
        HxW,
        ComputeInternalGradientsNCHWCUDAKernel,
        float,
        dim3(N, G),
        context_.cuda_stream(),
        G,
        K,
        HxW,
        dY_data,
        X_data,
        gamma_data,
        ds_data,
        db_data);

    // Computes dL/dX.
    const int M = math::DivUp(HxW, CAFFE_CUDA_NUM_THREADS);
    GroupNormBackwardNCHWCUDAKernel<float>
        <<<N * C * M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            G,
            K,
            M,
            HxW,
            dY_data,
            X_data,
            mu_data,
            rsig_data,
            gamma_data,
            ds_data,
            db_data,
            dX_data);

    // Computes dL/dgamma and dL/dbeta.
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK(
        HxW,
        GammaBetaBackwardNCHWCUDAKernel,
        float,
        C,
        context_.cuda_stream(),
        N,
        G,
        K,
        HxW,
        dY_data,
        X_data,
        mu_data,
        rsig_data,
        dgamma_data,
        dbeta_data);
  } else {
    // Computes dL/ds and dL/db.
    // dL/ds = Sum(dL/dY * gamma * X)
    // dL/db = Sum(dL/dY * gamma)
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK(
        K,
        ComputeInternalGradientsNHWCCUDAKernel,
        float,
        dim3(N, G),
        context_.cuda_stream(),
        G,
        K,
        HxW,
        dY_data,
        X_data,
        gamma_data,
        ds_data,
        db_data);

    // Computes dL/dX.
    const int M = math::DivUp(K, CAFFE_CUDA_NUM_THREADS);
    GroupNormBackwardNHWCCUDAKernel<float>
        <<<dim3(N * HxW, G, M),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            G,
            K,
            HxW,
            dY_data,
            X_data,
            mu_data,
            rsig_data,
            gamma_data,
            ds_data,
            db_data,
            dX_data);

    // Computes dL/dgamma and dL/dbeta.
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK(
        HxW,
        GammaBetaBackwardNHWCCUDAKernel,
        float,
        C,
        context_.cuda_stream(),
        N,
        G,
        K,
        HxW,
        dY_data,
        X_data,
        mu_data,
        rsig_data,
        dgamma_data,
        dbeta_data);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(GroupNorm, GroupNormOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    GroupNormGradient,
    GroupNormGradientOp<float, CUDAContext>);

} // namespace caffe2

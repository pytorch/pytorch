// ------------------------------------------------------------------
// GroupNorm op in Caffe2 for GPU
// Written by Kaiming He
// Improved by Xiaomeng Yang
// see https://arxiv.org/abs/1803.08494
// This is a stand-alone op: Y = gamma * (X - mu) / sig + beta
// ------------------------------------------------------------------

#include "group_norm_op.h"

#include <array>

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T>
inline __device__ T CubeCUDA(const T x) {
  return x * x * x;
}

__global__ void InvStdCUDAKernel(
    const int size,
    const float epsilon,
    const float* var,
    float* rsig) {
  CUDA_1D_KERNEL_LOOP(i, size) {
#if __CUDA_ARCH__ >= 350
    rsig[i] = rsqrtf(__ldg(var + i) + epsilon);
#else
    rsig[i] = rsqrtf(var[i] + epsilon);
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void GroupNormForwardCUDAKernel(
    const int size,
    const int G,
    const int D,
    const int HxW,
    const T* X,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* Y) {
  const int C = G * D;
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int i_mu = kOrder == StorageOrder::NCHW
        ? i / (D * HxW)
        : i / (C * HxW) * G + (i / D % G);
    const int i_gamma = kOrder == StorageOrder::NCHW ? (i / HxW) % C : i % C;
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(gamma + i_gamma) * (__ldg(X + i) - __ldg(mu + i_mu)) *
            __ldg(rsig + i_mu) +
        __ldg(beta + i_gamma);
#else
    Y[i] = gamma[i_gamma] * (X[i] - mu[i_mu]) * rsig[i_mu] + beta[i_gamma];
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void ComputeInternalGradientsCUDAKernel(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const T* dY,
    const T* X,
    const T* gamma,
    T* ds,
    T* db) {
  const int outer_size = N * G;
  const int inner_size = D * HxW;
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_val = 0;
    T db_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int i_gamma = i % G * D + j / HxW;
      const int index = kOrder == StorageOrder::NCHW
          ? i * inner_size + j
          : (i / G * HxW + j % HxW) * G * D + i_gamma;
#if __CUDA_ARCH__ >= 350
      ds_val += __ldg(gamma + i_gamma) * __ldg(dY + index) * __ldg(X + index);
      db_val += __ldg(gamma + i_gamma) * __ldg(dY + index);
#else
      ds_val += gamma[i_gamma] * dY[index] * X[index];
      db_val += gamma[i_gamma] * dY[index];
#endif
    }
    ds_val = BlockReduce<T>(ds_storage).Reduce(ds_val, cub::Sum());
    db_val = BlockReduce<T>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      ds[i] = ds_val;
      db[i] = db_val;
    }
    __syncthreads();
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
template <typename T, StorageOrder kOrder>
__global__ void GroupNormBackwardCUDAKernel(
    const int size,
    const int G,
    const int D,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* ds,
    const T* db,
    T* dX) {
  const int C = G * D;
  const T denom = T(1) / static_cast<T>(D * HxW);
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int i_mu = kOrder == StorageOrder::NCHW
        ? i / (D * HxW)
        : i / (C * HxW) * G + (i / D % G);
    const int i_gamma = kOrder == StorageOrder::NCHW ? (i / HxW) % C : i % C;
#if __CUDA_ARCH__ >= 350
    const T u = (__ldg(db + i_mu) * __ldg(mu + i_mu) - __ldg(ds + i_mu)) *
        (__ldg(X + i) - __ldg(mu + i_mu)) * CubeCUDA(__ldg(rsig + i_mu));
    const T v = __ldg(db + i_mu) * __ldg(rsig + i_mu);
    dX[i] = __ldg(gamma + i_gamma) * __ldg(dY + i) * __ldg(rsig + i_mu) +
        (u - v) * denom;
#else
    const T u = (db[i_mu] * mu[i_mu] - ds[i_mu]) * (X[i] - mu[i_mu]) *
        CubeCUDA(rsig[i_mu]);
    const T v = db[i_mu] * rsig[i_mu];
    dX[i] = gamma[i_gamma] * dY[i] * rsig[i_mu] + (u - v) * denom;
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void GammaBetaBackwardCUDAKernel(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mu,
    const T* rsig,
    T* dgamma,
    T* dbeta) {
  const int outer_size = G * D;
  const int inner_size = N * HxW;
  __shared__ typename BlockReduce<T>::TempStorage dg_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T dg_val = 0;
    T db_val = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int n = j / HxW;
      const int index = kOrder == StorageOrder::NCHW
          ? (n * outer_size + i) * HxW + j % HxW
          : j * outer_size + i;
      const int i_mu = n * G + i / D;
#if __CUDA_ARCH__ >= 350
      dg_val += __ldg(dY + index) * (__ldg(X + index) - __ldg(mu + i_mu)) *
          __ldg(rsig + i_mu);
      db_val += __ldg(dY + index);
#else
      dg_val += dY[index] * (X[index] - mu[i_mu]) * rsig[i_mu];
      db_val += dY[index];
#endif
    }
    dg_val = BlockReduce<T>(dg_storage).Reduce(dg_val, cub::Sum());
    db_val = BlockReduce<T>(db_storage).Reduce(db_val, cub::Sum());
    if (threadIdx.x == 0) {
      dgamma[i] = dg_val;
      dbeta[i] = db_val;
    }
    __syncthreads();
  }
}

} // namespace

template <>
bool GroupNormOp<float, CUDAContext>::RunOnDeviceImpl(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const float* X_data,
    const float* gamma_data,
    const float* beta_data,
    float* Y_data,
    float* mu_data,
    float* rsig_data) {
  const std::array<int, 4> dims = order_ == StorageOrder::NCHW
      ? std::array<int, 4>{N, G, D, HxW}
      : std::array<int, 4>{N, HxW, G, D};
  const std::array<int, 2> axes = order_ == StorageOrder::NCHW
      ? std::array<int, 2>{2, 3}
      : std::array<int, 2>{1, 3};

  // Computes mean and variance.
  math::Moments<float, CUDAContext>(
      4, dims.data(), 2, axes.data(), X_data, mu_data, rsig_data, &context_);

  // Uses rsqrt to computes 1 / std which is much faster than computes std.
  InvStdCUDAKernel<<<
      CAFFE_GET_BLOCKS(N * G),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(N * G, epsilon_, rsig_data, rsig_data);

  // Computes Y = gamma * (X - mu) * rsig + beta.
  const int size = N * G * D * HxW;
  if (order_ == StorageOrder::NCHW) {
    GroupNormForwardCUDAKernel<float, StorageOrder::NCHW>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            G,
            D,
            HxW,
            X_data,
            mu_data,
            rsig_data,
            gamma_data,
            beta_data,
            Y_data);
  } else {
    GroupNormForwardCUDAKernel<float, StorageOrder::NHWC>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            G,
            D,
            HxW,
            X_data,
            mu_data,
            rsig_data,
            gamma_data,
            beta_data,
            Y_data);
  }
  return true;
}

// Math:
// let: s = gamma * rsig
// let: b = beta - mu * gamma * rsig
// then: Y = s * X + b
template <>
bool GroupNormGradientOp<float, CUDAContext>::RunOnDeviceImpl(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const float* dY_data,
    const float* X_data,
    const float* mu_data,
    const float* rsig_data,
    const float* gamma_data,
    float* dX_data,
    float* dgamma_data,
    float* dbeta_data) {
  const int size = N * G * D * HxW;
  const int C = G * D;
  ds_.Resize(N, G);
  db_.Resize(N, G);
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  if (order_ == StorageOrder::NCHW) {
    // Computes dL/ds and dL/db.
    // dL/ds = Sum(dL/dY * gamma * X)
    // dL/db = Sum(dL/dY * gamma)
    ComputeInternalGradientsCUDAKernel<float, StorageOrder::NCHW>
        <<<std::min(N * G, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N, G, D, HxW, dY_data, X_data, gamma_data, ds_data, db_data);

    // Computes dL/dX.
    GroupNormBackwardCUDAKernel<float, StorageOrder::NCHW>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            G,
            D,
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
    GammaBetaBackwardCUDAKernel<float, StorageOrder::NCHW>
        <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            G,
            D,
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
    ComputeInternalGradientsCUDAKernel<float, StorageOrder::NHWC>
        <<<std::min(N * G, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N, G, D, HxW, dY_data, X_data, gamma_data, ds_data, db_data);

    // Computes dL/dX.
    GroupNormBackwardCUDAKernel<float, StorageOrder::NHWC>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size,
            G,
            D,
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
    GammaBetaBackwardCUDAKernel<float, StorageOrder::NHWC>
        <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            G,
            D,
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

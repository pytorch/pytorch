// Copyright 2004-present Facebook. All Rights Reserved.

// ------------------------------------------------------------------
// GroupNorm op in Caffe2
// Written by Kaiming He
// see https://arxiv.org/abs/1803.08494
// This is a stand-alone op: Y = gamma * (X - mu) / sig + beta
// ------------------------------------------------------------------

#include <cub/block/block_reduce.cuh>
#include "caffe2/core/context_gpu.h"
#include "group_norm_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// adapted from reduction_front_back_ops.cu
template <typename T, bool NORMALIZE>
__global__ void rowwise_sum_kernel(
    const int rows,
    const int cols,
    const T* data,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    T sum = 0;
    const int rowOffset = rowIndex * cols;
    const int length = lengths == nullptr ? cols : lengths[rowIndex];
    for (int colIndex = threadIdx.x; colIndex < length;
         colIndex += blockDim.x) {
      sum += data[rowOffset + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[rowIndex] = NORMALIZE ? sum / length : sum;
    }
    __syncthreads();
  }
}

// adapted from reduction_front_back_ops.cu
template <typename T, bool NORMALIZE>
__global__ void rowwise_sumsqr_kernel(
    const int rows,
    const int cols,
    const T* data,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    T sum = 0;
    const int rowOffset = rowIndex * cols;
    const int length = lengths == nullptr ? cols : lengths[rowIndex];
    for (int colIndex = threadIdx.x; colIndex < length;
         colIndex += blockDim.x) {
      sum += data[rowOffset + colIndex] * data[rowOffset + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[rowIndex] = NORMALIZE ? sum / length : sum;
    }
    __syncthreads();
  }
}

// adapted from reduction_front_back_ops.cu
template <typename T, bool NORMALIZE>
__global__ void rowwise_sumXY_kernel(
    const int rows,
    const int cols,
    const T* Xdata,
    const T* Ydata,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    T sum = 0;
    const int rowOffset = rowIndex * cols;
    const int length = lengths == nullptr ? cols : lengths[rowIndex];
    for (int colIndex = threadIdx.x; colIndex < length;
         colIndex += blockDim.x) {
      sum += Xdata[rowOffset + colIndex] * Ydata[rowOffset + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[rowIndex] = NORMALIZE ? sum / length : sum;
    }
    __syncthreads();
  }
}

// adapted from reduction_front_back_ops.cu
template <typename T, bool NORMALIZE>
__global__ void columnwise_sum_kernel(
    const int rows,
    const int cols,
    const T* data,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int colIndex = blockIdx.x; colIndex < cols; colIndex += gridDim.x) {
    T sum = 0;
    const int length = lengths == nullptr ? rows : lengths[colIndex];
    for (int rowIndex = threadIdx.x; rowIndex < length;
         rowIndex += blockDim.x) {
      sum += data[rowIndex * cols + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[colIndex] = NORMALIZE ? sum / length : sum;
    }
    __syncthreads();
  }
}

__global__ void StdKernel(
    const int n,
    float* std,
    const float* EX,
    const float* EX2,
    const float epsilon) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    std[i] = sqrt(max(EX2[i] - EX[i] * EX[i] + epsilon, 0.f));
  }
}

__global__ void GroupNormForwardKernel(
    const int n,
    float* Y, // (NG, C/G * HW)
    const float* X, // (NG, C/G * HW)
    const float* mu, // (N, G)
    const float* sig, // (N, G)
    const float* gamma, // (C,)
    const float* beta, // (C,)
    const int C,
    const int dim_per_gp,
    const int HxW) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int i_mu = i / (dim_per_gp * HxW);
    const int i_gamma = (i / HxW) % C;

    Y[i] = gamma[i_gamma] * (X[i] - mu[i_mu]) / sig[i_mu] + beta[i_gamma];
  }
}

__global__ void GammaBackwardKernel(
    const int n, // N * C
    float* dgamma, // (C,)
    const float* mu, // (N, G)
    const float* sig, // (N, G)
    const float* ds, // (N, C)
    const float* db, // (N, C)
    const int C,
    const int dim_per_gp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int i_gamma = i % C;
    const int i_mu = i / dim_per_gp;

    // dgamma[i_gamma] +=
    //   -db[i] * mu[i_mu] / sig[i_mu] + ds[i] / sig[i_mu];

    gpu_atomic_add(
        static_cast<float>(-db[i] * mu[i_mu] / sig[i_mu] + ds[i] / sig[i_mu]),
        dgamma + i_gamma);
  }
}

__global__ void ChannelScaleKernel(
    const int n, // N * C
    float* out, // (N, C)
    const float* in, // (N, C)
    const float* scale, // (C,)
    const int C) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int i_scale = i % C;
    out[i] = in[i] * scale[i_scale];
  }
}

__global__ void GroupNormBackwardKernel( // wrt X
    const int n, // N * C * H * W
    float* dX, // (N, C, H, W)
    const float* dY, // (N, C, H, W)
    const float* X, // (N, C, H, W)
    const float* mu, // (N, G)
    const float* sig, // (N, G)
    const float* gamma_ds_reduced, // (N, G)
    const float* gamma_db_reduced, // (N, G)
    const float* gamma, // (C,)
    const int C,
    const int dim_per_gp,
    const int HxW,
    const float denom // = 1 / (dim_per_gp * HxW)
) {
  // maths:
  // d/dX = d/dy * dy/dX + d/dsig * dsig/dX + d/dmu * dmu/dX
  // dy/dX = s = gamma / sig
  // d/dsig = d/ds * ds/dsig + d/db * db/dsig
  //        = d/ds * (-gamma / sig^2) + d/db * (gamma * mu / sig^2)
  // d/dmu = d/db * db/dmu = d/db * (-gamma / sig)
  // dsig/dX = (X - mu) / sig / n
  // dmu/dX = 1/n

  CUDA_1D_KERNEL_LOOP(i, n) {
    const int i_gamma = (i / HxW) % C; // for gamma, beta
    const int i_mu = (i / HxW) / dim_per_gp; // for mu, sig

    // d/dy * dy/dX
    const float val1 = dY[i] * gamma[i_gamma] / sig[i_mu];

    // d/dsig
    const float val2 =
        (gamma_db_reduced[i_mu] * mu[i_mu] - gamma_ds_reduced[i_mu]) /
        sig[i_mu] / sig[i_mu];

    // dsig/dX
    const float val3 = (X[i] - mu[i_mu]) / sig[i_mu] * denom;

    // d/dmu
    const float val4 = -gamma_db_reduced[i_mu] / sig[i_mu];

    // put together
    dX[i] = val1 + val2 * val3 + val4 * denom;
  }
}

} // namespace

template <>
bool GroupNormOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& gamma = Input(1);
  auto& beta = Input(2);

  auto* Y = Output(0);
  auto* mu = Output(1);
  auto* sig = Output(2);

  const int N = X.dim32(0);
  const int C = X.dim32(1);

  const int G = num_groups_;
  CAFFE_ENFORCE(C % G == 0);

  const int dim_per_gp = C / G;
  const int HxW = X.size() / N / C; // this can be THW

  CAFFE_ENFORCE(gamma.size() == C);
  CAFFE_ENFORCE(beta.size() == C);

  Y->ResizeLike(X);
  mu->Resize(N * G);
  sig->Resize(N * G);

  // --------------------------------------------------------
  // step 1: compute mean (EX)
  // mu := mean(X) along the last dimensions
  rowwise_sum_kernel<float, true> // true: to computer average
      <<<std::min(N * G, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          N * G,
          HxW * dim_per_gp,
          X.data<float>(),
          nullptr,
          mu->mutable_data<float>());

  // --------------------------------------------------------
  // step 2: compute mean of X^2 (EX2)
  // sig := mean(X^2) along the last dimensions
  rowwise_sumsqr_kernel<float, true> // true: to computer average
      <<<std::min(N * G, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          N * G,
          HxW * dim_per_gp,
          X.data<float>(),
          nullptr,
          sig->mutable_data<float>());

  // step 2.1: compute std sigma
  StdKernel<<<
      CAFFE_GET_BLOCKS(sig->size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      sig->size(),
      sig->mutable_data<float>(),
      mu->data<float>(), // EX
      sig->data<float>(), // EX2
      epsilon_);

  // --------------------------------------------------------
  // step 3: normalize and affine
  GroupNormForwardKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      Y->mutable_data<float>(),
      X.data<float>(),
      mu->data<float>(),
      sig->data<float>(),
      gamma.data<float>(),
      beta.data<float>(),
      C,
      dim_per_gp,
      HxW);

  return true;
}

template <>
bool GroupNormGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto& X = Input(1);
  auto& gamma = Input(2);
  auto& beta = Input(3);
  auto& mu = Input(4);
  auto& sig = Input(5);

  auto* dX = Output(0);
  auto* dgamma = Output(1);
  auto* dbeta = Output(2);

  const int N = X.dim32(0);
  const int C = X.dim32(1);

  const int G = num_groups_;
  CAFFE_ENFORCE(C % G == 0);

  const int dim_per_gp = C / G;
  const int HxW = X.size() / N / C; // this can be THW

  CAFFE_ENFORCE(gamma.size() == C);
  CAFFE_ENFORCE(beta.size() == C);

  dX->ResizeLike(X);
  dgamma->ResizeLike(gamma);
  dbeta->ResizeLike(beta);

  // if (sum_dY_.size() != NxG) sum_dY_.Resize(NxG);
  // if (sum_YdY_.size() != NxG) sum_YdY_.Resize(NxG);

  // set all-one vector
  // hack: this will be used as:
  // (i): len = HxW
  // (ii): len = C/G (for group norm)
  // (iii): len = N
  const int multiplier_size = max(N, max(HxW, dim_per_gp));
  if (sum_multiplier_.size() != multiplier_size) {
    sum_multiplier_.Resize(multiplier_size);
    math::Set<float, CUDAContext>(
        sum_multiplier_.size(),
        1.f,
        sum_multiplier_.mutable_data<float>(),
        &context_);
  }

  if (buffer_.size() != N * C)
    buffer_.Resize(N * C);
  if (buffer1_.size() != N * C)
    buffer1_.Resize(N * C);

  // let: s = gamma / sig                (shape: N * C)
  // let: b = beta - mu * gamma / sig    (shape: N * C)
  // then: Y = s * X + b
  // --------------------------------------------------------
  // step 1: grad of beta
  // d/db = sum (d/dY)       (shape: N * C)
  // d/dbeta = sum(d/db)     (shape: C)

  // Decompose Gemv into two so it is faster.
  auto& db = buffer_;

  // this appears to be slower than Gemv
  // rowwise_sum_kernel<float, false>  // false: to computer sum
  //     <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
  //        CAFFE_CUDA_NUM_THREADS,
  //        0,
  //        context_.cuda_stream()>>>(
  // N * C, HxW, dY.data<float>(),
  // nullptr, db.mutable_data<float>());

  math::Gemv<float, CUDAContext>(
      CblasNoTrans,
      N * C,
      HxW,
      float(1.),
      dY.data<float>(),
      sum_multiplier_.data<float>(), // len = HxW here
      float(0.),
      db.mutable_data<float>(), // N*C
      &context_);

  // columnwise_sum_kernel<float, false>  // false: to computer sum
  //     <<<std::min(C, CAFFE_MAXIMUM_NUM_BLOCKS),
  //        CAFFE_CUDA_NUM_THREADS,
  //        0,
  //        context_.cuda_stream()>>>(
  // N, C, db.data<float>(),
  // nullptr, dbeta->mutable_data<float>());

  math::Gemv<float, CUDAContext>(
      CblasTrans, // over the N axis
      N,
      C,
      float(1.),
      db.data<float>(),
      sum_multiplier_.data<float>(), // len = N
      float(0.),
      dbeta->mutable_data<float>(), // C
      &context_);

  // --------------------------------------------------------
  // step 2: grad of gamma
  // d/dgamma = d/db * db/dgamma + d/ds * ds/dgamma
  // = d/db * (-mu/sig) + d/ds / sig

  // step 2.1: compute d/ds
  // d/ds = sum (d/dY * X)

  // use buffer1_ as the buffer of ds (same size)
  auto& ds = buffer1_; // NxC size
  rowwise_sumXY_kernel<float, false> // false: to computer sum
      <<<std::min(N * C, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          N * C,
          HxW,
          X.data<float>(),
          dY.data<float>(),
          nullptr,
          ds.mutable_data<float>());

  // step 2.2: compute d/dgamma
  // set 0 as we will do atomicAdd
  math::Set<float, CUDAContext>(
      dgamma->size(), 0.f, dgamma->mutable_data<float>(), &context_);
  GammaBackwardKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N * C,
      dgamma->mutable_data<float>(),
      mu.data<float>(),
      sig.data<float>(),
      ds.data<float>(),
      db.data<float>(),
      C,
      dim_per_gp);

  // --------------------------------------------------------
  // step 3: grad of X
  // d/dX = d/dy * dy/dX + d/dsig * dsig/dX + d/dmu * dmu/dX
  // d/dsig = d/ds * ds/dsig + d/db * db/dsig
  //        = d/ds * (-gamma / sig^2) + d/db * (gamma * mu / sig^2)
  // d/dmu = d/db * db/dmu = d/db * (-gamma / sig)
  // dsig/dX = (X - mu) / sig / n
  // dmu/dX = 1/n

  if (buffer2_.size() != N * G)
    buffer2_.Resize(N * G);
  if (buffer3_.size() != N * G)
    buffer3_.Resize(N * G);

  auto& gamma_db = buffer_; // N * C, rewrite db (buffer_)
  ChannelScaleKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N * C,
      gamma_db.mutable_data<float>(),
      db.data<float>(),
      gamma.data<float>(),
      C);

  auto& gamma_ds = buffer1_; // N * C, rewrite ds (buffer1_)
  ChannelScaleKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N * C,
      gamma_ds.mutable_data<float>(),
      ds.data<float>(),
      gamma.data<float>(),
      C);

  auto& gamma_db_reduced = buffer2_; // NxG size
  // rowwise_sum_kernel<float, false>  // false: to computer sum
  //     <<<std::min(N * G, CAFFE_MAXIMUM_NUM_BLOCKS),
  //        CAFFE_CUDA_NUM_THREADS,
  //        0,
  //        context_.cuda_stream()>>>(
  // N * G, dim_per_gp, gamma_db.data<float>(),
  // nullptr, gamma_db_reduced.mutable_data<float>());
  math::Gemv<float, CUDAContext>(
      CblasNoTrans,
      N * G,
      dim_per_gp,
      float(1.),
      gamma_db.data<float>(),
      sum_multiplier_.data<float>(), // len = dim_per_gp here
      float(0.),
      gamma_db_reduced.mutable_data<float>(), // N*G
      &context_);

  auto& gamma_ds_reduced = buffer3_; // NxG size
  // rowwise_sum_kernel<float, false>  // false: to computer sum
  //     <<<std::min(N * G, CAFFE_MAXIMUM_NUM_BLOCKS),
  //        CAFFE_CUDA_NUM_THREADS,
  //        0,
  //        context_.cuda_stream()>>>(
  // N * G, dim_per_gp, gamma_ds.data<float>(),
  // nullptr, gamma_ds_reduced.mutable_data<float>());
  math::Gemv<float, CUDAContext>(
      CblasNoTrans,
      N * G,
      dim_per_gp,
      float(1.),
      gamma_ds.data<float>(),
      sum_multiplier_.data<float>(), // len = dim_per_gp here
      float(0.),
      gamma_ds_reduced.mutable_data<float>(), // N*G
      &context_);

  // compute d/dX
  GroupNormBackwardKernel<<<
      CAFFE_GET_BLOCKS(dX->size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      dX->size(),
      dX->mutable_data<float>(),
      dY.data<float>(),
      X.data<float>(),
      mu.data<float>(),
      sig.data<float>(),
      gamma_ds_reduced.data<float>(),
      gamma_db_reduced.data<float>(),
      gamma.data<float>(),
      C,
      dim_per_gp,
      HxW,
      1.f / HxW / dim_per_gp);

  // ------
  return true;
}

REGISTER_CUDA_OPERATOR(GroupNorm, GroupNormOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    GroupNormGradient,
    GroupNormGradientOp<float, CUDAContext>);
} // namespace caffe2

#include <ATen/native/layer_norm.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <c10/cuda/CUDAMathCompat.h>

namespace at {
namespace native {

namespace {

constexpr int kCUDANumThreads = 256;
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kColwiseReduceTileSize = 32;

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid] : 0;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N,
    T eps,
    const T* X,
    T* mean,
    T* rstd) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC m_shared[C10_WARP_SIZE];
  __shared__ T_ACC v_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    sum1 += static_cast<T_ACC>(X[index]);
    sum2 += static_cast<T_ACC>(X[index]) * static_cast<T_ACC>(X[index]);
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, m_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, v_shared);
  if (threadIdx.x == 0) {
    const T_ACC scale = T_ACC(1) / static_cast<T_ACC>(N);
    sum1 *= scale;
    sum2 = c10::cuda::compat::max(sum2 * scale - sum1 * sum1, T_ACC(0));
    mean[i] = sum1;
    rstd[i] = c10::cuda::compat::rsqrt(sum2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
__global__ void LayerNormForwardCUDAKernel(
    int64_t N,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
  using T_ACC = acc_type<T, true>;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    const T_ACC beta_v =
        beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
    Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
            static_cast<T_ACC>(rstd[i]) * gamma_v +
        beta_v;
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    acc_type<T, true>* ds,
    acc_type<T, true>* db) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC ds_shared[C10_WARP_SIZE];
  __shared__ T_ACC db_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    sum1 +=
        static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]) * gamma_v;
    sum2 += static_cast<T_ACC>(dY[index]) * gamma_v;
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, db_shared);
  if (threadIdx.x == 0) {
    ds[i] = sum1;
    db[i] = sum2;
  }
}

template <typename T>
__global__ void ComputeGradientFusedParamsCUDAKernel(
    int64_t M,
    int64_t N,
    const T* mean,
    const T* rstd,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    acc_type<T, true>* c1,
    acc_type<T, true>* c2) {
  using T_ACC = acc_type<T, true>;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < M) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(N);
    const T_ACC a = (db[index] * static_cast<T_ACC>(mean[index]) - ds[index]) *
        static_cast<T_ACC>(rstd[index]) * static_cast<T_ACC>(rstd[index]) *
        static_cast<T_ACC>(rstd[index]) * s;
    c1[index] = a;
    c2[index] =
        -(a * static_cast<T_ACC>(mean[index]) +
          db[index] * static_cast<T_ACC>(rstd[index]) * s);
  }
}

template <typename T>
__global__ void LayerNormBackwardCUDAKenrel(
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* a,
    const acc_type<T, true>* b,
    const acc_type<T, true>* c,
    T* dX) {
  using T_ACC = acc_type<T, true>;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    dX[index] =
        static_cast<T_ACC>(a[i]) * static_cast<T_ACC>(dY[index]) * gamma_v +
        b[i] * static_cast<T_ACC>(X[index]) + c[i];
  }
}

template <typename T>
__global__ void GammaBetaBackwardSimpleCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dg,
    T* db) {
  using T_ACC = acc_type<T, true>;
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += dg == nullptr ? T_ACC(0)
                            : static_cast<T_ACC>(dY[index]) *
              (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
              static_cast<T_ACC>(rstd[i]);
      sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index]);
    }
    if (dg != nullptr) {
      dg[j] = sum1;
    }
    if (db != nullptr) {
      db[j] = sum2;
    }
  }
}

template <typename T>
__global__ void GammaBetaBackwardCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dg,
    T* db) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  __shared__ T_ACC b_shared[kColwiseReduceTileSize][kColwiseReduceTileSize + 1];
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (j < N) {
    for (int64_t i = threadIdx.y; i < M; i += blockDim.y * 2) {
      const int64_t i1 = i;
      const int64_t i2 = i + blockDim.y;
      const int64_t index1 = i1 * N + j;
      const int64_t index2 = i2 * N + j;
      dg_sum1 += dg == nullptr ? T_ACC(0)
                               : static_cast<T_ACC>(dY[index1]) *
              (static_cast<T_ACC>(X[index1]) - static_cast<T_ACC>(mean[i1])) *
              static_cast<T_ACC>(rstd[i1]);
      db_sum1 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index1]);
      if (i2 < M) {
        dg_sum2 += dg == nullptr ? T_ACC(0)
                                 : static_cast<T_ACC>(dY[index2]) *
                (static_cast<T_ACC>(X[index2]) - static_cast<T_ACC>(mean[i2])) *
                static_cast<T_ACC>(rstd[i2]);
        db_sum2 += db == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index2]);
      }
    }
  }
  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = WarpReduceSum(sum1);
  sum2 = WarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = WarpReduceSum(sum1);
  sum2 = WarpReduceSum(sum2);
  if (threadIdx.x == 0) {
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (j < N) {
      if (dg != nullptr) {
        dg[j] = sum1;
      }
      if (db != nullptr) {
        db[j] = sum2;
      }
    }
  }
}

template <typename T>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  DCHECK_EQ(X.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  RowwiseMomentsCUDAKernel<T>
      <<<M, kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          N, eps, X_data, mean_data, rstd_data);
  LayerNormForwardCUDAKernel<T><<<M, kCUDANumThreads, 0, cuda_stream>>>(
      N, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);
  AT_CUDA_CHECK(cudaGetLastError());
}

void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      X.scalar_type(), "LayerNormKernelImpl", [&]() {
        LayerNormKernelImplInternal<scalar_t>(
            X, gamma, beta, M, N, static_cast<scalar_t>(eps), Y, mean, rstd);
      });
}

template <typename T>
void LayerNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  using T_ACC = acc_type<T, true>;
  DCHECK_EQ(dY.numel(), M * N);
  DCHECK_EQ(X.numel(), M * N);
  DCHECK_EQ(mean.numel(), M);
  DCHECK_EQ(rstd.numel(), M);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  const T* dY_data = dY.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const T* mean_data = mean.template data_ptr<T>();
  const T* rstd_data = rstd.template data_ptr<T>();
  const T* gamma_data =
      gamma.defined() ? gamma.template data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (dX_data != nullptr) {
    const auto kAccType = X.scalar_type() == kHalf ? kFloat : X.scalar_type();
    Tensor ds = at::empty({M}, X.options().dtype(kAccType));
    Tensor db = at::empty({M}, X.options().dtype(kAccType));
    Tensor scale = at::empty({M}, X.options().dtype(kAccType));
    Tensor bias = at::empty({M}, X.options().dtype(kAccType));
    T_ACC* ds_data = ds.template data_ptr<T_ACC>();
    T_ACC* db_data = db.template data_ptr<T_ACC>();
    T_ACC* scale_data = scale.template data_ptr<T_ACC>();
    T_ACC* bias_data = bias.template data_ptr<T_ACC>();
    ComputeInternalGradientsCUDAKernel<T>
        <<<M, kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
            N, dY_data, X_data, gamma_data, ds_data, db_data);
    const int64_t B = (M + kCUDANumThreads - 1) / kCUDANumThreads;
    ComputeGradientFusedParamsCUDAKernel<T>
        <<<B, kCUDANumThreads, 0, cuda_stream>>>(
            M,
            N,
            mean_data,
            rstd_data,
            ds_data,
            db_data,
            scale_data,
            bias_data);
    LayerNormBackwardCUDAKenrel<T><<<M, kCUDANumThreads, 0, cuda_stream>>>(
        N,
        dY_data,
        X_data,
        gamma_data,
        rstd_data,
        scale_data,
        bias_data,
        dX_data);
  }
  if (dgamma->defined() || dbeta->defined()) {
    T* dgamma_data =
        dgamma->defined() ? dgamma->template data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;
    if (M < 512) {
      // For small batch size, do colwise reduce directly.
      const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBetaBackwardSimpleCUDAKernel<T>
          <<<B, kCUDANumThreads, 0, cuda_stream>>>(
              M,
              N,
              dY_data,
              X_data,
              mean_data,
              rstd_data,
              dgamma_data,
              dbeta_data);
    } else {
      const int64_t B =
          (N + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
      constexpr int kThreadX = kColwiseReduceTileSize;
      constexpr int kThreadY = kColwiseReduceTileSize / 2;
      GammaBetaBackwardCUDAKernel<T>
          <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
              M,
              N,
              dY_data,
              X_data,
              mean_data,
              rstd_data,
              dgamma_data,
              dbeta_data);
    }
  }
}

void LayerNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
        LayerNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      });
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> layer_norm_cuda(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    int64_t M,
    int64_t N,
    double eps) {
  Tensor Y = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor mean = at::empty({M}, X.options());
  Tensor rstd = at::empty({M}, X.options());
  if (M > 0) {
    LayerNormKernelImpl(X, gamma, beta, M, N, eps, &Y, &mean, &rstd);
  }
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cuda(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::native::zeros_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::native::zeros_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (M > 0) {
    LayerNormBackwardKernelImpl(
        dY, X, mean, rstd, gamma, M, N, &dX, &dgamma, &dbeta);
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}


REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl);

} // namespace native
} // namespace at

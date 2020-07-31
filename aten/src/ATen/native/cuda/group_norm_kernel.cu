#include <ATen/native/group_norm.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <c10/cuda/CUDAMathCompat.h>

namespace at {
namespace native {

namespace {

constexpr int kCUDANumThreads = 256;
constexpr int kReduceTileSize = 32;

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
  sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, m_shared);
  sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, v_shared);
  if (threadIdx.x == 0) {
    const T_ACC scale = T_ACC(1) / static_cast<T_ACC>(N);
    sum1 *= scale;
    sum2 = c10::cuda::compat::max(sum2 * scale - sum1 * sum1, T_ACC(0));
    mean[i] = sum1;
    rstd[i] = c10::cuda::compat::rsqrt(sum2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    acc_type<T, true>* a,
    acc_type<T, true>* b) {
  using T_ACC = acc_type<T, true>;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C) {
    const int64_t ng = index / (C / group);
    const int64_t c = index % C;
    const T_ACC x = (gamma == nullptr)
        ? static_cast<T_ACC>(rstd[ng])
        : static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
    a[index] = x;
    b[index] = -x * static_cast<T_ACC>(mean[ng]) +
        (beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[c]));
  }
}

template <typename T>
__global__ void GroupNormForwardSimpleCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* X,
    const acc_type<T, true>* a,
    const acc_type<T, true>* b,
    T* Y) {
  using T_ACC = acc_type<T, true>;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * HxW) {
    const int64_t nc = index / HxW;
    Y[index] = a[nc] * static_cast<T_ACC>(X[index]) + b[nc];
  }
}

template <typename T>
__global__ void GroupNormForwardCUDAKernel(
    int64_t HxW,
    const T* X,
    const acc_type<T, true>* a,
    const acc_type<T, true>* b,
    T* Y) {
  using T_ACC = acc_type<T, true>;
  const int64_t nc = blockIdx.x;
  for (int64_t hw = threadIdx.x; hw < HxW; hw += blockDim.x) {
    const int64_t index = nc * HxW + hw;
    Y[index] = a[nc] * static_cast<T_ACC>(X[index]) + b[nc];
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t HxW,
    const T* dY,
    const T* X,
    acc_type<T, true>* ds,
    acc_type<T, true>* db) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC ds_shared[C10_WARP_SIZE];
  __shared__ T_ACC db_shared[C10_WARP_SIZE];
  const int64_t nc = blockIdx.x;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t hw = threadIdx.x; hw < HxW; hw += blockDim.x) {
    const int64_t index = nc * HxW + hw;
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]);
    sum2 += static_cast<T_ACC>(dY[index]);
  }
  sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  if (threadIdx.x == 0) {
    ds[nc] = sum1;
    db[nc] = sum2;
  }
}

template <typename T>
__global__ void ComputeGradOutputCoeffientCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* rstd,
    const T* gamma,
    acc_type<T, true>* c1) {
  using T_ACC = acc_type<T, true>;
  const int64_t nc = blockIdx.x * blockDim.x + threadIdx.x;
  if (nc < N * C) {
    const int64_t ng = nc / (C / group);
    const int64_t c = nc % C;
    c1[nc] = static_cast<T_ACC>(rstd[ng]) *
        (gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]));
  }
}

template <typename T>
__global__ void ComputeBackwardFusedParamsCUDAKernel(
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    acc_type<T, true>* c2,
    acc_type<T, true>* c3) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC ds_shared[C10_WARP_SIZE];
  __shared__ T_ACC db_shared[C10_WARP_SIZE];
  const int64_t G = group;
  const int64_t D = C / G;
  const int64_t n = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int64_t ng = n * G + g;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
    const int64_t index = ng * D + i;
    const int64_t c = g * D + i;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]);
    sum1 += ds[index] * gamma_v;
    sum2 += db[index] * gamma_v;
  }
  sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  if (threadIdx.x == 0) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
    const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
        static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
        static_cast<T_ACC>(rstd[ng]) * s;
    c2[ng] = x;
    c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
        sum2 * static_cast<T_ACC>(rstd[ng]) * s;
  }
}

template <typename T>
__global__ void GroupNormBackwardSimpleCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* dY,
    const T* X,
    const acc_type<T, true>* c1,
    const acc_type<T, true>* c2,
    const acc_type<T, true>* c3,
    T* dX) {
  using T_ACC = acc_type<T, true>;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N * C * HxW) {
    const int64_t nc = index / HxW;
    const int64_t ng = nc / (C / group);
    dX[index] = c1[nc] * static_cast<T_ACC>(dY[index]) +
        c2[ng] * static_cast<T_ACC>(X[index]) + c3[ng];
  }
}

template <typename T>
__global__ void GroupNormBackwardCUDAKernel(
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* dY,
    const T* X,
    const acc_type<T, true>* c1,
    const acc_type<T, true>* c2,
    const acc_type<T, true>* c3,
    T* dX) {
  using T_ACC = acc_type<T, true>;
  const int64_t D = C / group;
  const int64_t nc = blockIdx.x;
  const int64_t ng = nc / D;
  for (int64_t hw = threadIdx.x; hw < HxW; hw += blockDim.x) {
    const int64_t index = nc * HxW + hw;
    dX[index] = c1[nc] * static_cast<T_ACC>(dY[index]) +
        c2[ng] * static_cast<T_ACC>(X[index]) + c3[ng];
  }
}

template <typename T>
__global__ void GammaBetaBackwardSimpleCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    T* dgamma,
    T* dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t n = 0; n < N; ++n) {
      const int64_t nc = n * C + c;
      const int64_t ng = n * G + c / D;
      sum1 += (dgamma == nullptr)
          ? T_ACC(0)
          : ((ds[nc] - db[nc] * static_cast<T_ACC>(mean[ng])) *
             static_cast<T_ACC>(rstd[ng]));
      sum2 += (dbeta == nullptr) ? T_ACC(0) : db[nc];
    }
    if (dgamma != nullptr) {
      dgamma[c] = sum1;
    }
    if (dbeta != nullptr) {
      dbeta[c] = sum2;
    }
  }
}

template <typename T>
__global__ void GammaBetaBackwardCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    T* dgamma,
    T* dbeta) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC g_shared[kReduceTileSize][kReduceTileSize + 1];
  __shared__ T_ACC b_shared[kReduceTileSize][kReduceTileSize + 1];
  const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    for (int64_t n = threadIdx.y; n < N; n += blockDim.y * 2) {
      const int64_t n1 = n;
      const int64_t n2 = n + blockDim.y;
      const int64_t nc1 = n1 * C + c;
      const int64_t nc2 = n2 * C + c;
      const int64_t ng1 = n1 * G + c / D;
      const int64_t ng2 = n2 * G + c / D;
      dg_sum1 += dgamma == nullptr
          ? T_ACC(0)
          : ((ds[nc1] - db[nc1] * static_cast<T_ACC>(mean[ng1])) *
             static_cast<T_ACC>(rstd[ng1]));
      db_sum1 += dbeta == nullptr ? T_ACC(0) : db[nc1];
      if (n2 < N) {
        dg_sum2 += dgamma == nullptr
            ? T_ACC(0)
            : ((ds[nc2] - db[nc2] * static_cast<T_ACC>(mean[ng2])) *
               static_cast<T_ACC>(rstd[ng2]));
        db_sum2 += dbeta == nullptr ? T_ACC(0) : db[nc2];
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
  sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
  sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y;
    if (c < C) {
      if (dgamma != nullptr) {
        dgamma[c] = sum1;
      }
      if (dbeta != nullptr) {
        dbeta[c] = sum2;
      }
    }
  }
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
  sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (c < C) {
      if (dgamma != nullptr) {
        dgamma[c] = sum1;
      }
      if (dbeta != nullptr) {
        dbeta[c] = sum2;
      }
    }
  }
}

template <typename T>
void GroupNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  using T_ACC = acc_type<T, true>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) {
    return;
  }
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const auto kAccType = X.scalar_type() == kHalf ? kFloat : X.scalar_type();
  Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* a_data = a.data_ptr<T_ACC>();
  T_ACC* b_data = b.data_ptr<T_ACC>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  RowwiseMomentsCUDAKernel<T>
      <<<N * G, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          D * HxW, eps, X_data, mean_data, rstd_data);
  int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
  ComputeFusedParamsCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
      N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
  if (HxW < kCUDANumThreads) {
    B = (N * C * HxW + kCUDANumThreads - 1) / kCUDANumThreads;
    GroupNormForwardSimpleCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
        N, C, HxW, X_data, a_data, b_data, Y_data);
  } else {
    GroupNormForwardCUDAKernel<T><<<N * C, kCUDANumThreads, 0, cuda_stream>>>(
        HxW, X_data, a_data, b_data, Y_data);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void GroupNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormKernelImpl",
      [&]() {
        AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "GroupNormKernelImpl", [&]() {
          GroupNormKernelImplInternal<scalar_t>(
              X,
              gamma,
              beta,
              N,
              C,
              HxW,
              group,
              static_cast<scalar_t>(eps),
              Y,
              mean,
              rstd);
        });
      });
}

template <typename T>
void GroupNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * G);
  TORCH_CHECK(rstd.numel() == N * G);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (N == 0) {
    if (dgamma->defined()) {
      T* dgamma_data = dgamma->data_ptr<T>();
      AT_CUDA_CHECK(cudaMemsetAsync(
          dgamma_data, 0, dgamma->numel() * sizeof(T), cuda_stream));
    }
    if (dbeta->defined()) {
      T* dbeta_data = dbeta->data_ptr<T>();
      AT_CUDA_CHECK(cudaMemsetAsync(
          dbeta_data, 0, dbeta->numel() * sizeof(T), cuda_stream));
    }
    return;
  }

  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* mean_data = mean.data_ptr<T>();
  const T* rstd_data = rstd.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->data_ptr<T>() : nullptr;
  const auto kAccType = X.scalar_type() == kHalf ? kFloat : X.scalar_type();
  Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();
  ComputeInternalGradientsCUDAKernel<T>
      <<<N * C, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          HxW, dY_data, X_data, ds_data, db_data);
  if (dX != nullptr) {
    Tensor c1 = at::empty({N, C}, X.options().dtype(kAccType));
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c1_data = c1.data_ptr<T_ACC>();
    T_ACC* c2_data = c2.data_ptr<T_ACC>();
    T_ACC* c3_data = c3.data_ptr<T_ACC>();
    int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
    ComputeGradOutputCoeffientCUDAKernel<T>
        <<<B, kCUDANumThreads, 0, cuda_stream>>>(
            N, C, G, rstd_data, gamma_data, c1_data);
    ComputeBackwardFusedParamsCUDAKernel<T>
        <<<dim3(N, G),
           cuda_utils::kCUDABlockReduceNumThreads,
           0,
           cuda_stream>>>(
            C,
            HxW,
            G,
            mean_data,
            rstd_data,
            gamma_data,
            ds_data,
            db_data,
            c2_data,
            c3_data);
    if (HxW < kCUDANumThreads) {
      B = (N * C * HxW + kCUDANumThreads - 1) / kCUDANumThreads;
      GroupNormBackwardSimpleCUDAKernel<
          T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
          N, C, HxW, G, dY_data, X_data, c1_data, c2_data, c3_data, dX_data);
    } else {
      GroupNormBackwardCUDAKernel<T>
          <<<N * C, kCUDANumThreads, 0, cuda_stream>>>(
              C, HxW, G, dY_data, X_data, c1_data, c2_data, c3_data, dX_data);
    }
  }
  if (dgamma->defined() || dbeta->defined()) {
    T* dgamma_data = dgamma->defined() ? dgamma->data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta->defined() ? dbeta->data_ptr<T>() : nullptr;
    if (N < 512) {
      // For small batch size, do colwise reduce directly.
      const int64_t B = (C + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBetaBackwardSimpleCUDAKernel<T>
          <<<B, kCUDANumThreads, 0, cuda_stream>>>(
              N,
              C,
              G,
              mean_data,
              rstd_data,
              ds_data,
              db_data,
              dgamma_data,
              dbeta_data);
    } else {
      const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
      constexpr int kThreadX = kReduceTileSize;
      constexpr int kThreadY = kReduceTileSize / 2;
      GammaBetaBackwardCUDAKernel<T>
          <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
              N,
              C,
              G,
              mean_data,
              rstd_data,
              ds_data,
              db_data,
              dgamma_data,
              dbeta_data);
    }
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void GroupNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormBackwardKernelImpl",
      [&]() {
        AT_SKIP_BFLOAT16_IF_NOT_ROCM(
            scalar_t, "GroupNormBackwardKernelImpl", [&]() {
              GroupNormBackwardKernelImplInternal<scalar_t>(
                  dY,
                  X,
                  mean,
                  rstd,
                  gamma,
                  N,
                  C,
                  HxW,
                  group,
                  dX,
                  dgamma,
                  dbeta);
            });
      });
}

} // namespace

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl);

} // namespace native
} // namespace at

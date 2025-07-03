#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/group_norm.h>

#include <type_traits>

#include <thrust/tuple.h>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

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
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    val = cuda_utils::WarpReduce(val, welford_op);
  } else {
    // There will be a warning if we declare a __shared__ WelfordType array.
    // https://github.com/pytorch/pytorch/pull/13967
    __shared__ typename std::aligned_storage<
        sizeof(WelfordType),
        alignof(WelfordType)>::type val_shared[C10_WARP_SIZE];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = cuda_utils::BlockReduce(
        val,
        welford_op,
        /*identity_element=*/WelfordType(0, 0, 0, 0),
        val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
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
  const int64_t index = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < N * C) {
    const int64_t ng = index / (C / group);
    const int64_t c = index % C;
    const T_ACC scale = (gamma == nullptr)
        ? static_cast<T_ACC>(rstd[ng])
        : static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
    a[index] = scale;
    b[index] = -scale * static_cast<T_ACC>(mean[ng]) +
        ((beta == nullptr) ? 0 : static_cast<T_ACC>(beta[c]));
  }
}

template <typename T>
__global__ void Compute1dBackwardFusedParamsCUDAKernel(
    int64_t C,
    int64_t group,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    acc_type<T, true>* c2,
    acc_type<T, true>* c3) {
  using T_ACC = acc_type<T, true>;
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
    sum1 += dY[index] * X[index] * gamma_v;
    sum2 += dY[index] * gamma_v;
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE];
    __shared__ T_ACC db_shared[C10_WARP_SIZE];
    sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D);
    const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
        static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
        static_cast<T_ACC>(rstd[ng]) * s;
    c2[ng] = x;
    c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
        sum2 * static_cast<T_ACC>(rstd[ng]) * s;
  }
}

template <typename T>
__global__ void GammaBeta1dBackwardCUDAKernel1(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dgamma,
    T* dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t n = 0; n < N; ++n) {
      const int64_t nc = n * C + c;
      const int64_t ng = n * G + c / D;
      const T_ACC dy_acc = static_cast<T_ACC>(dY[nc]);
      const T_ACC x_acc = static_cast<T_ACC>(X[nc]);
      sum1 += (dgamma == nullptr)
          ? T_ACC(0)
          : ((dy_acc * x_acc - dy_acc * static_cast<T_ACC>(mean[ng])) *
             static_cast<T_ACC>(rstd[ng]));
      sum2 += (dbeta == nullptr) ? T_ACC(0) : dy_acc;
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
__global__ void GammaBeta1dBackwardCUDAKernel2(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dgamma,
    T* dbeta) {
  using T_ACC = acc_type<T, true>;
  __shared__ T_ACC g_shared[kReduceTileSize][kReduceTileSize + 1];
  __shared__ T_ACC b_shared[kReduceTileSize][kReduceTileSize + 1];
  const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    // Accumulate each 32 cols into a 32 * 32 tile.
    // Since the blockDim is (32, 16), accumulate twice for 1st and 2nd 16 rows
    // of a 32 contiguous elements.
    for (int64_t n = threadIdx.y; n < N; n += blockDim.y * 2) {
      const int64_t n1 = n;
      const int64_t n2 = n + blockDim.y;
      const int64_t nc1 = n1 * C + c;
      const int64_t nc2 = n2 * C + c;
      const int64_t ng1 = n1 * G + c / D;
      const int64_t ng2 = n2 * G + c / D;
      const T_ACC dy1_acc = static_cast<T_ACC>(dY[nc1]);
      const T_ACC x1_acc = static_cast<T_ACC>(X[nc1]);
      dg_sum1 += dgamma == nullptr
          ? T_ACC(0)
          : ((dy1_acc * x1_acc - dy1_acc * static_cast<T_ACC>(mean[ng1])) *
             static_cast<T_ACC>(rstd[ng1]));
      db_sum1 += dbeta == nullptr ? T_ACC(0) : dy1_acc;
      if (n2 < N) {
        const T_ACC dy2_acc = static_cast<T_ACC>(dY[nc2]);
        const T_ACC x2_acc = static_cast<T_ACC>(X[nc2]);
        dg_sum2 += dgamma == nullptr
            ? T_ACC(0)
            : ((dy2_acc * x2_acc - dy2_acc * static_cast<T_ACC>(mean[ng2])) *
               static_cast<T_ACC>(rstd[ng2]));
        db_sum2 += dbeta == nullptr ? T_ACC(0) : dy2_acc;
      }
    }
  }

  // Write accumulated tile to shared memory.
  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();

  // Do warp reduce for the 1st 16 cols in the tile.
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

  // Do warp reduce for the 2nd 16 cols in the tile.
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
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t HxW,
    const T* dY,
    const T* X,
    acc_type<T, true>* ds,
    acc_type<T, true>* db) {
  using T_ACC = acc_type<T, true>;
  const int64_t nc = blockIdx.x;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t hw = threadIdx.x; hw < HxW; hw += blockDim.x) {
    const int64_t index = nc * HxW + hw;
    sum1 += static_cast<T_ACC>(dY[index]) * static_cast<T_ACC>(X[index]);
    sum2 += static_cast<T_ACC>(dY[index]);
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE];
    __shared__ T_ACC db_shared[C10_WARP_SIZE];
    sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    ds[nc] = sum1;
    db[nc] = sum2;
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
  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE];
    __shared__ T_ACC db_shared[C10_WARP_SIZE];
    sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  }
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
__global__ void GammaBetaBackwardCUDAKernel1(
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
  const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
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
__global__ void GammaBetaBackwardCUDAKernel2(
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
  const int64_t c = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  T_ACC dg_sum1 = 0;
  T_ACC dg_sum2 = 0;
  T_ACC db_sum1 = 0;
  T_ACC db_sum2 = 0;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    // Accumulate each 32 cols into a 32 * 32 tile.
    // Since the blockDim is (32, 16), accumulate twice for 1st and 2nd 16 rows
    // of a 32 contiguous elements.
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

  // Write accumulated tile to shared memory.
  g_shared[threadIdx.y][threadIdx.x] = dg_sum1;
  g_shared[threadIdx.y + blockDim.y][threadIdx.x] = dg_sum2;
  b_shared[threadIdx.y][threadIdx.x] = db_sum1;
  b_shared[threadIdx.y + blockDim.y][threadIdx.x] = db_sum2;
  __syncthreads();

  // Do warp reduce for the 1st 16 cols in the tile.
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

  // Do warp reduce for the 2nd 16 cols in the tile.
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
void GroupNorm1dForward(
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t group,
    Tensor& Y) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  if (gamma.defined() && beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T gamma, T beta) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma) +
          static_cast<T_ACC>(beta);
    });
  } else if (gamma.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T gamma) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
    });
  } else if (beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T beta) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd) +
          static_cast<T_ACC>(beta);
    });
  } else {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D}))
                    .add_owned_const_input(X.view({N * G, D}))
                    .add_owned_input(mean.view({N * G, 1}))
                    .add_owned_input(rstd.view({N * G, 1}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd);
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
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
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  using T_ACC = acc_type<T, true>;
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  if (N == 0) {
    return;
  }
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = mean.mutable_data_ptr<T>();
  T* rstd_data = rstd.mutable_data_ptr<T>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = D * HxW < cuda_utils::kCUDABlockReduceNumThreads
      ? at::cuda::warp_size()
      : cuda_utils::kCUDABlockReduceNumThreads;
  RowwiseMomentsCUDAKernel<T><<<N * G, num_threads, 0, cuda_stream>>>(
      D * HxW, eps, X_data, mean_data, rstd_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (HxW == 1) {
    GroupNorm1dForward<T>(X, mean, rstd, gamma, beta, N, C, G, Y);
  } else if (!gamma.defined() && !beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D * HxW}))
                    .add_owned_const_input(X.view({N * G, D * HxW}))
                    .add_owned_input(mean.view({N * G, 1}))
                    .add_owned_input(rstd.view({N * G, 1}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd);
    });
  } else {
    const auto kAccType =
        (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
        ? kFloat
        : X.scalar_type();
    Tensor a = at::empty({N, C}, X.options().dtype(kAccType));
    Tensor b = at::empty({N, C}, X.options().dtype(kAccType));
    const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
    const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
    T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
    T_ACC* b_data = b.mutable_data_ptr<T_ACC>();

    // TODO: Since there is some issues in gpu_kernel_multiple_outputs, we are
    // using manual kernel here. Make it using gpu_kernel_multiple_outputs once
    // the issue fixed.
    const int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
    ComputeFusedParamsCUDAKernel<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
        N, C, G, mean_data, rstd_data, gamma_data, beta_data, a_data, b_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * C, HxW}))
                    .add_owned_const_input(X.view({N * C, HxW}))
                    .add_owned_input(a.view({N * C, 1}))
                    .add_owned_input(b.view({N * C, 1}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
      return a * static_cast<T_ACC>(x) + b;
    });
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
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormKernelImpl",
      [&]() {
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
}

template <typename T>
void GroupNorm1dBackward(
    const Tensor dY,
    const Tensor X,
    const Tensor mean,
    const Tensor rstd,
    const Tensor gamma,
    int64_t N,
    int64_t C,
    int64_t group,
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (dX.defined()) {
    const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
    const auto kAccType =
        (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
        ? kFloat
        : X.scalar_type();
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
    T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();
    const int64_t num_threads = (C / G) < cuda_utils::kCUDABlockReduceNumThreads
        ? at::cuda::warp_size()
        : cuda_utils::kCUDABlockReduceNumThreads;
    Compute1dBackwardFusedParamsCUDAKernel<T>
        <<<dim3(N, G), num_threads, 0, cuda_stream>>>(
            C,
            G,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            gamma_data,
            c2_data,
            c3_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N, G, D}))
                      .add_owned_const_input(dY.view({N, G, D}))
                      .add_owned_const_input(X.view({N, G, D}))
                      .add_owned_const_input(rstd.view({N, G, 1}))
                      .add_owned_const_input(gamma.view({1, G, D}))
                      .add_owned_const_input(c2.view({N, G, 1}))
                      .add_owned_const_input(c3.view({N, G, 1}))
                      .build();
      gpu_kernel(
          iter,
          [] GPU_LAMBDA(T dy, T x, T rstd, T gamma, T_ACC c2, T_ACC c3) -> T {
            const T_ACC c1 =
                static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
            return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                c3;
          });
    } else {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D}))
                      .add_owned_const_input(dY.view({N * G, D}))
                      .add_owned_const_input(X.view({N * G, D}))
                      .add_owned_const_input(rstd.view({N * G, 1}))
                      .add_owned_const_input(c2.view({N * G, 1}))
                      .add_owned_const_input(c3.view({N * G, 1}))
                      .build();
      gpu_kernel(
          iter, [] GPU_LAMBDA(T dy, T x, T rstd, T_ACC c2, T_ACC c3) -> T {
            const T_ACC c1 = static_cast<T_ACC>(rstd);
            return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                c3;
          });
    }
  }
  if (dgamma.defined() || dbeta.defined()) {
    T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
    if (N <= 128) {
      const int64_t B = (C + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBeta1dBackwardCUDAKernel1<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
          N,
          C,
          G,
          dY_data,
          X_data,
          mean_data,
          rstd_data,
          dgamma_data,
          dbeta_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
      // The algorithm for colwise reduction here is to accumulate each 32 cols
      // to a 32 * 32 tile and write the tile to shared memory. Then do warp
      // reduce for each col in the tile. So here the blockDim must be (32, 16).
      constexpr int kThreadX = kReduceTileSize;
      constexpr int kThreadY = kReduceTileSize / 2;
      GammaBeta1dBackwardCUDAKernel2<T>
          <<<B, dim3(kThreadX, kThreadY), 0, cuda_stream>>>(
              N,
              C,
              G,
              dY_data,
              X_data,
              mean_data,
              rstd_data,
              dgamma_data,
              dbeta_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
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
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
  using T_ACC = acc_type<T, true>;
  const int64_t G = group;
  const int64_t D = C / G;
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * G);
  TORCH_CHECK(rstd.numel() == N * G);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (N == 0) {
    if (dgamma.defined()) {
      dgamma.fill_(T(0));
    }
    if (dbeta.defined()) {
      dbeta.fill_(T(0));
    }
    return;
  }

  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* ds_data = ds.mutable_data_ptr<T_ACC>();
  T_ACC* db_data = db.mutable_data_ptr<T_ACC>();

  if (HxW == 1) {
    GroupNorm1dBackward<T>(
        dY, X, mean, rstd, gamma, N, C, G, dX, dgamma, dbeta);
    return;
  }

  int warp_size = at::cuda::warp_size();
  int64_t num_threads = HxW < cuda_utils::kCUDABlockReduceNumThreads
      ? warp_size
      : cuda_utils::kCUDABlockReduceNumThreads;
  ComputeInternalGradientsCUDAKernel<T><<<N * C, num_threads, 0, cuda_stream>>>(
      HxW, dY_data, X_data, ds_data, db_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (dX.defined()) {
    Tensor c1 = at::empty({0}, X.options().dtype(kAccType));
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
    T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .add_output(c1)
                      .add_owned_const_input(rstd.view({N, G, 1}))
                      .add_owned_const_input(gamma.view({1, G, D}))
                      .build();
      gpu_kernel(iter, [] GPU_LAMBDA(T rstd, T gamma) -> T_ACC {
        return static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
      });
    }

    num_threads = (C / G) < cuda_utils::kCUDABlockReduceNumThreads
        ? warp_size
        : cuda_utils::kCUDABlockReduceNumThreads;
    ComputeBackwardFusedParamsCUDAKernel<T>
        <<<dim3(N, G), num_threads, 0, cuda_stream>>>(
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (gamma.defined()) {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D, HxW}))
                      .add_owned_const_input(dY.view({N * G, D, HxW}))
                      .add_owned_const_input(X.view({N * G, D, HxW}))
                      .add_owned_const_input(c1.view({N * G, D, 1}))
                      .add_owned_const_input(c2.view({N * G, 1, 1}))
                      .add_owned_const_input(c3.view({N * G, 1, 1}))
                      .build();
      gpu_kernel(
          iter, [] GPU_LAMBDA(T dy, T x, T_ACC c1, T_ACC c2, T_ACC c3) -> T {
            return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                c3;
          });
    } else {
      auto iter = TensorIteratorConfig()
                      .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                      .resize_outputs(false)
                      .add_owned_output(dX.view({N * G, D * HxW}))
                      .add_owned_const_input(dY.view({N * G, D * HxW}))
                      .add_owned_const_input(X.view({N * G, D * HxW}))
                      .add_owned_const_input(rstd.view({N * G, 1}))
                      .add_owned_const_input(c2.view({N * G, 1}))
                      .add_owned_const_input(c3.view({N * G, 1}))
                      .build();
      gpu_kernel(
          iter, [] GPU_LAMBDA(T dy, T x, T_ACC c1, T_ACC c2, T_ACC c3) -> T {
            return c1 * static_cast<T_ACC>(dy) + c2 * static_cast<T_ACC>(x) +
                c3;
          });
    }
  }
  if (dgamma.defined() || dbeta.defined()) {
    T* dgamma_data = dgamma.defined() ? dgamma.mutable_data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta.defined() ? dbeta.mutable_data_ptr<T>() : nullptr;
    if (N <= 128) {
      // For small batch size, do colwise reduce directly.
      const int64_t B = (C + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBetaBackwardCUDAKernel1<T><<<B, kCUDANumThreads, 0, cuda_stream>>>(
          N,
          C,
          G,
          mean_data,
          rstd_data,
          ds_data,
          db_data,
          dgamma_data,
          dbeta_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      const int64_t B = (C + kReduceTileSize - 1) / kReduceTileSize;
      // The algorithm for colwise reduction here is to accumulate each 32 cols
      // to a 32 * 32 tile and write the tile to shared memory. Then do warp
      // reduce for each col in the tile. So here the blockDim must be (32, 16).
      constexpr int kThreadX = kReduceTileSize;
      constexpr int kThreadY = kReduceTileSize / 2;
      GammaBetaBackwardCUDAKernel2<T>
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
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
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
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "GroupNormBackwardKernelImpl",
      [&]() {
        GroupNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
      });
}

} // namespace

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl)
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl)

} // namespace at::native

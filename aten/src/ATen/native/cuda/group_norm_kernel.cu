#include "c10/core/ScalarType.h"
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/group_norm.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/MemoryFormat.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <thrust/tuple.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/ceil_div.h>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

namespace {

constexpr int kCUDANumThreads = 256;
constexpr int kReduceTileSize = 32;

typedef struct BlockParams {
  int64_t t; // threads per block
  int64_t d; // dimensionality (number of rows of data that each threadblock proceesses in parallel)
  int64_t f; // factor (number of different threadblocks needed to represent one row of data)
} BlockParams_t;

inline BlockParams_t CalcBlockParams(const int64_t ideal_num_threads, const int64_t threads_per_row, const int64_t snap = -1) {
  /*
  ideal_num_threads: absolute upper limit of threads that a block should have (e.g. a kernel that operates on only 30 elements should have a max TPB of 30 (ideal_num_threads=30))
  threads_per_row: determines the user-specified upper limit on the size of blockDim.x
    - meant to be set to the size of the last dimension, e.g. a kernel operating on tensor sized (N, R, C) would have threads_per_row=C
  snap: an optional constraint for threads per block. If set, the returned TPB % snap = 0 or snap % TPB = 0.
    - ex: D=1280, C=2560 -> threads_per_row=2560 -> f=5, TPB=512 (each group consists of 1280/512=2.5 blocks - will have to deal with nasty padding)
      - ex: D=1280, C=2560 -> threads_per_row=2560, snap=1280 -> f=8, TPB=320 (each group consists of exactly four blocks)
  */
  int64_t TPB = -1, d = 1, f = 1;
  TPB = std::min(kCUDANumThreads, (int)ideal_num_threads);
  if (threads_per_row < TPB)
    d = TPB / threads_per_row;
  else {
    f = ceil_div(threads_per_row, TPB); // lower bound for f
    TPB = ceil_div(threads_per_row * d, f);
    while (TPB % snap != 0 && snap % TPB != 0)
      TPB = ceil_div(threads_per_row * d, ++f);
  }
  TPB = ceil_div(threads_per_row * d, f);
  return {TPB, d, f};
}

__device__ int inline NextPow2(unsigned int x) {
  // Return the closest power of 2 greater than x (if x is 0 or a power of 2, return x).
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return (int)x;
}

int64_t ClosestFactor(int64_t n) {
  // finds the largest factor of n that is less than or equal to the square root of n
  int64_t factor = 1;
  for (int64_t i = 1; i * i <= n; i++)
    if (n % i == 0)
      factor = i;
  return factor;
}

template <typename T>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t group_span,
    T eps,
    const T* X,
    T* mean,
    T* rstd,
    int64_t C) {
  using T_ACC = acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;
  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int64_t j = threadIdx.x; j < group_span; j += blockDim.x) {
    const int64_t index = i * group_span + j;
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
__global__ void RowwiseMomentsCUDAKernelNHWC1(
    const int64_t C,
    const int64_t H,
    const int64_t W,
    const int64_t G,
    const T *X,
    WelfordData<acc_type<T, true>, int64_t> *welford_data) {
  /*
    Computes means and rstds of X on the W (width) dimension.
    grid: (x=N, y=H, z=f); block: (x=TPB/d, y=d)
    - TPB = Cd/f
    if TPB < C (f > 1, d=1)
      TPB = ceil(C/f) (aka f*TPB >= C)
      X shape: (N, R, C) -view-> (N, H, W, C) -view-> (N, H, W, 1, f, TPB); X stride: (HWC, WC, C, C, TPB, 1)
      dram reduction (per block): (W, 1, TPB) -reduce-> (1, TPB)
    else (block.x=C, block.y=d)
      TPB = Cd
      X shape: (N, H, W, C) -view-> (N, H, W/d, d, 1, C); X stride: (HWC, WC, dC, C, C, 1)
      dram reduction (per block): (W/d, d, C) -reduce-> (d, C)
    shmem reduction (per block):
      if G/f >= 1
        (TPB,) -view-> (d, G/f, D) -permute-> (d, D, G/f) -reduce-> G/f
        output buffer: (N, G, H)
      else (e.g. f/G > 1 aka more than one thread-block reduces one group)
        snap constraints require that D % TPB = 0 in this case so f/G = CDIV(f, G)
        (TPB,) -view-> (1, 1, D) -permute-> (1, D, 1) -reduce-> 1
        output buffer: (N, f*CDIV(G/f), H) = (N, f, H)
  */
  using T_ACC = acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;

  const int64_t TPB = (int64_t)(blockDim.y * blockDim.x);
  const int64_t c = (blockIdx.z * blockDim.x + threadIdx.x);
  const int64_t d = blockDim.y;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  if (c >= C) return;
  for (int64_t i = 0; i < ceil_div(W, d); ++i) {
    int64_t w = i * d + threadIdx.y;
    if (w >= W) continue; // handle indices which overflow width
    int64_t reduce_idx = 0;
    reduce_idx += (int64_t)blockIdx.x * H * W * C;
    reduce_idx += (int64_t)blockIdx.y * W * C;
    reduce_idx += w * C;
    reduce_idx += (int64_t)blockIdx.z * TPB;
    reduce_idx += threadIdx.x;
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx);
  }

  // shmem reduction
  const int64_t tid = (threadIdx.y * blockDim.x + threadIdx.x);
  const int64_t D = C / G;
  const int64_t f = gridDim.z;
  const int64_t gf = (int64_t)ceil_div(G, f); // cdiv in case G < f -> ceil(G/f) = 1
  const int64_t d_idx = threadIdx.y;
  const int64_t gf_idx = threadIdx.x / D;
  const int64_t D_idx = threadIdx.x % D;
  const int64_t reduce_n = TPB / gf; // number of inputs that gets reduced to a single output

  __shared__ typename std::aligned_storage<
        sizeof(WelfordType),
    alignof(WelfordType)>::type vals_reduced_arr[kCUDANumThreads];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  int64_t idx = 0;
  idx += d_idx * D * gf;
  idx += D_idx * gf;
  idx += gf_idx;
  vals_reduced[idx] = val;
  __syncthreads();

  for (int64_t stride = gf * NextPow2(reduce_n) / 2; stride >= gf; stride >>= 1) {
    if (tid < stride && (tid + stride) < TPB)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  // put reduced outputs into return buffers
  const int64_t fgf = (gf > 1) ? G : f;
  const int64_t fgf_idx = blockIdx.z * gf + gf_idx;
  if (fgf_idx >= fgf) return;
  int64_t out_idx = 0;
  out_idx += blockIdx.x * fgf * H;
  out_idx += fgf_idx * H;
  out_idx += blockIdx.y;
  welford_data[out_idx] = vals_reduced[gf_idx];
}

template <typename T>
__global__ void RowwiseMomentsCUDAKernelNHWC2(
    const int64_t H,
    const int64_t G,
    const int64_t fg,
    const T eps,
    WelfordData<acc_type<T, true>, int64_t> *welford_data,
    T *mean,
    T *rstd) {
  /*
    Computes means and rstds of X on the H (height) dimension.
    grid: (x=N, y=G); block: (x=TPB)
    - l = ceil(f/G) * H / TPB (l = number of times to loop a block to reduce the H dimension as well as any partially reduced group sections)
    if G/f (from RowwiseMomentsCUDAKernelNHWC_pt1) > 1
      welford_data shape: (N, G, H) -view-> (N, G, f, H/f); X stride: (GH, H, H/f, 1)
    else (i.e. f/G > 1)
      welford_data shape: (N, f, H) -view-> (N, G, f/G, H/f); X stride: (GH*gf, gf*H, gf*H/f, 1)
    dram reduction (per block): (CDIV(fg*H, TPB), TPB) -reduce-> (TPB,)
    shmem reduction (per block): (TPB,) -reduce-> (1,)
    output buffer: (N, G)
  */
  using T_ACC = acc_type<T, true>;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t,
        thrust::pair<T_ACC, T_ACC>>;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  const int64_t TPB = blockDim.x;

  const int64_t R = fg * H; // R for num "R"educe elements per block
  const int64_t l = ceil_div(R, TPB);
  for (int64_t i = 0; i < l; ++i) {
    int64_t r = i * TPB + threadIdx.x;
    if (r >= R) continue;
    int64_t idx = 0;
    idx += blockIdx.x * G * R;
    idx += blockIdx.y * R;
    idx += r;
    val = welford_op.combine(val, welford_data[idx]);
  }

  // shmem reduction
  __shared__ typename std::aligned_storage<
    sizeof(WelfordType),
    alignof(WelfordType)>::type vals_reduced_arr[kCUDANumThreads];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int64_t tid = threadIdx.x;
  vals_reduced[tid] = val;
  __syncthreads();

  for (int64_t stride = NextPow2(TPB) / 2; stride >= 1; stride >>= 1) {
    if (tid < stride && tid + stride < TPB)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
  }

  // put reduced outputs into return buffers
  if (tid != 0) return;
  thrust::pair<T_ACC, T_ACC> var_mean = welford_op.project(vals_reduced[tid]);
  T_ACC var = var_mean.first;
  T_ACC mu = var_mean.second;
  int64_t out_idx = 0;
  out_idx += blockIdx.x * G;
  out_idx += blockIdx.y;
  mean[out_idx] = mu;
  rstd[out_idx] = rsqrt(var + static_cast<T_ACC>(eps));
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
  const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
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
  const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
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
    for (int64_t n = threadIdx.y; n < N; n += (int64_t)blockDim.y * 2) {
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
__device__ void SumReduce(
    T vals_reduced,
    const int start_size,
    const int end_size) {
  // Sums a shared buffer (vals_reduced) containing start_size values (shape (reduce_n, end_size)) into (end_size,)
  // TODO: replace with a torch op (I couldn't find one)
  const int tid = (int)(threadIdx.y * blockDim.x + threadIdx.x);
  const int reduce_n = start_size / end_size;

  for (int stride = end_size * NextPow2(reduce_n) / 2; stride >= end_size; stride >>= 1) {
    if (tid < stride && tid + stride < start_size)
      vals_reduced[tid] += vals_reduced[tid + stride];
    __syncthreads();
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernelNHWC1(
    const int64_t C,
    const int64_t H,
    const int64_t W,
    const T* dY,
    const T* X,
    acc_type<T, true> *xdy_dy_sum_data) {
  /*
    Loops over W (width) dimension, loading and summing dy, X, and the activation derivative of Y. Outputs stored in xdy_dy_sum_data. Spatial dimension H is processed in a separate kernel.
    grid: (x=N, y=H, z=f); blockdim: (x=TPB/d, y=d)
      TPB = Cd/f
    if TPB < C (f > 1, d=1)
      C = f*TPB
      X shape: (N, H, W, C) -view-> (N, H, W, 1, f, TPB); X stride: (HWC, WC, C, C, TPB, 1)
      dram reduction (per block): (W, 1, TPB) -reduce-> (TPB,)
    else (block.x=C, block.y=d)
      TPB = Cd
      X shape: (N, H, W, C) -view-> (N, H, W/d, d, 1, C); X stride: (HWC, WC, dC, C, C, 1)
      dram reduction (per block): (W/d, d, C) -reduce-> (d, C)
    shmem reduction (per block): (TPB, 2) -> (d, C/f, 2) -reduce-> (C/f, 2) (the 2 comes from storing both xdy_sum and dy_sum in the same buffer)
    output buffer: (N, f, C/f, H, 2) -view-> (N, C, H, 2)
      xdy_dy_sum_data[:, :, :, 0] = x * dy * activation_derivative((x-mean)*rstd*weight+bias)
      xdy_dy_sum_data[:, :, :, 1] = dy * activation_derivative((x-mean)*rstd*weight+bias)
   */
  using T_ACC = acc_type<T, true>;

  const int64_t TPB = (int64_t)blockDim.y * blockDim.x;
  const int64_t d = blockDim.y;
  T_ACC xdy_sum = 0;
  T_ACC dy_sum = 0;

  const int64_t n = blockIdx.x;
  int64_t c = blockIdx.z * blockDim.x + threadIdx.x;
  if (c >= C) return;

  for (int64_t i = 0; i < (int64_t)ceil_div(W, d); ++i) {
    int64_t w = i * d + threadIdx.y;
    if (w >= W) continue; // handle overflowing indices
    int64_t reduce_idx = 0;
    reduce_idx += n * H * W * C;
    reduce_idx += blockIdx.y * W * C;
    reduce_idx += w * C;
    reduce_idx += blockIdx.z * TPB;
    reduce_idx += threadIdx.x;
    T_ACC dy_elem = static_cast<T_ACC>(dY[reduce_idx]);
    T_ACC X_elem = static_cast<T_ACC>(X[reduce_idx]);
    xdy_sum += dy_elem * X_elem;
    dy_sum += dy_elem;
  }

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[]; // size 2*TPB, TPB for sum1, TPB for sum2
  T_ACC *vals_reduced = reinterpret_cast<T_ACC*>(vals_reduced_uncasted);

  const int64_t tid = threadIdx.y * blockDim.x + threadIdx.x;

  if (TPB > C) {
    vals_reduced[2 * tid] = xdy_sum;
    vals_reduced[2 * tid + 1] = dy_sum;
    __syncthreads();
    SumReduce(vals_reduced, 2 * TPB, 2 * C); // does nothing if d=1
    xdy_sum = vals_reduced[2 * tid];
    dy_sum = vals_reduced[2 * tid + 1];
  }

  // put reduced outputs into return buffers
  if (threadIdx.y != 0) return;
  int64_t out_idx = 0;
  out_idx += n * C * H;
  out_idx += c * H;
  out_idx += blockIdx.y;

  xdy_dy_sum_data[2 * out_idx] = xdy_sum;
  xdy_dy_sum_data[2 * out_idx + 1] = dy_sum;
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernelNHWC2(
    const int H,
    const int C,
    T *xdy_dy_sum_data, // no need to specify T_ACC as T is already an accumulation type
    T *ds_data,
    T *db_data) {
  /*
    Same thing as ComputeInternalGradientsCUDAKernelNHWC1 but over the H (height) instead of the width dimension.
    grid: (x=N, y=C); block: (x=2H/f)
    X shape: (N, C, H, 2) -view-> (N, C, f, H/f, 2); X stride: (2CH, 2H, 2H/f, H/f, 1)
    dram reduction (per block): (f, H/f, 2) -reduce-> (H/f, 2)
    shmem reduction (per block): (H/f, 2) -reduce-> (2,)
    output buffer: (N, C, 2)
   */
  const int TPB = (int)blockDim.x;
  const int tid = (int)threadIdx.x;

  // shmem reduction
  extern __shared__ char vals_reduced_uncasted[];
  T *vals_reduced = reinterpret_cast<T*>(vals_reduced_uncasted);

  T sum = 0;
  for (int i = 0; i < ceil_div(2 * H, TPB); ++i) {
    const int h = i * TPB + tid;
    if (h >= 2 * H) continue;
    int64_t idx = 0;
    idx += (int64_t)blockIdx.x * C * H * 2;
    idx += (int64_t)blockIdx.y * H * 2;
    idx += h;
    sum += xdy_dy_sum_data[idx];
  }

  vals_reduced[tid] = sum;
  __syncthreads();
  SumReduce(vals_reduced, TPB, 2);

  // put reduced outputs into return buffers
  if (tid != 0) return;
  int64_t out_idx = blockIdx.x * C + blockIdx.y;
  ds_data[out_idx] = vals_reduced[0];
  db_data[out_idx] = vals_reduced[1];
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
  const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
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

  // Do warp reduce for the 2st 16 cols in the tile.
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

  at::MemoryFormat x_format = X.suggest_memory_format();
  Y.is_contiguous(x_format);

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = D * HxW < cuda_utils::kCUDABlockReduceNumThreads
      ? at::cuda::warp_size()
      : cuda_utils::kCUDABlockReduceNumThreads;

  switch (x_format) {
    case MemoryFormat::Contiguous: {
      RowwiseMomentsCUDAKernel<T><<<N * G, num_threads, 0, cuda_stream>>>(
        D * HxW, eps, X_data, mean_data, rstd_data, C);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    case MemoryFormat::ChannelsLast3d:
    case MemoryFormat::ChannelsLast: {
      const int64_t H = ClosestFactor(HxW);
      const int64_t W = HxW / H;
      auto [TPB, d, f1] = CalcBlockParams(W * C, C, C / G);
      const int64_t gf = ceil_div(G, f1); // number of groups processed per block in RowwiseMomentsCUDAKernelNHWC_pt1, needed here because it determines size of welford_data
      const int64_t fgf = (gf == 1) ? f1 : G; // f1 * gf but in case gf > 1, return G (e.g. G=1031, f1=6, gf=172, f1*gf=1032 != 1031; if fgf > G, we know for certain it is because each group needs multiple blocks)

      using WelfordType = WelfordData<T_ACC, int64_t>;

      WelfordType *welford_data = reinterpret_cast<WelfordType*>(
        at::empty(
          {N, fgf, H, sizeof(WelfordType)},
          X.options().dtype(at::ScalarType::Byte)
        ).data_ptr<uint8_t>()
      );

      // compute means/rstds over width dimension
      {
        auto [TPB, d, f] = CalcBlockParams(W * C, C, C / G); // same fn + args as the one a couple lines up but repeated for clarity
        RowwiseMomentsCUDAKernelNHWC1<T><<<dim3(N, H, f), dim3(TPB / d, d), 0, cuda_stream>>>(
          C, H, W, G,
          X_data,
          welford_data
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }

      // compute means/rstds over height dimension
      {
        const int64_t fg = ceil_div(f1, G); // number of blocks to process one group
        auto [TPB, d, f] = CalcBlockParams(fg * H, fg * H);
        RowwiseMomentsCUDAKernelNHWC2<T><<<dim3(N, G), TPB, 0, cuda_stream>>>(
          H, G, fg, eps,
          welford_data,
          mean_data, rstd_data
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      break;
    }
    default:
      TORCH_CHECK(
        false,
        "Unsupported memory format for group normalization: ",
        x_format);
      break;
  }

  if (HxW == 1) {
    GroupNorm1dForward<T>(X, mean, rstd, gamma, beta, N, C, G, Y);
  }
  else if (!gamma.defined() && !beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D, HxW}))
                    .add_owned_const_input(X.view({N, G, D, HxW}))
                    .add_owned_input(mean.view({N, G, 1}))
                    .add_owned_input(rstd.view({N, G, 1}))
                    .build();

    gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) *
          static_cast<T_ACC>(rstd);
    });
  }
  else {
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

    TensorIterator iter =
        TensorIteratorConfig()
            .check_all_same_dtype(std::is_same_v<T, T_ACC>)
            .resize_outputs(false)
            .add_owned_output(Y.view({N, C, HxW}))
            .add_owned_const_input(X.view({N, C, HxW}))
            .add_owned_input(a.view({N, C, 1}))
            .add_owned_input(b.view({N, C, 1}))
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

  at::MemoryFormat x_format = X.suggest_memory_format();
  dY.is_contiguous(x_format);

  int warp_size = at::cuda::warp_size();
  int64_t num_threads = HxW < cuda_utils::kCUDABlockReduceNumThreads
      ? warp_size
      : cuda_utils::kCUDABlockReduceNumThreads;

  switch (x_format) {
    case MemoryFormat::Contiguous: {
      ComputeInternalGradientsCUDAKernel<T><<<N * C, num_threads, 0, cuda_stream>>>(
        HxW, dY_data, X_data, ds_data, db_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    case MemoryFormat::ChannelsLast3d:
    case MemoryFormat::ChannelsLast: {
      const int64_t H = ClosestFactor(HxW);
      const int64_t W = HxW / H;
      T_ACC* xdy_dy_sum_data = at::empty(
        {N, C, H, 2},
        X.options().dtype(kAccType)).mutable_data_ptr<T_ACC>();

      // sum over width dimension
      {
        auto [TPB, d, f] = CalcBlockParams(W * C, C);
        ComputeInternalGradientsCUDAKernelNHWC1<T>
          <<<dim3(N, H, f), dim3(TPB / d, d), sizeof(T_ACC) * 2*TPB, cuda_stream>>>(
            C, H, W, dY_data, X_data, xdy_dy_sum_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }

      // sum over height dimension
      {
        auto [TPB, d, f] = CalcBlockParams(2 * H, 2);
        ComputeInternalGradientsCUDAKernelNHWC2<T_ACC>
          <<<dim3(N, C), TPB, sizeof(T_ACC) * TPB, cuda_stream>>>(
            H, C, xdy_dy_sum_data,
            ds_data, db_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      break;
    }
    default:
      TORCH_CHECK(
        false,
        "Unsupported memory format for group normalization backward: ",
        x_format);
      break;
  }

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
                      .add_owned_output(dX.view({N, G, D, HxW}))
                      .add_owned_const_input(dY.view({N, G, D, HxW}))
                      .add_owned_const_input(X.view({N, G, D, HxW}))
                      .add_owned_const_input(c1.view({N, G, D, 1}))
                      .add_owned_const_input(c2.view({N, G, 1, 1}))
                      .add_owned_const_input(c3.view({N, G, 1, 1}))
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
                      .add_owned_output(dX.view({N, G, D, HxW}))
                      .add_owned_const_input(dY.view({N, G, D, HxW}))
                      .add_owned_const_input(X.view({N, G, D, HxW}))
                      .add_owned_const_input(rstd.view({N, G, 1, 1}))
                      .add_owned_const_input(c2.view({N, G, 1, 1}))
                      .add_owned_const_input(c3.view({N, G, 1, 1}))
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

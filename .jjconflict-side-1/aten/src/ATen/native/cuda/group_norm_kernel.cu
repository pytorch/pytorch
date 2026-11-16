#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/group_norm.h>

#include <type_traits>
#include <utility>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/IntegerDivider.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
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

// Reduce across exactly 32 lanes (offsets 16, 8, 4, 2, 1).
// On NVIDIA (warp=32) this is identical to WarpReduceSum.
// On AMD (wavefront=64) this avoids summing across two tile columns
// when the block is (32, 16) and consecutive y-rows share a wavefront.
template <typename T>
__inline__ __device__ T ReduceSum32(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <int VecSize, typename T, typename T_ACC>
__global__ void RowwiseMomentsContiguousCUDAKernel(
    int64_t N,
    T eps,
    const T* __restrict__ X,
    T* __restrict__ mean,
    T* __restrict__ rstd,
    T_ACC* __restrict__ mean_acc,
    T_ACC* __restrict__ rstd_acc) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  using LoadT = memory::aligned_vector<T, VecSize>;
  const auto* X_vec = reinterpret_cast<const LoadT*>(X + i * N);
  const int64_t N_vec = N / VecSize;
  for (int64_t j = threadIdx.x; j < N_vec; j += blockDim.x) {
    const LoadT values = X_vec[j];
#pragma unroll
    for (int k = 0; k < VecSize; ++k) {
      val = welford_op.reduce(
          val, static_cast<T_ACC>(values.val[k]), j * VecSize + k);
    }
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    val = cuda_utils::WarpReduce(val, welford_op);
  } else {
    // Use a byte array with alignas instead of a __shared__ WelfordType array
    // directly, because nvcc warns on non-trivial constructors in __shared__.
    // alignas must precede __shared__; HIP's clang rejects it placed between
    // __shared__ and the type.
    alignas(WelfordType) __shared__ char
        val_shared[sizeof(WelfordType) * C10_WARP_SIZE_UPPER_BOUND];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = cuda_utils::BlockReduce(
        val,
        welford_op,
        /*identity_element=*/WelfordType(0, 0, 0, 0),
        val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    auto [m2, m1] = welford_op.project(val);
    T_ACC rstd_val = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
    mean[i] = m1;
    rstd[i] = rstd_val;
    // save off the accelerated-precision output, if different
    if constexpr (!std::is_same_v<T, T_ACC>) {
      mean_acc[i] = m1;
      rstd_acc[i] = rstd_val;
    }
  }
}

template <int GroupsPerBlock, typename T, typename T_ACC>
__global__ void RowwiseMomentsChannelsLastSmallDCUDAKernel(
    int64_t C,
    int64_t HxW,
    int64_t D,
    int64_t G,
    T eps,
    const T* __restrict__ X,
    T* __restrict__ mean,
    T* __restrict__ rstd,
    T_ACC* __restrict__ mean_acc,
    T_ACC* __restrict__ rstd_acc) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  const int64_t group_blocks = (G + GroupsPerBlock - 1) / GroupsPerBlock;
  const int64_t n = blockIdx.x / group_blocks;
  const int64_t first_group = (blockIdx.x % group_blocks) * GroupsPerBlock;
  const int64_t channels_per_block = GroupsPerBlock * D;
  const int64_t lane = threadIdx.x % channels_per_block;
  const int64_t hw_lane = threadIdx.x / channels_per_block;
  const int64_t hw_step = blockDim.x / channels_per_block;
  const int64_t local_group = lane / D;
  const int64_t g = first_group + local_group;
  const int64_t c = g * D + lane % D;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  if (g < G) {
    for (int64_t hw = hw_lane; hw < HxW; hw += hw_step) {
      const int64_t index = (n * HxW + hw) * C + c;
      val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
    }
  }

  alignas(WelfordType)
      __shared__ char val_shared[sizeof(WelfordType) * kCUDANumThreads];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  val_shared_ptr[threadIdx.x] = val;
  __syncthreads();

  if (threadIdx.x < GroupsPerBlock && first_group + threadIdx.x < G) {
    val = WelfordType(0, 0, 0, 0);
    for (int64_t i = 0; i < hw_step; ++i) {
      const int64_t base = i * channels_per_block + threadIdx.x * D;
      for (int64_t j = 0; j < D; ++j) {
        val = welford_op.combine(val, val_shared_ptr[base + j]);
      }
    }
    const int64_t ng = n * G + first_group + threadIdx.x;
    auto [m2, m1] = welford_op.project(val);
    const T_ACC rstd_val =
        c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
    mean[ng] = m1;
    rstd[ng] = rstd_val;
    if constexpr (!std::is_same_v<T, T_ACC>) {
      mean_acc[ng] = m1;
      rstd_acc[ng] = rstd_val;
    }
  }
}

template <typename T, typename T_ACC>
__global__ void RowwiseMomentsChannelsLastCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t D,
    T eps,
    const T* X,
    T* mean,
    T* rstd,
    T_ACC* mean_acc,
    T_ACC* rstd_acc) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;
  const int64_t ng = blockIdx.x;
  const int64_t n = ng / (C / D);
  const int64_t g = ng % (C / D);
  const int64_t group_size = D * HxW;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  int64_t hw = threadIdx.x / D;
  int64_t c = g * D + threadIdx.x % D;
  const int64_t hw_step = blockDim.x / D;
  const int64_t c_step = blockDim.x % D;
  for (int64_t j = threadIdx.x; j < group_size; j += blockDim.x) {
    const int64_t index = (n * HxW + hw) * C + c;
    val = welford_op.reduce(
        val, static_cast<T_ACC>(X[index]), ng * group_size + j);
    hw += hw_step;
    c += c_step;
    if (c >= (g + 1) * D) {
      c -= D;
      ++hw;
    }
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    val = cuda_utils::WarpReduce(val, welford_op);
  } else {
    alignas(WelfordType) __shared__ char
        val_shared[sizeof(WelfordType) * C10_WARP_SIZE_UPPER_BOUND];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = cuda_utils::BlockReduce(
        val,
        welford_op,
        /*identity_element=*/WelfordType(0, 0, 0, 0),
        val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    auto [m2, m1] = welford_op.project(val);
    T_ACC rstd_val = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
    mean[ng] = m1;
    rstd[ng] = rstd_val;
    if constexpr (!std::is_same_v<T, T_ACC>) {
      mean_acc[ng] = m1;
      rstd_acc[ng] = rstd_val;
    }
  }
}

template <typename index_t, typename T, typename T_ACC>
__global__ void GroupNormBackwardChannelsLastCUDAKernel(
    index_t numel,
    index_t G,
    at::cuda::detail::IntDivider<index_t> C_divider,
    at::cuda::detail::IntDivider<index_t> HxW_divider,
    at::cuda::detail::IntDivider<index_t> D_divider,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T* __restrict__ rstd,
    const T* __restrict__ gamma,
    const T_ACC* __restrict__ c2,
    const T_ACC* __restrict__ c3,
    T* __restrict__ dX) {
  for (index_t index =
           static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel;
       index += static_cast<index_t>(blockDim.x) * gridDim.x) {
    const auto nhw_c = C_divider.divmod(index);
    const index_t c = nhw_c.mod;
    const index_t n = HxW_divider.div(nhw_c.div);
    const index_t g = D_divider.div(c);
    const index_t ng = n * G + g;
    const T_ACC scale = static_cast<T_ACC>(rstd[ng]) *
        (gamma ? static_cast<T_ACC>(gamma[c]) : T_ACC(1));
    dX[index] = scale * static_cast<T_ACC>(dY[index]) +
        c2[ng] * static_cast<T_ACC>(X[index]) + c3[ng];
  }
}

template <typename T, typename T_ACC>
__global__ void ComputeFusedParamsCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    const T_ACC* mean,
    const T_ACC* rstd,
    const T* gamma,
    const T* beta,
    T_ACC* a,
    T_ACC* b) {
  const int64_t index = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < N * C) {
    const int64_t ng = index / (C / group);
    const int64_t c = index % C;
    const T_ACC scale =
        gamma ? rstd[ng] * static_cast<T_ACC>(gamma[c]) : rstd[ng];
    a[index] = scale;
    b[index] = -scale * mean[ng] + (beta ? static_cast<T_ACC>(beta[c]) : 0);
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
    const T_ACC gamma_v = gamma ? static_cast<T_ACC>(gamma[c]) : T_ACC(1);
    const T_ACC dY_acc = static_cast<T_ACC>(dY[index]);
    sum1 += dY_acc * static_cast<T_ACC>(X[index]) * gamma_v;
    sum2 += dY_acc * gamma_v;
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE_UPPER_BOUND];
    __shared__ T_ACC db_shared[C10_WARP_SIZE_UPPER_BOUND];
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
  const int64_t c = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
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
      sum1 += dgamma
          ? ((dy_acc * x_acc - dy_acc * static_cast<T_ACC>(mean[ng])) *
             static_cast<T_ACC>(rstd[ng]))
          : T_ACC(0);
      sum2 += dbeta ? dy_acc : T_ACC(0);
    }
    if (dgamma) {
      dgamma[c] = sum1;
    }
    if (dbeta) {
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
  const int64_t c = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
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
      dg_sum1 += dgamma
          ? ((dy1_acc * x1_acc - dy1_acc * static_cast<T_ACC>(mean[ng1])) *
             static_cast<T_ACC>(rstd[ng1]))
          : T_ACC(0);
      db_sum1 += dbeta ? dy1_acc : T_ACC(0);
      if (n2 < N) {
        const T_ACC dy2_acc = static_cast<T_ACC>(dY[nc2]);
        const T_ACC x2_acc = static_cast<T_ACC>(X[nc2]);
        dg_sum2 += dgamma
            ? ((dy2_acc * x2_acc - dy2_acc * static_cast<T_ACC>(mean[ng2])) *
               static_cast<T_ACC>(rstd[ng2]))
            : T_ACC(0);
        db_sum2 += dbeta ? dy2_acc : T_ACC(0);
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
  // Use ReduceSum32 (not WarpReduceSum) to reduce exactly 32 lanes.
  // On AMD wavefront-64, WarpReduceSum would incorrectly sum across two
  // tile columns since consecutive y-rows share a wavefront.
  sum1 = ReduceSum32<T_ACC>(sum1);
  sum2 = ReduceSum32<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y;
    if (c < C) {
      if (dgamma) {
        dgamma[c] = sum1;
      }
      if (dbeta) {
        dbeta[c] = sum2;
      }
    }
  }

  // Do warp reduce for the 2nd 16 cols in the tile.
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = ReduceSum32<T_ACC>(sum1);
  sum2 = ReduceSum32<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (c < C) {
      if (dgamma) {
        dgamma[c] = sum1;
      }
      if (dbeta) {
        dbeta[c] = sum2;
      }
    }
  }
}

template <int VecSize, typename T>
__global__ void ComputeInternalGradientsContiguousCUDAKernel(
    int64_t HxW,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    acc_type<T, true>* __restrict__ ds,
    acc_type<T, true>* __restrict__ db) {
  using T_ACC = acc_type<T, true>;
  const int64_t nc = blockIdx.x;
  using LoadT = memory::aligned_vector<T, VecSize>;
  const auto* dY_vec = reinterpret_cast<const LoadT*>(dY + nc * HxW);
  const auto* X_vec = reinterpret_cast<const LoadT*>(X + nc * HxW);
  const int64_t HxW_vec = HxW / VecSize;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t hw = threadIdx.x; hw < HxW_vec; hw += blockDim.x) {
    const LoadT dY_values = dY_vec[hw];
    const LoadT X_values = X_vec[hw];
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      const T_ACC dy = static_cast<T_ACC>(dY_values.val[i]);
      sum1 += dy * static_cast<T_ACC>(X_values.val[i]);
      sum2 += dy;
    }
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE_UPPER_BOUND];
    __shared__ T_ACC db_shared[C10_WARP_SIZE_UPPER_BOUND];
    sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    ds[nc] = sum1;
    db[nc] = sum2;
  }
}

template <int ChannelsPerBlock, typename T>
__global__ void ComputeInternalGradientsChannelsLastCUDAKernel(
    int64_t C,
    int64_t HxW,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    acc_type<T, true>* __restrict__ ds,
    acc_type<T, true>* __restrict__ db) {
  using T_ACC = acc_type<T, true>;
  const int64_t channel_blocks = (C + ChannelsPerBlock - 1) / ChannelsPerBlock;
  const int64_t n = blockIdx.x / channel_blocks;
  const int64_t first_channel =
      (blockIdx.x % channel_blocks) * ChannelsPerBlock;
  const int64_t local_channel = threadIdx.x % ChannelsPerBlock;
  const int64_t hw_lane = threadIdx.x / ChannelsPerBlock;
  const int64_t hw_step = blockDim.x / ChannelsPerBlock;
  const int64_t c = first_channel + local_channel;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  if (c < C) {
    for (int64_t hw = hw_lane; hw < HxW; hw += hw_step) {
      const int64_t index = (n * HxW + hw) * C + c;
      const T_ACC dy = static_cast<T_ACC>(dY[index]);
      sum1 += dy * static_cast<T_ACC>(X[index]);
      sum2 += dy;
    }
  }

  __shared__ T_ACC ds_shared[kCUDANumThreads];
  __shared__ T_ACC db_shared[kCUDANumThreads];
  ds_shared[threadIdx.x] = sum1;
  db_shared[threadIdx.x] = sum2;
  __syncthreads();
  if (threadIdx.x < ChannelsPerBlock && first_channel + threadIdx.x < C) {
    sum1 = 0;
    sum2 = 0;
    for (int64_t i = 0; i < hw_step; ++i) {
      const int64_t index = i * ChannelsPerBlock + threadIdx.x;
      sum1 += ds_shared[index];
      sum2 += db_shared[index];
    }
    const int64_t nc = n * C + first_channel + threadIdx.x;
    ds[nc] = sum1;
    db[nc] = sum2;
  }
}

template <typename T>
__global__ void ComputeInternalGradientsChannelsLastFallbackCUDAKernel(
    int64_t C,
    int64_t HxW,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    acc_type<T, true>* __restrict__ ds,
    acc_type<T, true>* __restrict__ db) {
  using T_ACC = acc_type<T, true>;
  const int64_t nc = blockIdx.x;
  const int64_t n = nc / C;
  const int64_t c = nc % C;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t hw = threadIdx.x; hw < HxW; hw += blockDim.x) {
    const int64_t index = (n * HxW + hw) * C + c;
    const T_ACC dy = static_cast<T_ACC>(dY[index]);
    sum1 += dy * static_cast<T_ACC>(X[index]);
    sum2 += dy;
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE_UPPER_BOUND];
    __shared__ T_ACC db_shared[C10_WARP_SIZE_UPPER_BOUND];
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
    const T_ACC gamma_v = gamma ? static_cast<T_ACC>(gamma[c]) : T_ACC(1);
    sum1 += ds[index] * gamma_v;
    sum2 += db[index] * gamma_v;
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE_UPPER_BOUND];
    __shared__ T_ACC db_shared[C10_WARP_SIZE_UPPER_BOUND];
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
  const int64_t c = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  if (c < C) {
    const int64_t G = group;
    const int64_t D = C / G;
    T_ACC sum1 = 0;
    T_ACC sum2 = 0;
    for (int64_t n = 0; n < N; ++n) {
      const int64_t nc = n * C + c;
      const int64_t ng = n * G + c / D;
      sum1 += dgamma ? ((ds[nc] - db[nc] * static_cast<T_ACC>(mean[ng])) *
                        static_cast<T_ACC>(rstd[ng]))
                     : T_ACC(0);
      sum2 += dbeta ? db[nc] : T_ACC(0);
    }
    if (dgamma) {
      dgamma[c] = sum1;
    }
    if (dbeta) {
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
  const int64_t c = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
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
      dg_sum1 += dgamma ? ((ds[nc1] - db[nc1] * static_cast<T_ACC>(mean[ng1])) *
                           static_cast<T_ACC>(rstd[ng1]))
                        : T_ACC(0);
      db_sum1 += dbeta ? db[nc1] : T_ACC(0);
      if (n2 < N) {
        dg_sum2 += dgamma
            ? ((ds[nc2] - db[nc2] * static_cast<T_ACC>(mean[ng2])) *
               static_cast<T_ACC>(rstd[ng2]))
            : T_ACC(0);
        db_sum2 += dbeta ? db[nc2] : T_ACC(0);
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
  // Use ReduceSum32 for correctness on AMD wavefront-64 (see above).
  T_ACC sum1 = g_shared[threadIdx.x][threadIdx.y];
  T_ACC sum2 = b_shared[threadIdx.x][threadIdx.y];
  sum1 = ReduceSum32<T_ACC>(sum1);
  sum2 = ReduceSum32<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y;
    if (c < C) {
      if (dgamma) {
        dgamma[c] = sum1;
      }
      if (dbeta) {
        dbeta[c] = sum2;
      }
    }
  }

  // Do warp reduce for the 2nd 16 cols in the tile.
  sum1 = g_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum2 = b_shared[threadIdx.x][threadIdx.y + blockDim.y];
  sum1 = ReduceSum32<T_ACC>(sum1);
  sum2 = ReduceSum32<T_ACC>(sum2);
  if (threadIdx.x == 0) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.y + blockDim.y;
    if (c < C) {
      if (dgamma) {
        dgamma[c] = sum1;
      }
      if (dbeta) {
        dbeta[c] = sum2;
      }
    }
  }
}

template <typename T, typename T_ACC>
void GroupNorm1dForward(
    const Tensor& X,
    const Tensor& mean_acc,
    const Tensor& rstd_acc,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t group,
    Tensor& Y) {
  const int64_t G = group;
  const int64_t D = C / G;
  if (gamma.defined() && beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean_acc.view({N, G, 1}))
                    .add_owned_input(rstd_acc.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(
        iter, [] GPU_LAMBDA(T x, T_ACC mean, T_ACC rstd, T gamma, T beta) -> T {
          return (static_cast<T_ACC>(x) - mean) * rstd *
              static_cast<T_ACC>(gamma) +
              static_cast<T_ACC>(beta);
        });
  } else if (gamma.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean_acc.view({N, G, 1}))
                    .add_owned_input(rstd_acc.view({N, G, 1}))
                    .add_owned_const_input(gamma.view({1, G, D}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC mean, T_ACC rstd, T gamma) -> T {
      return (static_cast<T_ACC>(x) - mean) * rstd * static_cast<T_ACC>(gamma);
    });
  } else if (beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N, G, D}))
                    .add_owned_const_input(X.view({N, G, D}))
                    .add_owned_input(mean_acc.view({N, G, 1}))
                    .add_owned_input(rstd_acc.view({N, G, 1}))
                    .add_owned_const_input(beta.view({1, G, D}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC mean, T_ACC rstd, T beta) -> T {
      return (static_cast<T_ACC>(x) - mean) * rstd + static_cast<T_ACC>(beta);
    });
  } else {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D}))
                    .add_owned_const_input(X.view({N * G, D}))
                    .add_owned_input(mean_acc.view({N * G, 1}))
                    .add_owned_input(rstd_acc.view({N * G, 1}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC mean, T_ACC rstd) -> T {
      return (static_cast<T_ACC>(x) - mean) * rstd;
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename T_ACC = acc_type<T, true>>
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
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);

  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.const_data_ptr<T>();

  const bool needMeanAcc{
      X.scalar_type() == kHalf || X.scalar_type() == kBFloat16};
  const auto kAccTypeOpts{
      X.options().dtype(needMeanAcc ? kFloat : X.scalar_type())};

  T* mean_data = mean.mutable_data_ptr<T>();
  T* rstd_data = rstd.mutable_data_ptr<T>();
  Tensor mean_acc = needMeanAcc ? at::empty(mean.sizes(), kAccTypeOpts) : mean;
  Tensor rstd_acc = needMeanAcc ? at::empty(rstd.sizes(), kAccTypeOpts) : rstd;
  T_ACC* mean_acc_data = mean_acc.mutable_data_ptr<T_ACC>();
  T_ACC* rstd_acc_data = rstd_acc.mutable_data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const bool channels_last = X.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      X.is_contiguous(at::MemoryFormat::ChannelsLast3d);
  const int64_t num_threads = D * HxW < cuda_utils::kCUDABlockReduceNumThreads
      ? at::cuda::warp_size()
      : cuda_utils::kCUDABlockReduceNumThreads;
  if (channels_last) {
    if ((D == 1 || D == 2) && G >= 4 &&
        D * HxW >= cuda_utils::kCUDABlockReduceNumThreads) {
      constexpr int kGroupsPerBlock = 4;
      const int64_t group_blocks = (G + kGroupsPerBlock - 1) / kGroupsPerBlock;
      RowwiseMomentsChannelsLastSmallDCUDAKernel<kGroupsPerBlock, T, T_ACC>
          <<<N * group_blocks, kCUDANumThreads, 0, cuda_stream>>>(
              C,
              HxW,
              D,
              G,
              eps,
              X_data,
              mean_data,
              rstd_data,
              mean_acc_data,
              rstd_acc_data);
    } else {
      RowwiseMomentsChannelsLastCUDAKernel<T, T_ACC>
          <<<N * G, num_threads, 0, cuda_stream>>>(
              N,
              C,
              HxW,
              D,
              eps,
              X_data,
              mean_data,
              rstd_data,
              mean_acc_data,
              rstd_acc_data);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    Tensor a = at::empty({N, C}, kAccTypeOpts);
    Tensor b = at::empty({N, C}, kAccTypeOpts);
    const int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
    ComputeFusedParamsCUDAKernel<T, T_ACC>
        <<<B, kCUDANumThreads, 0, cuda_stream>>>(
            N,
            C,
            G,
            mean_acc_data,
            rstd_acc_data,
            gamma.defined() ? gamma.const_data_ptr<T>() : nullptr,
            beta.defined() ? beta.const_data_ptr<T>() : nullptr,
            a.mutable_data_ptr<T_ACC>(),
            b.mutable_data_ptr<T_ACC>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto iter =
        TensorIteratorConfig()
            .check_all_same_dtype(std::is_same_v<T, T_ACC>)
            .resize_outputs(false)
            .add_owned_output(Y.as_strided({N, HxW, C}, {HxW * C, C, 1}))
            .add_owned_const_input(X.as_strided({N, HxW, C}, {HxW * C, C, 1}))
            .add_owned_const_input(a.view({N, 1, C}))
            .add_owned_const_input(b.view({N, 1, C}))
            .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
      return a * static_cast<T_ACC>(x) + b;
    });
    return;
  }
  constexpr int kVecSize = 16 / sizeof(T);
  if (D * HxW % kVecSize == 0 &&
      memory::can_vectorize_up_to<T>(reinterpret_cast<const char*>(X_data)) >=
          kVecSize) {
    RowwiseMomentsContiguousCUDAKernel<kVecSize, T, T_ACC>
        <<<N * G, num_threads, 0, cuda_stream>>>(
            D * HxW,
            eps,
            X_data,
            mean_data,
            rstd_data,
            mean_acc_data,
            rstd_acc_data);
  } else {
    RowwiseMomentsContiguousCUDAKernel<1, T, T_ACC>
        <<<N * G, num_threads, 0, cuda_stream>>>(
            D * HxW,
            eps,
            X_data,
            mean_data,
            rstd_data,
            mean_acc_data,
            rstd_acc_data);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (HxW == 1) {
    GroupNorm1dForward<T, T_ACC>(
        X, mean_acc, rstd_acc, gamma, beta, N, C, G, Y);
  } else if (!gamma.defined() && !beta.defined()) {
    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(std::is_same_v<T, T_ACC>)
                    .resize_outputs(false)
                    .add_owned_output(Y.view({N * G, D * HxW}))
                    .add_owned_const_input(X.view({N * G, D * HxW}))
                    .add_owned_input(mean_acc.view({N * G, 1}))
                    .add_owned_input(rstd_acc.view({N * G, 1}))
                    .build();
    gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC mean, T_ACC rstd) -> T {
      return (static_cast<T_ACC>(x) - mean) * rstd;
    });
  } else {
    Tensor a = at::empty({N, C}, kAccTypeOpts);
    Tensor b = at::empty({N, C}, kAccTypeOpts);
    const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
    const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
    T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
    T_ACC* b_data = b.mutable_data_ptr<T_ACC>();

    // TODO: Since there is some issues in gpu_kernel_multiple_outputs, we are
    // using manual kernel here. Make it using gpu_kernel_multiple_outputs once
    // the issue fixed.
    const int64_t B = (N * C + kCUDANumThreads - 1) / kCUDANumThreads;
    ComputeFusedParamsCUDAKernel<T, T_ACC>
        <<<B, kCUDANumThreads, 0, cuda_stream>>>(
            N,
            C,
            G,
            mean_acc_data,
            rstd_acc_data,
            gamma_data,
            beta_data,
            a_data,
            b_data);
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
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
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

  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const T* mean_data = mean.const_data_ptr<T>();
  const T* rstd_data = rstd.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const bool channels_last = X.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      X.is_contiguous(at::MemoryFormat::ChannelsLast3d);

  if (HxW == 1 && !channels_last) {
    GroupNorm1dBackward<T>(
        dY, X, mean, rstd, gamma, N, C, G, dX, dgamma, dbeta);
    return;
  }

  const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
  Tensor ds = at::empty({N, C}, X.options().dtype(kAccType));
  Tensor db = at::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* ds_data = ds.mutable_data_ptr<T_ACC>();
  T_ACC* db_data = db.mutable_data_ptr<T_ACC>();

  int warp_size = at::cuda::warp_size();
  int64_t num_threads = HxW < cuda_utils::kCUDABlockReduceNumThreads
      ? warp_size
      : cuda_utils::kCUDABlockReduceNumThreads;
  if (channels_last) {
    constexpr int kChannelsPerBlock = 16;
    if (C >= kChannelsPerBlock &&
        HxW >= cuda_utils::kCUDABlockReduceNumThreads) {
      const int64_t channel_blocks =
          (C + kChannelsPerBlock - 1) / kChannelsPerBlock;
      ComputeInternalGradientsChannelsLastCUDAKernel<kChannelsPerBlock, T>
          <<<N * channel_blocks, kCUDANumThreads, 0, cuda_stream>>>(
              C, HxW, dY_data, X_data, ds_data, db_data);
    } else {
      ComputeInternalGradientsChannelsLastFallbackCUDAKernel<T>
          <<<N * C, num_threads, 0, cuda_stream>>>(
              C, HxW, dY_data, X_data, ds_data, db_data);
    }
  } else {
    constexpr int kVecSize = 16 / sizeof(T);
    if (HxW % kVecSize == 0 &&
        memory::can_vectorize_up_to<T>(
            reinterpret_cast<const char*>(dY_data)) >= kVecSize &&
        memory::can_vectorize_up_to<T>(reinterpret_cast<const char*>(X_data)) >=
            kVecSize) {
      ComputeInternalGradientsContiguousCUDAKernel<kVecSize, T>
          <<<N * C, num_threads, 0, cuda_stream>>>(
              HxW, dY_data, X_data, ds_data, db_data);
    } else {
      ComputeInternalGradientsContiguousCUDAKernel<1, T>
          <<<N * C, num_threads, 0, cuda_stream>>>(
              HxW, dY_data, X_data, ds_data, db_data);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (dX.defined()) {
    Tensor c1 = at::empty({0}, X.options().dtype(kAccType));
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c2_data = c2.mutable_data_ptr<T_ACC>();
    T_ACC* c3_data = c3.mutable_data_ptr<T_ACC>();

    if (gamma.defined() && !channels_last) {
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

    if (channels_last) {
      const int64_t numel = N * C * HxW;
      const int64_t blocks = (numel + kCUDANumThreads - 1) / kCUDANumThreads;
      if (at::cuda::detail::canUse32BitIndexMath(X)) {
        using index_t = uint32_t;
        GroupNormBackwardChannelsLastCUDAKernel<index_t, T, T_ACC>
            <<<blocks, kCUDANumThreads, 0, cuda_stream>>>(
                static_cast<index_t>(numel),
                static_cast<index_t>(G),
                at::cuda::detail::IntDivider<index_t>(C),
                at::cuda::detail::IntDivider<index_t>(HxW),
                at::cuda::detail::IntDivider<index_t>(D),
                dY_data,
                X_data,
                rstd_data,
                gamma_data,
                c2_data,
                c3_data,
                dX.mutable_data_ptr<T>());
      } else {
        using index_t = int64_t;
        GroupNormBackwardChannelsLastCUDAKernel<index_t, T, T_ACC>
            <<<blocks, kCUDANumThreads, 0, cuda_stream>>>(
                numel,
                G,
                at::cuda::detail::IntDivider<index_t>(C),
                at::cuda::detail::IntDivider<index_t>(HxW),
                at::cuda::detail::IntDivider<index_t>(D),
                dY_data,
                X_data,
                rstd_data,
                gamma_data,
                c2_data,
                c3_data,
                dX.mutable_data_ptr<T>());
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (gamma.defined()) {
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

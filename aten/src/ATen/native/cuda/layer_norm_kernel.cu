#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/layer_norm.h>

#include <type_traits>

#include <thrust/tuple.h>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/thread_constants.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/env.h>


namespace at::native {

namespace {

constexpr int kCUDANumThreads = 256;
constexpr unsigned int kWarpSize = C10_WARP_SIZE;
constexpr int vec_size = 4; //we could make it dependent on dtype, but that would lead to different results between float and low-p types

// aligned vector generates vectorized load/store on CUDA (copy-pasted from MemoryAccess.cuh)
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

// Checks alignment of buffers for using vectorized loads / stores
template<typename T>
bool can_vectorize(const T * ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
};


template <typename T, typename T_ACC>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N,
    T_ACC eps,
    const T* X,
    T_ACC* mean,
    T_ACC* rstd) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;

  __shared__
      typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::
          type val_shared[C10_WARP_SIZE];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  val = cuda_utils::BlockReduce(
      val,
      welford_op,
      /*identity_element=*/WelfordType(0, 0, 0, 0),
      val_shared_ptr);

  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = c10::cuda::compat::rsqrt(m2 + eps);
  }
}

template <typename T, typename T_ACC>
__global__ void LayerNormForwardCUDAKernel(
    int64_t N,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    const T* gamma,
    const T* beta,
    T* Y) {
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

struct WelfordDataLN{
  float mean;
  float sigma2;
  float count;
  C10_HOST_DEVICE WelfordDataLN(): mean(0.f), sigma2(0.f), count(0.f){}
  C10_HOST_DEVICE WelfordDataLN(float mean, float sigma2, float count): mean(mean), sigma2(sigma2), count(count) {}
};

template<typename U> __device__
WelfordDataLN cuWelfordOnlineSum(
  const U val,
  const WelfordDataLN& curr_sum)
{
  U delta = val - curr_sum.mean;
  U new_count = curr_sum.count + 1.f;
  U new_mean = curr_sum.mean + delta * (1.f/new_count); //proper division is slow, this is less accurate but noticeably faster
  return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}

__device__
WelfordDataLN cuWelfordCombine(
  const WelfordDataLN dataB,
  const WelfordDataLN dataA
) {
  using U = decltype(dataB.count);
  U delta = dataB.mean - dataA.mean;
  U count = dataA.count + dataB.count;
  U mean, sigma2;
  if (count > decltype(dataB.count){0}) {
    auto coef = 1.f/count; //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
    auto nA = dataA.count * coef;
    auto nB = dataB.count * coef;
    mean = nA*dataA.mean + nB*dataB.mean;
    sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
  } else {
    mean = U(0);
    sigma2 = U(0);
  }
  return {mean, sigma2, count};
}

template<typename T>
__device__ WelfordDataLN compute_stats(
  const T*  __restrict__ X,
  const int N,
  float * buf
  ) {
    //X points to the row to read
    using vec_t = aligned_vector<T, vec_size>;
    using acc_t = acc_type<T, true>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(X);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;
    WelfordDataLN wd(0.f, 0.f, 0.f);
    //no tail, we check that N is multiple of vec_size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++){
        wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
      }
    }
    // intra-warp reduction
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset), WARP_SHFL_DOWN(wd.count, offset)};
        wd = cuWelfordCombine(wd, wdB);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float * meansigmabuf = buf;
      float * countbuf = buf + blockDim.y;
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          meansigmabuf[2*wrt_y] = wd.mean;
          meansigmabuf[2*wrt_y+1] = wd.sigma2;
          countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          WelfordDataLN wdB{meansigmabuf[2*threadIdx.y],
                          meansigmabuf[2*threadIdx.y+1],
                          countbuf[threadIdx.y]};
          wd = cuWelfordCombine(wd, wdB);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0 && threadIdx.y ==0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2/float(N);
      }
      __syncthreads();
      return WelfordDataLN{meansigmabuf[0], meansigmabuf[1],0.f};

    } else {
      return WelfordDataLN{WARP_SHFL(wd.mean,0), WARP_SHFL(wd.sigma2,0)/float(N), 0.f};
    }
}


template <typename T, typename T_ACC,
typename std::enable_if_t<!std::is_same_v<T, double>, int> = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_impl(
  const int N,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
    extern __shared__ float s_data[]; //if we made smem WelfordDataLN type, there would be bank conflicts,
    //as one thread would have to write 3 consecutive floats
    auto i1 = blockIdx.x;
    const T * block_row = X + i1 * N;
    WelfordDataLN wd = compute_stats(block_row, N, s_data);

    using vec_t = aligned_vector<T, vec_size>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(block_row);
    const vec_t * gamma_vec = (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    const vec_t * beta_vec = (beta != nullptr) ? reinterpret_cast<const vec_t*>(beta) : nullptr;
    vec_t * Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);

    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;

    T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);

    // No tail, N is guaranteed to be multiple of vec size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;

      // Computation is performed in T_ACC, X is cast to T_ACC and result is implicitly cast to T
      if (gamma_vec != nullptr && beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
            + static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else if (gamma_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        }
      } else if (beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) + static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
        }
      }
      Y_vec[i] = out;
    }
    if (thrx == 0) {
      mean[i1] = wd.mean;
      rstd[i1] = rstd_val;
    }
}

template <typename T, typename T_ACC,
typename std::enable_if_t<std::is_same_v<T, double>, int> = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_impl(
  const int /*N*/,
  T_ACC /*eps*/,
  const  T* __restrict__ /*X*/,
  const  T* /*gamma*/,
  const  T* /*beta*/,
  T_ACC* /*mean*/,
  T_ACC* /*rstd*/,
  T* /*Y*/){
    CUDA_KERNEL_ASSERT(false && "doesn't work with double");
  }

//to avoid windows SFINAE errors
template <typename T, typename T_ACC>
__global__ void vectorized_layer_norm_kernel(
  const int N,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
    vectorized_layer_norm_kernel_impl(N, eps, X, gamma, beta, mean, rstd, Y);
  }


template<typename T, typename T_ACC>
__device__ __inline__ void compute_gI(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T* dX,
  const int N,
  T_ACC * buf){
    const auto i1 = blockIdx.x;
    const T_ACC mean_val = mean[i1];
    const T_ACC rstd_val = rstd[i1];
    T_ACC stats_x1{0}, stats_x2{0};
    constexpr int unroll = 4;
    auto l = unroll * threadIdx.x;
    const T * X_i = X + i1 * N;
    const T * dY_i = dY + i1 * N;
    T * dX_i = dX + i1 * N;
    //vectorized reads don't improve perf, so use regular unrolling

    for (; l+unroll - 1 < N; l += blockDim.x * unroll){
      #pragma unroll
      for (int k=0; k< unroll; k++){
          const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l+k]) : T_ACC(1);
          const auto c_h = static_cast<T_ACC>(X_i[l+k]);
          const auto c_loss = static_cast<T_ACC>(dY_i[l+k]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
    }
    for (;  l < N; l ++) {
          const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);
          const auto c_h = static_cast<T_ACC>(X_i[l]);
          const auto c_loss = static_cast<T_ACC>(dY_i[l]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
    }

    stats_x1 = cuda_utils::BlockReduceSum(stats_x1, buf);
    stats_x2 = cuda_utils::BlockReduceSum(stats_x2, buf);
    if (threadIdx.x == 0) {
      buf[0] = stats_x1;
      buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    T_ACC fH = N;
    T_ACC term1 = (T_ACC(1) / fH) * rstd_val;

    for (int l = threadIdx.x; l < N; l += blockDim.x){
        const auto x = X_i[l];
        const auto dy = dY_i[l];
        const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);

        T_ACC f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        dX_i[l] = f_grad_input;
    }
  }


template<typename T, typename T_ACC>
__global__ void layer_norm_grad_input_kernel(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T*  dX,
  const int N){
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&s_data1);

    compute_gI(dY, X, mean, rstd, gamma, dX, N, buf);
  }


// This implementation gets called when input buffers (dY, X, gamma and dX) are aligned
// to vec_size * sizeof(T). Compared to the unvectorized implementation, it is about 10%
// faster measured at PT operator level, with cases seeing a 2X speedup (where N >> M).
// There are no noticeable regressions on the rest of the sizes.

template<typename T, typename T_ACC>
__global__ void layer_norm_grad_input_kernel_vectorized(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T* dX,
  const int N) {
  alignas(sizeof(double)) extern __shared__ char shared_data[];
  T_ACC* reduce_buf = reinterpret_cast<T_ACC*>(&shared_data);

  const auto bIdx = blockIdx.x;
  const T_ACC mean_val = mean[bIdx];
  const T_ACC rstd_val = rstd[bIdx];
  const T* X_i = X + bIdx * N;
  const T* dY_i = dY + bIdx * N;
  T* dX_i = dX + bIdx * N;

  using vec_t = aligned_vector<T, vec_size>;
  const vec_t* const X_i_vec_ptr = reinterpret_cast<const vec_t*>(X_i);
  const vec_t* const dY_i_vec_ptr = reinterpret_cast<const vec_t*>(dY_i);
  const vec_t* const gamma_vec_ptr = (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
  vec_t* const dX_i_vec = reinterpret_cast<vec_t*>(dX_i);

  vec_t X_i_vec_reg, dY_i_vec_reg, gamma_vec_reg, dX_i_vec_reg;
  for (int k = 0; k < vec_size; ++k) {
    gamma_vec_reg.val[k] = T(1);
  }

  T_ACC stats_x1{0}, stats_x2{0};
  unsigned int l = threadIdx.x * vec_size;
  for (; l + vec_size - 1 < N; l += blockDim.x * vec_size) {
    unsigned int vec_idx = l / vec_size;
    if (gamma != nullptr) {
      gamma_vec_reg = gamma_vec_ptr[vec_idx];
    }

    X_i_vec_reg = X_i_vec_ptr[vec_idx];
    dY_i_vec_reg = dY_i_vec_ptr[vec_idx];

    for (int k = 0; k < vec_size; ++k) {
      const auto gamma_val = static_cast<T_ACC>(gamma_vec_reg.val[k]);
      const auto c_h = static_cast<T_ACC>(X_i_vec_reg.val[k]);
      const auto c_loss = static_cast<T_ACC>(dY_i_vec_reg.val[k]);
      stats_x1 += c_loss * gamma_val;
      stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
    }
  }

  // Tail Loop
  for (; l < N; l++) {
    const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);
    const auto c_h = static_cast<T_ACC>(X_i[l]);
    const auto c_loss = static_cast<T_ACC>(dY_i[l]);
    stats_x1 += c_loss * gamma_val;
    stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
  }

  // Reduction in Shared Memory
  stats_x1 = cuda_utils::BlockReduceSum(stats_x1, reduce_buf);
  stats_x2 = cuda_utils::BlockReduceSum(stats_x2, reduce_buf);
  if (threadIdx.x == 0) {
    reduce_buf[0] = stats_x1;
    reduce_buf[1] = stats_x2;
  }
  __syncthreads();
  stats_x1 = reduce_buf[0];
  stats_x2 = reduce_buf[1];

  T_ACC fH = N;
  T_ACC term1 = (T_ACC(1) / fH) * rstd_val;

  l = threadIdx.x * vec_size;
  for (; l + vec_size - 1 < N; l += blockDim.x * vec_size) {
    unsigned int vec_idx = l / vec_size;
    if (gamma != nullptr) {
      gamma_vec_reg = gamma_vec_ptr[vec_idx];
    }

    X_i_vec_reg = X_i_vec_ptr[vec_idx];
    dY_i_vec_reg = dY_i_vec_ptr[vec_idx];

    for (int k = 0; k < vec_size; ++k) {
      const auto gamma_val = static_cast<T_ACC>(gamma_vec_reg.val[k]);
      const auto x = static_cast<T_ACC>(X_i_vec_reg.val[k]);
      const auto dy = static_cast<T_ACC>(dY_i_vec_reg.val[k]);

      T_ACC f_grad_input = fH * gamma_val * dy;
      f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
      f_grad_input -= stats_x1;
      f_grad_input *= term1;
      dX_i_vec_reg.val[k] = f_grad_input;
    }

    dX_i_vec[vec_idx] = dX_i_vec_reg;
  }

  // Tail Loop
  for (; l < N; l += blockDim.x) {
    const auto x = X_i[l];
    const auto dy = dY_i[l];
    const auto gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);

    T_ACC f_grad_input = fH * gamma_val * dy;
    f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
    f_grad_input -= stats_x1;
    f_grad_input *= term1;
    dX_i[l] = f_grad_input;
  }
}


template <typename T, typename T_ACC>
__global__ void GammaBetaBackwardSimpleCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dg,
    T* db) {
  const int64_t j = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
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

template <typename T, typename T_ACC,
unsigned int block_dim_x,
unsigned int block_dim_y,
unsigned int rows_per_block_y,
bool check_x,
bool check_y>
__device__
__forceinline__
void
blockReduceGammaBetaBackwardsHelper(
    int64_t M_start,
    int64_t M,
    int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd,
    T* __restrict__ dg,
    T* __restrict__ db,
    T_ACC &dg_sum,
    T_ACC &db_sum
) {
  constexpr int rows_per_thread_y = rows_per_block_y / block_dim_y;
  int64_t thread_x = blockIdx.x * block_dim_x + threadIdx.x;

    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) & (kWarpSize - 1);
    int64_t mean_index = M_start + threadIdx.y * rows_per_thread_y;
    T_ACC warp_mean = 0, warp_rstd = 0;
    if (lane_id < rows_per_thread_y && mean_index + lane_id < M) {
      warp_mean = mean[mean_index + lane_id];
      warp_rstd = rstd[mean_index + lane_id];
    }
    // We do a WARP_SYNC() here because we use WARP_SHFL below to access
    // warp_mean and warp_rstd.
    WARP_SYNC();

    T_ACC dY_regs[rows_per_thread_y] = {0};
    T_ACC X_regs[rows_per_thread_y] = {0};
    #pragma unroll
    for (int i = 0; i < rows_per_thread_y; ++i) {
      int64_t current_y = M_start + threadIdx.y * rows_per_thread_y + i;
      bool active = true;
      if (check_x && thread_x >= N) {
        active = false;
      }
      if (check_y && current_y >= M) {
        active = false;
      }
      if (active) {
        dY_regs[i] = dY[current_y * N + thread_x];
        X_regs[i] = X[current_y * N + thread_x];
      }
    }

    #pragma unroll
    for (int i = 0; i < rows_per_thread_y; ++i) {
      T_ACC mean_reg = WARP_SHFL(warp_mean, i, kWarpSize);
      T_ACC rstd_reg = WARP_SHFL(warp_rstd, i, kWarpSize);
      dg_sum += dY_regs[i] * (X_regs[i] - mean_reg) * rstd_reg;
      db_sum += dY_regs[i];
    }
}

template <typename T, typename T_ACC,
unsigned int block_dim_x,
unsigned int block_dim_y,
unsigned int rows_per_block_y,
bool check_x,
bool check_y>
__device__
__forceinline__
void
blockReduceGammaBetaBackwardsWithChecks(
    int64_t M,
    int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd,
    T* __restrict__ dg,
    T* __restrict__ db,
    T_ACC &dg_sum,
    T_ACC &db_sum
) {
  for (int64_t M_start = blockIdx.y * rows_per_block_y;
        M_start < M;
        M_start += rows_per_block_y * gridDim.y) {
    int64_t M_end = M_start + rows_per_block_y - 1;
    if (!check_y || M_end < M) {
      blockReduceGammaBetaBackwardsHelper<T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, check_x, false>
      (M_start, M, N, dY, X, mean, rstd, dg, db, dg_sum, db_sum);
    } else {
      blockReduceGammaBetaBackwardsHelper<T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, check_x, true>
      (M_start, M, N, dY, X, mean, rstd, dg, db, dg_sum, db_sum);
    }
  }
}

// block_dim_x is the number of threads in the x dimension per block.
// block_dim_y is the number of threads in the y dimension per block.
// rows_per_block_y is the size of the tile (number of data elements)
// in the y dimension per block.
// partial_reduction indicates whether we need to reduce across threads
// or not. If set to true, we will not reduce across threads. This can
// be faster in the M >> N case but requires another kernel to do a full
// final reduction.
// aligned_grid means the data size is a multiple of tile size. In that
// case we don't need to check for boundary conditions which can provide
// a further speedup by not needing instructions to check for edge cases
// and not needing predicate registers.
template <typename T, typename T_ACC,
unsigned int block_dim_x, unsigned int block_dim_y,
unsigned int rows_per_block_y,
bool partial_reduction,
bool aligned_grid
>
__global__
void
__launch_bounds__(block_dim_x * block_dim_y)
 GammaBetaBackwardCUDAKernelTemplate(
    int64_t M,
    int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd,
    T* __restrict__ dg,
    T* __restrict__ db) {
  // This assert is a compile-time check only.
  constexpr int rows_per_thread_y = rows_per_block_y / block_dim_y;
  static_assert(rows_per_thread_y <= kWarpSize);

  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;

  if (aligned_grid) {
    // When N and M align perfectly with block_dim_x and block_dim_y, we
    // can skip boundary condition checks that waste instruction issue slots.
    blockReduceGammaBetaBackwardsWithChecks
          <T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, false, false>
          (M, N, dY, X, mean, rstd, dg, db, dg_sum, db_sum);
  } else {
    // In the general case we need to check boundary conditions in the M
    // dimension. However, we can still avoid boundary checks in the N dimension
    // for the inner blocks. So try to avoid those checks when possible.
    if (blockIdx.x * block_dim_x + block_dim_x - 1 < N) {
      blockReduceGammaBetaBackwardsWithChecks
          <T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, false, true>
          (M, N, dY, X, mean, rstd, dg, db, dg_sum, db_sum);
    } else {
      blockReduceGammaBetaBackwardsWithChecks
          <T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, true, true>
          (M, N, dY, X, mean, rstd, dg, db, dg_sum, db_sum);
    }
  }

  int64_t thread_x = ((int64_t)blockIdx.x) * block_dim_x + threadIdx.x;

  // When partial_reduction is requested, we don't reduce within a block.
  // We also don't reduce if we are only a single block in the y dimension.
  if (partial_reduction || (blockDim.y == 1 && gridDim.y == 1)) {
    if (aligned_grid || thread_x < N) {
      int64_t thread_y = ((int64_t)blockIdx.y) * blockDim.y + threadIdx.y;
      if (dg) {
        dg[thread_y * N + thread_x] = dg_sum;
      }
      if (db) {
        db[thread_y * N + thread_x] = db_sum;
      }
    }
  } else {
    // The caller requested a full reduction so we must reduce across
    // warps using shared memory and warp shuffles.
    static_assert(rows_per_thread_y <= C10_WARP_SIZE);
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
    T_ACC* s_dg;
    T_ACC* s_db;
    int padded_bx = (block_dim_x + 1);
    // Transpose dg and db.
    s_dg = s_data_typed;
    s_db = s_data_typed + (padded_bx * block_dim_y);
    s_dg[threadIdx.y * padded_bx + threadIdx.x] = dg_sum;
    s_db[threadIdx.y * padded_bx + threadIdx.x] = db_sum;
    __syncthreads();

    // Load transposed so that a warp holds an entire column
    // Because block_dim_x != block_dim_y in the general case, we need
    // some code to handle the general case.
    static_assert(block_dim_x * block_dim_y % C10_WARP_SIZE == 0);
    constexpr int warps_available_to_reduce = block_dim_x * block_dim_y / C10_WARP_SIZE;
    int thread_id = threadIdx.y * block_dim_x + threadIdx.x;
    int warp_id = thread_id / C10_WARP_SIZE;
    int lane_id = thread_id & (C10_WARP_SIZE - 1);
    #pragma unroll
    for (int i = warp_id; i < block_dim_x; i += warps_available_to_reduce) {
      T_ACC reg_db, reg_dg;
      if (lane_id < block_dim_y) {
        reg_dg = s_dg[lane_id * padded_bx + i];
        reg_db = s_db[lane_id * padded_bx + i];
      }
      #pragma unroll
      for (unsigned delta = block_dim_y >> 1; delta >= 1; delta >>= 1) {
        reg_dg += WARP_SHFL_XOR(reg_dg, delta, kWarpSize);
        reg_db += WARP_SHFL_XOR(reg_db, delta, kWarpSize);
      }
      // Reduce is done. Now write it out to global memory.
      int64_t out_index = ((int64_t)blockIdx.x) * block_dim_x + i;
      if (threadIdx.x == 0 && (aligned_grid || out_index < N)) {
        if (dg) {
          dg[out_index] = reg_dg;
        }
        if (db) {
          db[out_index] = reg_db;
        }
      }
    }
  }
}

template<typename T, typename T_ACC,
int block_dim_x, int block_dim_y,
int rows_per_block_y,
bool partial_reduction>
void LaunchAndCheckGammaBetaBackwardKernel(
  bool aligned_grid,
  dim3 blocks,
  dim3 threads,
  size_t shmem_sz,
  cudaStream_t cuda_stream,
  const T* dY_data,
  const T* X_data,
  const T_ACC* mean_data,
  const T_ACC* rstd_data,
  int64_t M,
  int64_t N,
  T* dgamma_data,
  T* dbeta_data) {
if (aligned_grid) {
    GammaBetaBackwardCUDAKernelTemplate<T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, partial_reduction, true>
        <<<blocks, threads, shmem_sz, cuda_stream>>>(
            M,
            N,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            dgamma_data,
            dbeta_data);
  } else {
    GammaBetaBackwardCUDAKernelTemplate<T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, partial_reduction, false>
        <<<blocks, threads, shmem_sz, cuda_stream>>>(
            M,
            N,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            dgamma_data,
            dbeta_data);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T, typename T_ACC,
int block_dim_x, int block_dim_y,
int rows_per_block_y>
void ConfigureAndLaunchGammaBetaBackwardKernel(
    const T* dY_data,
    const T* X_data,
    const T_ACC* mean_data,
    const T_ACC* rstd_data,
    int64_t M,
    int64_t N,
    Tensor* dgamma,
    Tensor* dbeta,
    cudaStream_t cuda_stream) {
  T* dgamma_data =
    dgamma->defined() ? dgamma->template data_ptr<T>() : nullptr;
  T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;
  bool aligned_grid = (M % rows_per_block_y == 0) && (N % block_dim_x == 0);
  dim3 threads{block_dim_x, block_dim_y};
  dim3 blocks;
  blocks.x = (N + block_dim_x - 1) / block_dim_x;
  blocks.y = 1;
  size_t shmem_sz = (block_dim_x + 1) * block_dim_y * sizeof(T_ACC) * 2;
  if (blocks.y == 1 && threads.y == 1) {
    // Optimization: since there is just one thread doing all the summation, we don't need a reduction
    // across threads. So we set partial_reduction to true.
    LaunchAndCheckGammaBetaBackwardKernel<T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, true>(
      aligned_grid, blocks, threads, shmem_sz, cuda_stream, dY_data, X_data, mean_data, rstd_data, M, N, dgamma_data, dbeta_data);
  } else {
    LaunchAndCheckGammaBetaBackwardKernel<T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, false>(
      aligned_grid, blocks, threads, shmem_sz, cuda_stream, dY_data, X_data, mean_data, rstd_data, M, N, dgamma_data, dbeta_data);
  }

}

template<typename T, typename T_ACC>
void LaunchGammaBetaBackwardCUDAKernel(
    const T* dY_data,
    const T* X_data,
    const T_ACC* mean_data,
    const T_ACC* rstd_data,
    int64_t M,
    int64_t N,
    Tensor* dgamma,
    Tensor* dbeta,
    cudaStream_t cuda_stream) {
  constexpr int block_dim_x = 32;
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (M > 64 * 1024 && N / block_dim_x < sm_count / 2) {
    // We have a situation where M >> N and N is small.
    // In this case we can speed up the computation by parallelizing in the M dimension.
    // We launch multiple blocks in the y-dimension, and compute partial sums for the
    // gradient in the first pass. Then we do a .sum(0) to do a final reduction.
    // Although we launch 2 kernels, we can get up to a 10x speedup for large M.
    constexpr int block_dim_y = 1;
    constexpr int rows_per_block_y = 32;
    bool aligned_grid = (M % rows_per_block_y == 0) && (N % block_dim_x == 0);
    dim3 threads{block_dim_x, block_dim_y};
    dim3 blocks;
    blocks.x = (N + block_dim_x - 1) / block_dim_x;
    // int rows_per_block = my_gamma_beta_unroll_factor *
    blocks.y = (M + rows_per_block_y - 1) / rows_per_block_y;
    constexpr int max_grid_size = 64 * 1024 / 2;
    blocks.y = std::min<unsigned int>(max_grid_size / blocks.x, blocks.y);
    Tensor dgamma_blocks;
    Tensor dbeta_blocks;
    T * dgamma_blocks_ptr = nullptr;
    T * dbeta_blocks_ptr = nullptr;
    if (dgamma->defined()) {
      auto options = dgamma->options();
      dgamma_blocks = at::empty({blocks.y * threads.y, dgamma->size(-1)}, options);
      dgamma_blocks_ptr = dgamma_blocks.data_ptr<T>();
    }
    if (dbeta->defined()) {
      auto options = dbeta->options();
      dbeta_blocks = at::empty({blocks.y * threads.y, dgamma->size(-1)}, options);
      dbeta_blocks_ptr = dbeta_blocks.data_ptr<T>();
    }
    LaunchAndCheckGammaBetaBackwardKernel<T, T_ACC, block_dim_x, block_dim_y, rows_per_block_y, true>(
      aligned_grid, blocks, threads, 0, cuda_stream, dY_data, X_data, mean_data, rstd_data, M, N, dgamma_blocks_ptr, dbeta_blocks_ptr);

    if (dgamma_blocks.defined()) {
      *dgamma = dgamma_blocks.sum(0);
    }
    if (dbeta_blocks.defined()) {
      *dbeta = dbeta_blocks.sum(0);
    }
  } else {
    // We are in the normal case where M is not that large.
    // We can change the tile shape (which is the last template parameter) in accordance with M.
    // For small M it is faster to have a smaller tile, otherwise we could have idle threads.
    // For larger M we use a bigger tile size.
    if (M < 64) {
      ConfigureAndLaunchGammaBetaBackwardKernel<T, T_ACC, block_dim_x, 1, 8>(dY_data, X_data, mean_data, rstd_data, M, N, dgamma, dbeta, cuda_stream);
    } else if (M < 128) {
      ConfigureAndLaunchGammaBetaBackwardKernel<T, T_ACC, block_dim_x, 8, 64>(dY_data, X_data, mean_data, rstd_data, M, N, dgamma, dbeta, cuda_stream);
    } else if (M < 256) {
      ConfigureAndLaunchGammaBetaBackwardKernel<T, T_ACC, block_dim_x, 16, 128>(dY_data, X_data, mean_data, rstd_data, M, N, dgamma, dbeta, cuda_stream);
    } else {
      ConfigureAndLaunchGammaBetaBackwardKernel<T, T_ACC, block_dim_x, 32, 256>(dY_data, X_data, mean_data, rstd_data, M, N, dgamma, dbeta, cuda_stream);
    }
  }
}

template <typename T, typename T_ACC>
void launch_vectorized_layer_norm_kernel(
  int N,
  int64_t M,
  T_ACC eps,
  const T* X_data,
  const T* gamma_data,
  const T* beta_data,
  T* Y_data,
  T_ACC* mean_data,
  T_ACC* rstd_data
) {
    //constexpr int alignment = 16; //currently unused to make sure float and half results are bw accurate
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();
    const dim3 threads(warp_size, num_threads() / warp_size, 1);
    dim3 blocks(M);

#ifdef USE_ROCM
    uint64_t workgroupSize = static_cast<uint64_t>(blocks.x) * static_cast<uint64_t>(threads.x);
    // this caused invalid configuration problem
    if (workgroupSize > std::numeric_limits<uint32_t>::max()) {
      // Fix invalid configuration https://github.com/pytorch/pytorch/issues/136291
      blocks.x = std::numeric_limits<uint32_t>::max() / threads.x;
    }
#endif

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(threads.y % 2 == 0 || threads.y == 1);
    int nshared = threads.y > 1 ? threads.y * 3/2 *sizeof(T_ACC) : 0;
    vectorized_layer_norm_kernel<<<blocks, threads, nshared, stream>>>(N, eps, X_data,
    gamma_data, beta_data, mean_data, rstd_data, Y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef USE_ROCM
    // the blocks.x contains the max grid x dimention without invalid configuration error
    // Fix invalid configuration https://github.com/pytorch/pytorch/issues/136291
    // Ensure all elements are processed. Prepare for next round
    int64_t remaining = M - blocks.x;
    const T* X_data2 = X_data;
    T_ACC* mean_data2 = mean_data;
    T_ACC* rstd_data2 = rstd_data;
    T* Y_data2 = Y_data;

    while (remaining > 0) {
      X_data2 += N * blocks.x;
      mean_data2 += blocks.x;
      rstd_data2 += blocks.x;
      Y_data2 += N * blocks.x;

      blocks.x = (remaining > blocks.x) ? blocks.x : remaining;

      vectorized_layer_norm_kernel<<<blocks, threads, nshared, stream>>>(N, eps, X_data2,
        gamma_data, beta_data, mean_data2, rstd_data2, Y_data2);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      remaining -= blocks.x;
    }
#endif

}

template <typename T, typename T_ACC>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T_ACC eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  // assumes input, gamma and beta are of proper shape, this was checked in _check_layer_norm_inputs
  // assumes all tensors are contiguous
  TORCH_CHECK(M <= at::cuda::getCurrentDeviceProperties()->maxGridSize[0], "M should be less than maximum CUDA grid size, \
  file a support request to support bigger batches");
  const T* X_data = X.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T_ACC* mean_data = mean->data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd->data_ptr<T_ACC>();

  // check if can take fast path - all tensors are properly aligned, N is less than 2^24 (to use float count),
  // N is multiple of vec_size (so that all rows are aligned if tensor is aligned)
  constexpr int num_vec_elems = vec_size;
  constexpr int alignment = num_vec_elems * sizeof(T);
  bool can_vec_X = can_vectorize(X_data, alignment);
  bool can_vec_Y = can_vectorize(Y_data, alignment);
  bool can_vec_gamma = gamma.defined() ? can_vectorize(gamma_data, alignment) : true;
  bool can_vec_beta = beta.defined() ? can_vectorize(beta_data, alignment) : true;

  if ((std::is_same_v<T, float> || std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16>) &&
  N <= static_cast<int64_t>(1ULL << std::numeric_limits<float>::digits) && N % num_vec_elems == 0 &&
  can_vec_X && can_vec_Y && can_vec_gamma && can_vec_beta) {
    launch_vectorized_layer_norm_kernel(static_cast<int>(N), M, eps, X_data, gamma_data, beta_data, Y_data, mean_data, rstd_data);
  } else {
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  RowwiseMomentsCUDAKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          N, eps, X_data, mean_data, rstd_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  LayerNormForwardCUDAKernel<T, T_ACC><<<M, kCUDANumThreads, 0, cuda_stream>>>(
      N, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormKernelImpl",
      [&]() {
        using acc_t = acc_type<scalar_t, true>;
        LayerNormKernelImplInternal<scalar_t, acc_t>(
            X, gamma, beta, M, N, static_cast<acc_t>(eps), Y, mean, rstd);
      });
}

template<typename T, typename T_ACC> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    T_ACC* warp_buf1,
    T_ACC* warp_buf2,
    const T* input,
    const T* dout,
    const int i1_end,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd)
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    T_ACC curr_mean = mean[i1];
    T_ACC curr_rstd = rstd[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*N+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<N) {
        T curr_input = static_cast<T>(input[load_idx]);
        T curr_dout = static_cast<T>(dout[load_idx]);
        warp_buf1[write_idx] = curr_dout;
        warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_rstd;
      } else {
        warp_buf1[write_idx] = T(0);
        warp_buf2[write_idx] = T(0);
      }
    }
  } else {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      warp_buf1[write_idx] = T(0);
      warp_buf2[write_idx] = T(0);
    }
  }
}

template<typename T, typename T_ACC> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    T_ACC* warp_buf1,
    T_ACC* warp_buf2,
    const T* input,
    const T* dout,
    const int i1_end,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd)
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    T_ACC curr_mean = mean[i1];
    T_ACC curr_rstd = rstd[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*N+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<N) {
        T_ACC curr_input = static_cast<T_ACC>(input[load_idx]);
        T_ACC curr_dout = static_cast<T_ACC>(dout[load_idx]);
        warp_buf1[write_idx] += curr_dout;
        warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_rstd;
      }
    }
  }
}

template<typename T, typename T_ACC> __global__
void cuComputePartGradGammaBeta(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const int64_t M,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd,
    T_ACC* part_grad_gamma,
    T_ACC* part_grad_beta)
{
    const int numsegs_M = (M+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_M + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < M ? i1_beg_plus_one : M;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    alignas(sizeof(double)) extern __shared__ char shared[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&shared); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    T_ACC* warp_buf1 = (T_ACC*)buf;
    T_ACC* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,N,mean,rstd);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y;  i1_block < i1_end;  i1_block+=blockDim.y*blockDim.y) {
      cuLoadAddStridedInputs(i1_block,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,N,mean,rstd);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    T_ACC acc1 = T_ACC(0);
    T_ACC acc2 = T_ACC(0);
    for (int k = 0;  k < blockDim.y;  ++k) {
      int row1 = threadIdx.y + k*blockDim.y;
      int idx1 = row1*row_stride + threadIdx.x;
      acc1 += warp_buf1[idx1];
      acc2 += warp_buf2[idx1];
    }
    warp_buf1[threadIdx.y*row_stride+threadIdx.x] = acc1;
    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2;  offset > 1;  offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + offset;
        int idx1 = row1*row_stride + threadIdx.x;
        int idx2 = row2*row_stride + threadIdx.x;
        warp_buf1[idx1] += warp_buf1[idx2];
        warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < N) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1*row_stride + threadIdx.x;
      int idx2 = row2*row_stride + threadIdx.x;
      part_grad_beta[blockIdx.y*N+i2] = warp_buf1[idx1] + warp_buf1[idx2];
      part_grad_gamma[blockIdx.y*N+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename T, typename T_ACC> __global__
void cuComputeGradGammaBeta(
    const T_ACC* part_grad_gamma,
    const T_ACC* part_grad_beta,
    const int part_size,
    const int64_t M,
    const int64_t N,
    T* grad_gamma,
    T* grad_beta)
{
    // sum partial gradients for gamma and beta
    alignas(sizeof(double)) extern __shared__ char shared[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&shared);
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;

    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / blockDim.y;
    T_ACC sum_gamma = T_ACC(0);
    T_ACC sum_beta = T_ACC(0);
    const T_ACC* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * N + i2;
    const T_ACC* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * N + i2;

    if (i2 < N) {
        for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
          sum_gamma += part_grad_gamma_ptr[warp_offset*N];
          sum_beta += part_grad_beta_ptr[warp_offset*N];
        }
    }

    // inter-warp reductions
    const int nbsize3 = blockDim.x * blockDim.y / 2;
    for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
        buf[write_idx+nbsize3] = sum_beta;
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
        sum_gamma += buf[read_idx];
        sum_beta += buf[read_idx+nbsize3];
      }
      __syncthreads();
    }

    // write out fully summed gradients
    if (threadIdx.y == 0 && i2 < N) {
      if (grad_gamma) {
          grad_gamma[i2] = sum_gamma;
      }
      if (grad_beta) {
          grad_beta[i2] = sum_beta;
      }
    }
}

template<typename T, typename T_ACC> __global__
void cuComputeGradInput(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const int64_t M,
    const int64_t N,
    const T_ACC* __restrict__ mean,
    const T_ACC* __restrict__ rstd,
    const T* gamma,
    T* grad_input)
{
  for (int i1=blockIdx.y; i1 < M; i1 += gridDim.y) {
    T_ACC sum_loss1 = T_ACC(0);
    T_ACC sum_loss2 = T_ACC(0);
    T_ACC c_mean = mean[i1];
    const T_ACC c_rstd = rstd[i1];
    const T* k_input = input + i1*N;
    const T* k_dout = dout + i1*N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      // Optimization for ROCm MI100
      for( int l = 0; l < N ; l += numx) {
        int idx = l + thrx;
        const T_ACC gamma_idx = static_cast<T_ACC>((idx<N) ? gamma[idx] : T(0));
        const T_ACC c_h = static_cast<T_ACC>((idx<N) ? k_input[idx] : T(0));
        const T_ACC c_loss = static_cast<T_ACC>((idx<N) ? k_dout[idx] : T(0));
        sum_loss1 += c_loss * gamma_idx;
        sum_loss2 += c_loss * gamma_idx * (c_h - c_mean) * c_rstd;
      }
    } else {
      for( int l = 0; l < N ; l += numx) {
        int idx = l + thrx;
        const T_ACC c_h = static_cast<T_ACC>((idx<N) ? k_input[idx] : T(0));
        const T_ACC c_loss = static_cast<T_ACC>((idx<N) ? k_dout[idx] : T(0));
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_rstd;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      alignas(sizeof(double)) extern __shared__ char shared[];
      T_ACC * buf = reinterpret_cast<T_ACC*>(&shared);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2*wrt_i] = sum_loss1;
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2*read_i];
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2*threadIdx.x] = sum_loss1;
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss1 = buf[2*threadIdx.x];
        sum_loss2 = buf[2*threadIdx.x+1];
      }
    }
    // all threads now have the two sums over l
    T_ACC fH = (T_ACC)N;
    T_ACC term1 = (T_ACC(1) / fH) * c_rstd;
    T* k_grad_input = grad_input + i1*N;
    if (gamma != NULL) {
      for (int l = thrx;  l < N;  l+=numx) {
        const T_ACC c_h = static_cast<T_ACC>(k_input[l]);
        const T_ACC c_loss = static_cast<T_ACC>(k_dout[l]);
        T_ACC f_grad_input = fH * c_loss * gamma[l];
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_rstd * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < N;  l+=numx) {
        const T_ACC c_h = static_cast<T_ACC>(k_input[l]);
        const T_ACC c_loss = static_cast<T_ACC>(k_dout[l]);
        T_ACC f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_rstd * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
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
  TORCH_CHECK(dY.numel() == M * N);
  TORCH_CHECK(mean.numel() == M);
  TORCH_CHECK(rstd.numel() == M);
  TORCH_CHECK(M <= at::cuda::getCurrentDeviceProperties()->maxGridSize[0], "M should be less than maximum CUDA grid size, \
  file a support request to support bigger batches");
  TORCH_CHECK(N <= std::numeric_limits<int>::max(), "Normalized shape should have less than INT_MAX elements, \
  file a support request to support bigger normalized shapes");
  const T* dY_data = dY.template const_data_ptr<T>();
  const T* X_data = X.template const_data_ptr<T>();
  const T_ACC* mean_data = mean.template const_data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.template const_data_ptr<T_ACC>();
  const T* gamma_data =
      gamma.defined() ? gamma.template const_data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int warp_size = at::cuda::warp_size();
  if (dX_data != nullptr) {
#ifdef USE_ROCM
    if (M >= 32768) {
      const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
      const dim3 blocks1(1, std::min((uint64_t)M, maxGridY), 1);
      dim3 threads1(warp_size, 4, 1);
      threads1.y = 2; // Optimization for ROCm
      int nshared =
              threads1.y > 1 ?
              threads1.y*threads1.x*sizeof(T_ACC) :
              0;
      cuComputeGradInput<<<blocks1, threads1, nshared, cuda_stream>>>(
              dY_data,
              X_data,
              M, N,
              mean_data,
              rstd_data,
              gamma_data,
              dX_data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      const dim3 blocks(M);
      int nshared = (num_threads()/warp_size) * sizeof(T_ACC);
      layer_norm_grad_input_kernel<<<blocks, num_threads(), nshared, cuda_stream>>>(dY_data,
      X_data, mean_data, rstd_data, gamma_data, dX_data, N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
#else
    const dim3 blocks(M);
    int nshared = (num_threads() / warp_size) * sizeof(T_ACC);

    bool bVectorSizeMultiple = (N % vec_size == 0);
    bool bTargetDataTypes = (std::is_same_v<T, float> || std::is_same_v<T, at::Half> ||
      std::is_same_v<T, at::BFloat16>);
    const unsigned int alignment = sizeof(T) * vec_size;
    bool bAlignedBuffers = can_vectorize(dY_data, alignment) && can_vectorize(X_data, alignment) &&
      can_vectorize(gamma_data, alignment) && can_vectorize(dX_data, alignment);

    if (bAlignedBuffers && bTargetDataTypes && bVectorSizeMultiple) {
      layer_norm_grad_input_kernel_vectorized<<<blocks, num_threads(), nshared, cuda_stream>>>(dY_data,
          X_data, mean_data, rstd_data, gamma_data, dX_data, N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      layer_norm_grad_input_kernel<<<blocks, num_threads(), nshared, cuda_stream>>>(dY_data,
          X_data, mean_data, rstd_data, gamma_data, dX_data, N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
#endif
  }

  if (dgamma->defined() || dbeta->defined()) {
    T* dgamma_data =
        dgamma->defined() ? dgamma->template data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;

#if defined(USE_ROCM)
    if (M < 128) {
      // For small batch size, do colwise reduce directly.
      const int64_t B = (N + kCUDANumThreads - 1) / kCUDANumThreads;
      GammaBetaBackwardSimpleCUDAKernel<T, T_ACC>
          <<<B, kCUDANumThreads, 0, cuda_stream>>>(
              M,
              N,
              dY_data,
              X_data,
              mean_data,
              rstd_data,
              dgamma_data,
              dbeta_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // For small batch size, do colwise reduce directly.
      const int part_size = warp_size;
      const dim3 threads2(warp_size, 4, 1);
      const dim3 blocks2((N + threads2.x - 1) / threads2.x, part_size, 1);
      const int nshared2_a = 2 * sizeof(T_ACC) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(T_ACC);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;

      const auto part_grad_dtype = at::toAccumulateType(X.scalar_type(), true);
      Tensor part_grad_gamma = at::empty({part_size,N}, gamma.options().dtype(part_grad_dtype));
      Tensor part_grad_beta = at::native::empty_like(part_grad_gamma);

      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, cuda_stream>>>(
                      dY_data,
                      X_data,
                      M,N,
                      mean_data,
                      rstd_data,
                      part_grad_gamma.template data_ptr<T_ACC>(),
                      part_grad_beta.template data_ptr<T_ACC>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      const dim3 threads3(warp_size, 8, 1); // Optimization for ROCm
      const dim3 blocks3((N + threads3.x - 1) / threads3.x, 1, 1);
      const int nshared3 = threads3.x * threads3.y * sizeof(T_ACC);

      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, cuda_stream>>>(
                      part_grad_gamma.template data_ptr<T_ACC>(),
                      part_grad_beta.template data_ptr<T_ACC>(),
                      part_size,
                      M,N,
                      dgamma_data,
                      dbeta_data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
#else
    LaunchGammaBetaBackwardCUDAKernel(
      dY_data, X_data, mean_data, rstd_data, M, N, dgamma, dbeta, cuda_stream);
#endif
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormBackwardKernelImpl",
      [&]() {
        LayerNormBackwardKernelImplInternal<scalar_t>(
            dY.contiguous(), X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      });
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> layer_norm_cuda(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      *X,
      std::nullopt /* dtype */,
      std::nullopt /* layout */,
      std::nullopt /* device */,
      std::nullopt /* pin_memory */,
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  Tensor mean = at::empty({M}, X->options().dtype(acc_type));
  Tensor rstd = at::empty({M}, X->options().dtype(acc_type));
  // Calling the kernel for M==0 gives a CUDA error
  // See: https://github.com/pytorch/pytorch/pull/28614
  if (M > 0) {
    LayerNormKernelImpl(*X, *gamma, *beta, M, N, eps, &Y, &mean, &rstd);
  }
  const auto input_shape = input.sizes();
  const size_t axis = input.dim() - normalized_shape.size();

  std::vector<int64_t> stat_shape;
  for (const auto idx: c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  for ([[maybe_unused]] const auto idx : c10::irange(axis, input.dim())) {
    stat_shape.push_back(1);
  }

  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);

  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cuda(
    const Tensor& dY,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    std::array<bool, 3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        *X,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         std::nullopt /* dtype */,
                         std::nullopt /* layout */,
                         std::nullopt /* device */,
                         std::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                   : at::native::zeros_like(
                         *gamma,
                         std::nullopt /* dtype */,
                         std::nullopt /* layout */,
                         std::nullopt /* device */,
                         std::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        std::nullopt /* dtype */,
                        std::nullopt /* layout */,
                        std::nullopt /* device */,
                        std::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                  : at::native::zeros_like(
                        *beta,
                        std::nullopt /* dtype */,
                        std::nullopt /* layout */,
                        std::nullopt /* device */,
                        std::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (M > 0 && N > 0) {
    LayerNormBackwardKernelImpl(
        dY, *X, mean, rstd, *gamma, M, N, &dX, &dgamma, &dbeta);
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl)
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl)

} // namespace at::native

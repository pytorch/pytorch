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

namespace at {
namespace native {

namespace {

constexpr int kCUDANumThreads = 256;
constexpr int kColwiseReduceTileSize = 32;
constexpr int vec_size = 4; //we could make it dependent on dtype, but that would lead to different results between float and low-p types

// aligned vector generates vectorized load/store on CUDA (copy-pasted from MemoryAccess.cuh)
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};


template <typename T, typename T_ACC>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N,
    T_ACC eps,
    const T* X,
    T_ACC* mean,
    T_ACC* rstd) {
  using WelfordType = WelfordData<T_ACC, int64_t, T_ACC>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, T_ACC, thrust::pair<T_ACC, T_ACC>>;

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
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
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
    vec_t * Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;
    T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
    //no tail, N is guaranteed to be multiple of vec size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;
      //computation is performed in T_ACC, X is cast to T_ACC and result is implicitly cast to T
      if (gamma != nullptr && beta != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma[i*vec_size + ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
          + static_cast<T_ACC>(beta[i*vec_size + ii]);
        }
      } else if (gamma != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma[i*vec_size + ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        }
      } else if (beta != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) + static_cast<T_ACC>(beta[i*vec_size + ii]);
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
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
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
__global__ __inline__ void vectorized_layer_norm_kernel(
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
  sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
  sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  if (threadIdx.x == 0) {
    ds[i] = sum1;
    db[i] = sum2;
  }
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
          T_ACC gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l+k]) : T_ACC(1);
          const T_ACC c_h = static_cast<T_ACC>(X_i[l+k]);
          const T_ACC c_loss = static_cast<T_ACC>(dY_i[l+k]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
    }
    for (;  l < N; l ++) {
          T_ACC gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);
          const T_ACC c_h = static_cast<T_ACC>(X_i[l]);
          const T_ACC c_loss = static_cast<T_ACC>(dY_i[l]);
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
        const T_ACC x = X_i[l];
        const T_ACC dy = dY_i[l];
        T_ACC gamma_val = (gamma != nullptr) ? static_cast<T_ACC>(gamma[l]) : T_ACC(1);
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


template <typename T, typename T_ACC>
__global__ void ComputeGradientFusedParamsCUDAKernel(
    int64_t M,
    int64_t N,
    const T_ACC* mean,
    const T_ACC* rstd,
    const acc_type<T, true>* ds,
    const acc_type<T, true>* db,
    acc_type<T, true>* c1,
    acc_type<T, true>* c2) {
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

template <typename T, typename T_ACC>
__global__ void LayerNormBackwardCUDAKernel(
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T_ACC* a,
    const acc_type<T, true>* b,
    const acc_type<T, true>* c,
    T* dX) {
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

template <typename T, typename T_ACC>
__global__ void GammaBetaBackwardCUDAKernel1(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dg,
    T* db) {
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
  sum1 = cuda_utils::WarpReduceSum(sum1);
  sum2 = cuda_utils::WarpReduceSum(sum2);
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
  sum1 = cuda_utils::WarpReduceSum(sum1);
  sum2 = cuda_utils::WarpReduceSum(sum2);
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




template <typename T, typename T_ACC>
__global__ void GammaBetaBackwardCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dg,
    T* db) {
  alignas(sizeof(double)) extern __shared__ char s_data1[];
  T_ACC * s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int unroll = 8;
  T dYs[unroll];
  T Xs[unroll];
  T_ACC *  means = s_data_typed;
  T_ACC * rstds = s_data_typed + unroll * blockDim.y;
  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;
  if (j < N) {
    int bcounter;
    for (bcounter = 0; bcounter < M/(blockDim.y * unroll); bcounter++){
      int offset = (bcounter * blockDim.y + threadIdx.y) * unroll;
      #pragma unroll
      for (int ii=0; ii<unroll; ii++){
        if (threadIdx.x == 0) {
          means[ii*blockDim.y + threadIdx.y] = mean[offset + ii];
          rstds[ii*blockDim.y + threadIdx.y] = rstd[offset + ii];
        }
        dYs[ii] = dY[(offset + ii) * N + j ];
        Xs[ii] = X[(offset + ii) * N + j];

      }
      __syncthreads();
      #pragma unroll
      for (int ii=0; ii<unroll; ii++){
        dg_sum += dYs[ii] * (Xs[ii] - means[ii*blockDim.y + threadIdx.y]) * rstds[ii * blockDim.y + threadIdx.y];
        db_sum += dYs[ii];
      }
      __syncthreads();
    }
    int offset = (bcounter * blockDim.y + threadIdx.y) * unroll;
    for (int ii = 0; ii<8; ii++ ){
      T_ACC mean_val, rstd_val; // we don't use smem in the tail to avoid awkward synchronizations, perf penalty is negligible
      if ((offset + ii) < M) {
        mean_val = mean[offset+ii];
        rstd_val = rstd[offset+ii];
        dYs[0] = dY[(offset + ii) * N + j ];
        Xs[0] = X[(offset + ii) * N + j];
        dg_sum += dYs[0] * (Xs[0] - mean_val) * rstd_val;
        db_sum += dYs[0];
      }
    }
    s_data_typed[threadIdx.y * blockDim.x + threadIdx.x] = dg_sum;
    s_data_typed[blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] = db_sum;
    __syncthreads();
    for (int offset = blockDim.y/2; offset >=1; offset /= 2){
      if (threadIdx.y < offset) {
        s_data_typed[threadIdx.y * blockDim.x + threadIdx.x] += s_data_typed[(threadIdx.y + offset) * blockDim.x + threadIdx.x];
        s_data_typed[blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] +=
        s_data_typed[blockDim.x * blockDim.y + (threadIdx.y + offset) * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    if (threadIdx.y == 0) {
      if (dg) {
        dg[j] = s_data_typed[threadIdx.x];
      }
      if (db) {
        db[j] = s_data_typed[threadIdx.x + blockDim.x * blockDim.y];
      }
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
    const int num_threads = 128;
    const dim3 threads(C10_WARP_SIZE,num_threads/C10_WARP_SIZE,1);
    const dim3 blocks(M);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(threads.y % 2 == 0 || threads.y == 1);
    int nshared = threads.y > 1 ? threads.y * 3/2 *sizeof(T_ACC) : 0;
    vectorized_layer_norm_kernel<<<blocks, threads, nshared, stream>>>(N, eps, X_data,
    gamma_data, beta_data, mean_data, rstd_data, Y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T_ACC* mean_data = mean->data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd->data_ptr<T_ACC>();
  // check if can take fast path - all tensors are properly aligned, N is less than 2^24 (to use float count),
  // N is multiple of vec_size (so that all rows are aligned if tensor is aligned)
  auto can_vectorize = [&](const T * ptr, int alignment){uint64_t addr = reinterpret_cast<uint64_t>(ptr); return addr % alignment == 0;};
  constexpr int num_vec_elems = vec_size;
  constexpr int alignment = num_vec_elems * sizeof(T);
  if ((std::is_same<T, float>::value || std::is_same<T, at::Half>::value) &&
  N <= 1ULL << std::numeric_limits<float>::digits && N % num_vec_elems == 0 &&
  can_vectorize(X_data, alignment) && can_vectorize(Y_data, alignment)) {
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
  const T* dY_data = dY.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const T_ACC* mean_data = mean.template data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.template data_ptr<T_ACC>();
  const T* gamma_data =
      gamma.defined() ? gamma.template data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (dX_data != nullptr) {
    const int num_threads = 128;
    const dim3 blocks(M);
    int nshared = (num_threads/C10_WARP_SIZE) * sizeof(T_ACC);
    layer_norm_grad_input_kernel<<<blocks, num_threads, nshared, cuda_stream>>>(dY_data,
    X_data, mean_data, rstd_data, gamma_data, dX_data, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (dgamma->defined() || dbeta->defined()) {
    T* dgamma_data =
        dgamma->defined() ? dgamma->template data_ptr<T>() : nullptr;
    T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;
    if (M < 512) {
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
      dim3 threads{16, 32};
      int blocks = (N + threads.x-1)/threads.x;
      GammaBetaBackwardCUDAKernel<T, T_ACC>
          <<<blocks, threads, 2 * sizeof(T_ACC) * threads.x * threads.y, cuda_stream>>>(
              M,
              N,
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
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
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
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  Tensor mean = at::empty({M}, X->options().dtype(acc_type));
  Tensor rstd = at::empty({M}, X->options().dtype(acc_type));
  if (M > 0) {
    LayerNormKernelImpl(*X, *gamma, *beta, M, N, eps, &Y, &mean, &rstd);

    const auto input_shape = input.sizes();
    const size_t axis = input.dim() - normalized_shape.size();

    std::vector<int64_t> stat_shape;
    for (size_t idx = 0; idx < axis; ++idx) {
      stat_shape.push_back(input_shape[idx]);
    }
    for (size_t idx = axis; idx < input.dim(); ++idx) {
      stat_shape.push_back(1);
    }

    mean = mean.view(stat_shape);
    rstd = rstd.view(stat_shape);
  }
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cuda(
    const Tensor& dY,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
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
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                   : at::native::zeros_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                  : at::native::zeros_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (M > 0) {
    LayerNormBackwardKernelImpl(
        dY, *X, mean, rstd, *gamma, M, N, &dX, &dgamma, &dbeta);
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}


} // namespace native
} // namespace at

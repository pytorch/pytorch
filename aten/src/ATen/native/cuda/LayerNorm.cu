#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCDeviceUtils.cuh>


namespace at {
namespace native {
namespace {

template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int M,
  const int N,
  const int i1,
  U& mu,
  U& sigma2,
  U* buf)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over N
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < M) {
    // one warp normalizes one M index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * N;
    int l = 4 * thrx;
    for (; l + 3 < N; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        printf("i1=%ld, N=%ld, index=%ld, lvals[l + k]=%lf\n", long(i1), long(N), long(i1 * N + l + k), double(lvals[l + k]));
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      }
    }
    for (; l < N; ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      U muB = WARP_SHFL(mu, srcLaneB);
      U countB = WARP_SHFL(count, srcLaneB);
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2 * threadIdx.y];
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / U(N);
      // don't care about final value of count, we know count == N
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / U(N), 0);
    }
  }
}

template<> __device__
void cuWelfordMuSigma2(
  const at::Half* __restrict__ vals,
  const int M,
  const int N,
  const int i1,
  float& mu,
  float& sigma2,
  float* buf)
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over N
  float count = 0.0f;
  mu = float(0);
  sigma2 = float(0);
  if (i1 < M) {
    // one warp normalizes one M index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const at::Half* lvals = vals + i1 * N;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < N; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2*)(lvals + l + k)));
        cuWelfordOnlineSum(curr.x, mu, sigma2, count);
        cuWelfordOnlineSum(curr.y, mu, sigma2, count);
      }
    }
    for (; l < N; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      float muB = WARP_SHFL(mu, srcLaneB);
      float countB = WARP_SHFL(count, srcLaneB);
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(N);
      // don't care about final value of count, we know count == N
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / float(N), 0);
    }
  }
}

template<typename U> __device__ __host__ inline U rsqrt(U v) {
  return U(1) / ::sqrt(v);
}
template<> __device__ __host__ inline float rsqrt(float v) {
  return ::rsqrtf(v);
}
template<> __device__ __host__ inline double rsqrt(double v) {
  return ::rsqrt(v);
}

// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return nullptr;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};

template<typename T, typename U> __global__
void cuApplyLayerNorm(
  T* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ rstd,
  const T* __restrict__ vals,
  const int M,
  const int N,
  const U eps,
  const T* __restrict__ weight,
  const T* __restrict__ bias
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (int i1 = blockIdx.y; i1 < M; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, M, N, i1, mu, sigma2, buf);
    const T* lvals = vals + i1 * N;
    T* ovals = output_vals + i1 * N;
    U c_rstd = rsqrt(sigma2 + eps);
    printf("i1=%ld, sigma2=%lf, c_rstd=%lf\n", long(i1), double(sigma2), double(c_rstd));
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (weight != nullptr && bias != nullptr) {
      for (int i = thrx; i < N; i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = weight[i] * static_cast<T>(c_rstd * (curr - mu)) + bias[i];
      }
    } else {
      for (int i = thrx; i < N; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<T>(c_rstd * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      rstd[i1] = c_rstd;
    }
  }
}

template<typename T, typename U> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const T* grad_out,
    const int i1_end,
    const int N,
    const U* __restrict__ mean,
    const U* __restrict__ rstd
    )
{
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = mean[i1];
    U curr_rstd = rstd[i1];
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * N + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < N) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_grad_out = static_cast<U>(grad_out[load_idx]);
        warp_buf1[write_idx] = curr_grad_out;
        warp_buf2[write_idx] = curr_grad_out * (curr_input - curr_mean) * curr_rstd;
      } else {
        warp_buf1[write_idx] = U(0);
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0; k < blockDim.y; ++k) {
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      warp_buf1[write_idx] = U(0);
      warp_buf2[write_idx] = U(0);
    }
  }
}

template<typename T, typename U> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const T* grad_out,
    const int i1_end,
    const int N,
    const U* __restrict__ mean,
    const U* __restrict__ rstd
    )
{
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = mean[i1];
    U curr_rstd = rstd[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * N + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < N) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_grad_out = static_cast<U>(grad_out[load_idx]);
        warp_buf1[write_idx] += curr_grad_out;
        warp_buf2[write_idx] += curr_grad_out * (curr_input - curr_mean) * curr_rstd;
      }
    }
  }
}

template<typename T, typename U> __global__
void cuComputePartGradWeightBias(
    const T* __restrict__ grad_out,
    const T* __restrict__ input,
    const int M,
    const int N,
    const U* __restrict__ mean,
    const U* __restrict__ rstd,
    U eps,
    U* part_grad_weight,
    U* part_grad_bias)
{
    const int numsegs_M = (M + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
    const int segs_per_block = (numsegs_M + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y + 1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < M ? i1_beg_plus_one : M;
    const int row_stride = blockDim.x + 1;
    const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2, input, grad_out, i1_end, N, mean, rstd);
    for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end; i1_block += blockDim.y * blockDim.y) {
      cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2, input, grad_out, i1_end, N, mean, rstd);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0; k < blockDim.y; ++k) {
      int row1 = threadIdx.y + k * blockDim.y;
      int idx1 = row1 * row_stride + threadIdx.x;
      acc1 += warp_buf1[idx1];
      acc2 += warp_buf2[idx1];
    }
    warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
    warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + offset;
        int idx1 = row1 * row_stride + threadIdx.x;
        int idx2 = row2 * row_stride + threadIdx.x;
        warp_buf1[idx1] += warp_buf1[idx2];
        warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < N) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      part_grad_bias[blockIdx.y * N + i2] = warp_buf1[idx1] + warp_buf1[idx2];
      part_grad_weight[blockIdx.y * N + i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename T, typename U> __global__
void cuComputeGradWeightBias(
    const U* part_grad_weight,
    const U* part_grad_bias,
    const int part_size,
    const int M,
    const int N,
    T* grad_weight,
    T* grad_bias)
{
    // sum partial gradients for weight and bias
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < N) {
      // each warp does sequential reductions until reduced part_size is num_warps
      int num_warp_reductions = part_size / blockDim.y;
      U sum_weight = U(0);
      U sum_bias = U(0);
      const U* part_grad_weight_ptr = part_grad_weight + threadIdx.y * num_warp_reductions * N + i2;
      const U* part_grad_bias_ptr = part_grad_bias + threadIdx.y * num_warp_reductions * N + i2;
      for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
        sum_weight += part_grad_weight_ptr[warp_offset * N];
        sum_bias += part_grad_bias_ptr[warp_offset * N];
      }
      // inter-warp reductions
      const int nbsize3 = blockDim.x * blockDim.y / 2;
      for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[write_idx] = sum_weight;
          buf[write_idx+nbsize3] = sum_bias;
        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
          const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_weight += buf[read_idx];
          sum_bias += buf[read_idx + nbsize3];
        }
        __syncthreads();
      }
      // write out fully summed gradients
      if (threadIdx.y == 0) {
        grad_weight[i2] = sum_weight;
        grad_bias[i2] = sum_bias;
      }
    }
}

template<typename T, typename U> __global__
void cuComputeGradInput(
    const T* __restrict__ grad_out,
    const T* __restrict__ input,
    const int M,
    const int N,
    const U* __restrict__ mean,
    const U* __restrict__ rstd,
    U eps,
    const T* weight,
    T* grad_input)
{
  for (int i1=blockIdx.y; i1 < M; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = mean[i1];
    const U c_rstd = rstd[i1];
    const T* k_input = input + i1 * N;
    const T* k_grad_out = grad_out + i1 * N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (weight != nullptr) {
      int l = 4 * thrx;
      for (; l + 3 < N; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_grad_out[l + k]);
          sum_loss1 += c_loss * weight[l + k];
          sum_loss2 += c_loss * weight[l + k] * (c_h - c_mean) * c_rstd;
        }
      }
      for (; l < N; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_grad_out[l]);
        sum_loss1 += c_loss * weight[l];
        sum_loss2 += c_loss * weight[l] * (c_h - c_mean) * c_rstd;
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < N; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_grad_out[l + k]);
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_rstd;
        }
      }
      for (; l < N; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_grad_out[l]);
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_rstd;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
      sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer();
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2 * wrt_i] = sum_loss1;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2 * read_i];
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x] = sum_loss1;
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss1 = buf[2 * threadIdx.x];
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)N;
    U term1 = (U(1) / fH) * c_rstd;
    T* k_grad_input = grad_input + i1 * N;
    if (weight != nullptr) {
      for (int l = thrx; l < N; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_grad_out[l]);
        U f_grad_input = fH * c_loss * weight[l];
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_rstd * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < N; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_grad_out[l]);
        U f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_rstd * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
  }
}

template<typename T, typename U>
void HostApplyLayerNorm(
    T* output,
    U* mean,
    U* rstd,
    const T* input,
    int M,
    int N,
    double eps,
    const T* weight,
    const T* bias
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32, 4, 1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks(1, std::min((uint64_t)M, maxGridY), 1);
    int nshared = threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
    cuApplyLayerNorm<<<blocks, threads, nshared, stream>>>(
        output,
        mean,
        rstd,
        input,
        M, N,
        U(eps),
        weight, bias);
}

template<typename T, typename U>
void HostLayerNormGradient(
    const T* grad_out,
    const U* mean,
    const U* rstd,
    const Tensor &input,
    int M,
    int N,
    const T* weight,
    double eps,
    T* grad_input,
    T* grad_weight,
    T* grad_bias)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (grad_weight != nullptr && grad_bias != nullptr) {
      // compute grad_weight(j) and grad_bias(j)
      const int part_size = 16;
      const dim3 threads2(32, 4, 1);
      const dim3 blocks2((N + threads2.x - 1) / threads2.x, part_size, 1);
      const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(U);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      Tensor part_grad_weight = at::empty({part_size, N}, input.options().dtype(input.scalar_type() == at::ScalarType::Half ? at::ScalarType::Float : input.scalar_type()));
      Tensor part_grad_bias = at::empty_like(part_grad_weight);
      cuComputePartGradWeightBias<<<blocks2, threads2, nshared2, stream>>>(
          grad_out,
          input.data_ptr<T>(),
          M, N,
          mean,
          rstd,
          U(eps),
          part_grad_weight.data_ptr<U>(),
          part_grad_bias.data_ptr<U>());

      const dim3 threads3(32,8,1);
      const dim3 blocks3((N + threads2.x - 1) / threads2.x, 1, 1);
      const int nshared3 = threads3.x * threads3.y * sizeof(U);
      cuComputeGradWeightBias<<<blocks3, threads3, nshared3, stream>>>(
          part_grad_weight.data_ptr<U>(),
          part_grad_bias.data_ptr<U>(),
          part_size,
          M, N,
          grad_weight,
          grad_bias);
    }

    // compute grad_input
    if (grad_input != nullptr) {
      const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
      const dim3 blocks1(1, std::min((uint64_t)M, maxGridY), 1);
      const dim3 threads1(32,4,1);
      int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;
      cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
              grad_out,
              input.data_ptr<T>(),
              M, N,
              mean,
              rstd,
              U(eps),
              weight,
              grad_input);
    }
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_cuda(
  const Tensor& input,
  const Tensor& weight /* optional */,
  const Tensor& bias /* optional */,
  int64_t M,
  int64_t N,
  double eps)
{
  Tensor output = at::empty_like(input);
  Tensor mean = at::empty({M}, input.options().dtype(input.scalar_type() == at::ScalarType::Half ? at::ScalarType::Float : input.scalar_type()));
  Tensor rstd = at::empty_like(mean);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layer_norm_cuda_kernel",
    [&]() {
      using accscalar_t = at::acc_type<scalar_t, true>;
      HostApplyLayerNorm(
        output.data_ptr<scalar_t>(),
        mean.data_ptr<accscalar_t>(),
        rstd.data_ptr<accscalar_t>(),
        input.data_ptr<scalar_t>(),
        M, N,
        eps,
        weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
        bias.defined() ? bias.data_ptr<scalar_t>() : nullptr
      );
    });

  return std::make_tuple(output, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> non_differentiable_native_layer_norm_backward_cuda(
  const Tensor& grad_out,
  const Tensor& input,
  const Tensor& mean,
  const Tensor& rstd,
  const Tensor& weight,
  int64_t M,
  int64_t N,
  double eps,
  std::array<bool, 3> grad_input_mask)
{
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::native::empty_like(input);
  }
  if (grad_input_mask[1] || grad_input_mask[2]) {
    grad_weight = at::native::empty_like(weight);
    grad_bias = at::native::empty_like(weight);
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cuComputeGradInput",
    [&]() {
      using accscalar_t = at::acc_type<scalar_t, true>;
      HostLayerNormGradient(
        grad_out.data_ptr<scalar_t>(),
        mean.data_ptr<accscalar_t>(),
        rstd.data_ptr<accscalar_t>(),
        input,
        M, N,
        weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
        eps,
        grad_input.defined() ? grad_input.data_ptr<scalar_t>() : nullptr,
        grad_weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr,
        grad_bias.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr
      );
    });
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}}  // namespace at::native
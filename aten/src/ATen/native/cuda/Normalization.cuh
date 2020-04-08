#pragma once

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct SumOp {
  __device__ SumOp(const PTA& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PTA& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    accscalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, C10_WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffeling reduction.
// First each warp (of C10_WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most C10_WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than C10_WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[C10_WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % C10_WARP_SIZE == 0) {
    shared[tid / C10_WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / C10_WARP_SIZE && tid < C10_WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / C10_WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, bool train, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> output,
    const GenericPackedTensorAccessor<typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type, 1, RestrictPtrTraits, index_t> mean_,
    const GenericPackedTensorAccessor<typename std::conditional<train, stat_accscalar_t, stat_scalar_t>::type, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> bias,
    stat_accscalar_t epsilon) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  stat_accscalar_t gamma = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : static_cast<stat_accscalar_t>(1);
  stat_accscalar_t beta = bias.size(0) > 0 ? static_cast<stat_accscalar_t>(bias[plane]) : static_cast<stat_accscalar_t>(0);
  stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
  stat_accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<stat_accscalar_t>(1) / device_sqrt(static_cast<stat_accscalar_t>(var_or_invstd[plane]) + epsilon);
  }

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<input_scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}

template<typename T>
struct InvStd {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / device_sqrt(var + epsilon);
    }
    return invstd;
  }
};

template<typename T>
struct Var {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    return var;
  }
};

template <template<typename T> class VarTransform, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_collect_statistics_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    GenericPackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_transformed_var) {

  __shared__ int shared_n[2 * 2 * C10_WARP_SIZE + C10_WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  stat_accscalar_t* shared_avg_var = (stat_accscalar_t*) &shared_n[C10_WARP_SIZE];

  // first the reductions each thread does separately
  stat_accscalar_t avg = 0;
  stat_accscalar_t var_n = 0;
  int n = 0;
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      stat_accscalar_t v = input[batch][plane][x];
      stat_accscalar_t d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, C10_WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, C10_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, C10_WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % C10_WARP_SIZE == 0) {
    shared_n[tid / C10_WARP_SIZE] = n;
    shared_avg_var[tid / C10_WARP_SIZE * 2] = avg;
    shared_avg_var[tid / C10_WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < C10_WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / C10_WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / C10_WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y  / C10_WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
  }
  for (int i = 0; i < getMSB(C10_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, C10_WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, C10_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, C10_WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    if (save_mean.data() != NULL) {
      save_mean[plane] = avg;
    }
    if (save_transformed_var.data() != NULL) {
      save_transformed_var[plane] = VarTransform<stat_accscalar_t>{}(var_n / N, epsilon);
    }
    if (running_mean.data() != NULL) {
      running_mean[plane] = static_cast<stat_scalar_t>((1 - momentum) * running_mean[plane] + momentum * avg);
    }
    if (running_var.data() != NULL) {
      stat_accscalar_t unbiasedVar = var_n / (N - 1);
      running_var[plane] = static_cast<stat_scalar_t>((1 - momentum) * running_var[plane] + momentum * unbiasedVar);
    }
  }

}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_bias,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> running_mean,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> running_var,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_mean,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> save_invstd,
    bool train,
    stat_accscalar_t epsilon) {

  index_t plane = blockIdx.x;
  index_t N = grad_output.size(0) * grad_output.size(2);

  stat_accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = static_cast<stat_accscalar_t>(running_mean[plane]);
    invstd = static_cast<stat_accscalar_t>(1) / device_sqrt(static_cast<stat_accscalar_t>(running_var[plane]) + epsilon);
  }

  stat_accscalar_t weight_val = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : stat_accscalar_t(1);
  stat_accscalar_t norm = stat_accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<input_scalar_t, stat_accscalar_t, GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>> g(mean, input, grad_output);
  Float2<input_scalar_t, stat_accscalar_t> res = reduce<Float2<input_scalar_t, stat_accscalar_t>, GradOp<input_scalar_t, stat_accscalar_t,
                                                                                   GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>>>(g, grad_output, plane);
  stat_accscalar_t grad_output_sum = res.v1;
  stat_accscalar_t dot_p = res.v2;

  stat_accscalar_t grad_mean = grad_output_sum * norm;
  stat_accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  stat_accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        input_scalar_t go = grad_output[batch][plane][x];
        if (train) {
          stat_accscalar_t inp = input[batch][plane][x];
          stat_accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<input_scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<input_scalar_t>(go * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<stat_scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<stat_scalar_t>(grad_output_sum);
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_reduce_statistics_kernel(
    const GenericPackedTensorAccessor<accscalar_t, 2, RestrictPtrTraits, index_t> vec_mean,
    const GenericPackedTensorAccessor<accscalar_t, 2, RestrictPtrTraits, index_t> vec_invstd,
    GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t> mean,
    GenericPackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t> invstd,
    GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    const accscalar_t epsilon,
    const accscalar_t momentum,
    const GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> counts) {

  int feature_size = vec_mean.size(1);
  int world_size = vec_mean.size(0);

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // first the reductions each thread does separately
  for (int i = bid*blockDim.x+tid; i < feature_size; i += gridDim.x*blockDim.x) {
    accscalar_t avg = 0;
    accscalar_t var_n = 0;
    index_t n = 0;
    for (int j = 0; j < world_size; j++) {
      scalar_t count = counts[j];
      accscalar_t m = vec_mean[j][i];
      accscalar_t v = accscalar_t(1.0) / (vec_invstd[j][i]);
      v = (v * v - epsilon) * count;
      accscalar_t factor = 1.0 / (n + count);
      var_n += v + (avg - m) * (avg - m) * n * count * factor;
      avg = n * factor * avg + count * factor * m;
      n += count;
    }
    mean[i] = avg;
    invstd[i] = static_cast<accscalar_t>(1) / device_sqrt(var_n / n + epsilon);
    if (running_mean.data() != NULL) {
      running_mean[i] = static_cast<scalar_t>((1 - momentum) * running_mean[i] + momentum * avg);
    }
    accscalar_t unbiasedVar = var_n / (n - 1);
    if (running_var.data() != NULL) {
      running_var[i] = static_cast<scalar_t>((1 - momentum) * running_var[i] + momentum * unbiasedVar);
    }
  }

}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_reduce_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> invstd,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean_dy,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean_dy_xmu,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_bias) {

  index_t plane = blockIdx.x;
  index_t N = input.size(0) * input.size(2);

  stat_accscalar_t r_mean = mean[plane];
  stat_accscalar_t factor = invstd[plane];

  GradOp<input_scalar_t, stat_accscalar_t, GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>> g(r_mean, input, grad_output);
  Float2<input_scalar_t, stat_accscalar_t> res = reduce<Float2<input_scalar_t, stat_accscalar_t>, GradOp<input_scalar_t, stat_accscalar_t,
                                                                                   GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>>>(g, grad_output, plane);

  stat_accscalar_t norm = stat_accscalar_t(1) / N;
  if (threadIdx.x == 0) {
    if (grad_weight.size(0) > 0) {
      grad_weight[plane] = static_cast<stat_scalar_t>(res.v2 * factor);
    }
    if (grad_bias.size(0) > 0) {
      grad_bias[plane] = static_cast<stat_scalar_t>(res.v1);
    }
    if (mean_dy.size(0) > 0) {
      mean_dy[plane] = static_cast<stat_accscalar_t>(res.v1 * norm);
    }
    if (mean_dy_xmu.size(0) > 0) {
      mean_dy_xmu[plane] = static_cast<stat_accscalar_t>(res.v2 * norm);
    }
  }
}

template <typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_elemt_kernel(
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> input,
    const GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> invstd,
    const GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean_dy,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> mean_dy_xmu,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  stat_accscalar_t m_c = mean[plane];
  stat_accscalar_t m_dy_c = mean_dy[plane];
  stat_accscalar_t factor_1_c = invstd[plane];
  stat_accscalar_t factor_2_c = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : stat_accscalar_t(1);
  factor_2_c *= factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * mean_dy_xmu[plane];

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto g_i = grad_input[batch][plane];
    auto g_o = grad_output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      g_i[feature] = static_cast<input_scalar_t>((g_o[feature] - m_dy_c - (i[feature] - m_c) * factor_1_c) * factor_2_c);
    }
  }
}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return GenericPackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.generic_packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
void batch_norm_cuda_template(Tensor& output_, Tensor& save_mean_, Tensor& save_invstd_, const Tensor& input_, const Tensor& weight_, const Tensor& bias_,
                                                            const Tensor& running_mean_, const Tensor& running_var_,
                                                            bool train, double momentum, double epsilon) {

  TensorArg output_arg{ output_, "output", 1 },
            save_mean_arg{ save_mean_, "save_mean", 2 },
            save_invstd_arg{ save_invstd_, "save_invstd", 3 },
            input_arg{ input_, "input", 4 },
            weight_arg{ weight_, "weight", 5 },
            bias_arg{ bias_, "bias", 6 },
            run_mean_arg{ running_mean_, "running_mean", 7 },
            run_var_arg{ running_var_, "running_var", 8 };
  CheckedFrom c = "batch_norm_cuda";
  checkAllSameGPU(c, {output_arg, save_mean_arg, save_invstd_arg, input_arg, weight_arg, bias_arg, run_mean_arg, run_var_arg});

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);
  auto input = input_reshaped.generic_packed_accessor<input_scalar_t, 3, RestrictPtrTraits, index_t>();
  auto input_options = input_.options();
  if (input_.scalar_type() == at::ScalarType::Half || input_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  auto output = output_reshaped.generic_packed_accessor<input_scalar_t, 3, RestrictPtrTraits, index_t>();
  auto weight = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(weight_);
  auto bias = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(bias_);
  auto running_mean = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(running_var_);
  auto save_mean = save_mean_.generic_packed_accessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>();
  auto save_invstd = save_invstd_.generic_packed_accessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  // The input_transform kernel is pointwise, but we need to balance reading parameters (save_var/mean,
  // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
  // and good occupancy. Quite likely, we could go with even more blocks than 1024.
  // The various planes are independent, so we use blocks for them.
  int tf = std::max<int>(getNumThreads(input.size(2)/4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64/tf, 1);
  dim3 blocks_trans(input.size(1), std::max<int>(1, std::min<int>((256*1024)/input.size(1),
                                                                  (input.size(0)+tb-1)/tb)));
  blocks_trans.y = std::min<int>(blocks_trans.y, 65535);
  dim3 threads_trans(tf, tb);
  if (!train) {
    batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, false, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
      (input, output, running_mean, running_var, weight, bias, epsilon);
  } else {
    // for the reduction, we cannot use blocks for the batch dim, but if we have few threads in
    // the feature dimension, we'll use some threads for blocks
    dim3 blocks(input.size(1));
    tf = getNumThreads(input.size(2));
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
    batch_norm_collect_statistics_kernel<InvStd, input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
      (input, epsilon, momentum, running_mean, running_var, save_mean, save_invstd);
    batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, true, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
      (input, output, save_mean, save_invstd, weight, bias, epsilon);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template(const Tensor& grad_out_, const Tensor& input_, const Tensor& weight_,
                                                                     const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
                                                                     bool train, double epsilon, std::array<bool,3> grad_input_mask) {

  using accscalar_t = at::acc_type<stat_scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  auto input = input_reshaped.generic_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_output = grad_output_reshaped.generic_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_input = packed_accessor_or_dummy<input_scalar_t, 3, DefaultPtrTraits, index_t>(grad_input_reshaped);
  auto weight = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(weight_);
  auto grad_weight = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_weight_);
  auto grad_bias = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_bias_);
  auto running_mean = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(running_var_);
  auto save_mean = packed_accessor_or_dummy<accscalar_t, 1, DefaultPtrTraits, index_t>(save_mean_);
  auto save_invstd = packed_accessor_or_dummy<accscalar_t, 1, DefaultPtrTraits, index_t>(save_invstd_);

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

  batch_norm_backward_kernel<input_scalar_t, stat_scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, grad_output, grad_input, grad_weight, grad_bias, weight, running_mean, running_var,
     save_mean, save_invstd, train, epsilon);
  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

template<typename scalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_stats_cuda_template(const Tensor& input_, double epsilon) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);
  auto input = input_reshaped.generic_packed_accessor<scalar_t, 3, RestrictPtrTraits, index_t>();
  auto input_options = input_.options();
  dummy_mean_ = at::empty({0}, input_options);
  dummy_var_ = at::empty({0}, input_options);
  // promote only mean_/invstd_ precision
  if (input_.scalar_type() == at::ScalarType::Half || input_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_options);
  invstd_ = at::empty({n_input}, input_options);
  auto mean = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(invstd_);
  auto dummy_mean = dummy_mean_.generic_packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto dummy_invstd = dummy_var_.generic_packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  batch_norm_collect_statistics_kernel<InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(mean_, invstd_);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
void batch_norm_elemt_cuda_template(Tensor& output_, const Tensor& input_, const Tensor& weight_, const Tensor& bias_,
                                                                  const Tensor& mean_, const Tensor& invstd_,
                                                                  double epsilon) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);
  auto input = input_reshaped.generic_packed_accessor<input_scalar_t, 3, RestrictPtrTraits, index_t>();
  auto input_options = input_.options();
  if (input_.scalar_type() == at::ScalarType::Half || input_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  auto output = output_reshaped.generic_packed_accessor<input_scalar_t, 3, RestrictPtrTraits, index_t>();
  auto weight = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(weight_);
  auto bias = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(bias_);
  auto mean = packed_accessor_or_dummy<stat_accscalar_t, 1, RestrictPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<stat_accscalar_t, 1, RestrictPtrTraits, index_t>(invstd_);
  auto stream = at::cuda::getCurrentCUDAStream();

  // The input_transform kernel is pointwise, but we need to balance reading parameters (save_var/mean,
  // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
  // and good occupancy. Quiet likely, we could go with even more blocks than 1024.
  // The various planes are independent, so we use blocks for them.
  int tf = std::max<int>(getNumThreads(input.size(2)/4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64/tf, 1);
  dim3 blocks_trans(input.size(1), std::max<int>(1, std::min<int>((256*1024)/input.size(1),
                                                                  (input.size(0)+tb-1)/tb)));
  dim3 threads_trans(tf, tb);
  batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, true, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
    (input, output, mean, invstd, weight, bias, epsilon);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<typename scalar_t, typename accscalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_gather_stats_cuda_template(const Tensor& mean_, const Tensor& invstd_,
                                                                 const Tensor& running_mean_, const Tensor& running_var_,
                                                                 double momentum, double epsilon, const Tensor& counts_) {

  Tensor save_mean_;
  Tensor save_invstd_;

  auto features = mean_.size(1);
  auto input_options = mean_.options();
  if (mean_.scalar_type() == at::ScalarType::Half || mean_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  save_mean_ = at::empty({features}, input_options);
  save_invstd_ = at::empty({features}, input_options);

  auto mean = packed_accessor_or_dummy<accscalar_t, 2, RestrictPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t, 2, RestrictPtrTraits, index_t>(invstd_);
  auto running_mean = packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(running_var_);
  auto counts = packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(counts_);

  auto save_mean = save_mean_.generic_packed_accessor<accscalar_t, 1, RestrictPtrTraits, index_t>();
  auto save_invstd = save_invstd_.generic_packed_accessor<accscalar_t, 1, RestrictPtrTraits, index_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  int block = getNumThreads(features);
  int grid = std::max<int>(1, features/block);
  batch_norm_reduce_statistics_kernel<scalar_t, accscalar_t, index_t> <<<grid, block, 0, stream>>>
      (mean, invstd, save_mean, save_invstd, running_mean, running_var, epsilon, momentum, counts);
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(save_mean_, save_invstd_);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda_template(const Tensor& grad_out_, const Tensor& input_,
                                                                                    const Tensor& mean_, const Tensor& invstd_, const Tensor& weight_,
                                                                                    const bool input_g, const bool weight_g, const bool bias_g) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor mean_dy_;
  Tensor mean_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (input_g) {
    mean_dy_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    mean_dy_xmu_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (weight_g) {
    grad_weight_ = at::empty({n_input}, weight_.options());
  }
  if (bias_g) {
    grad_bias_ = at::empty({n_input}, weight_.options());
  }

  auto input = input_reshaped.generic_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_output = grad_output_reshaped.generic_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_weight = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_weight_);
  auto grad_bias = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(grad_bias_);
  auto mean = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(invstd_);
  auto mean_dy = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_dy_);
  auto mean_dy_xmu = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_dy_xmu_);

  auto batch_size = input_reshaped.size(0);
  auto feature_size = input_reshaped.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();

  int block_y = std::min<int>(lastPow2(batch_size), MAX_BLOCK_SIZE/C10_WARP_SIZE);
  // We want block_x to be at least a warp width
  int block_x = std::min<int>(std::max<int>(getNumThreads(feature_size), C10_WARP_SIZE), MAX_BLOCK_SIZE/block_y);
  const dim3 block(block_x, block_y);
  const dim3 grid(n_input);

  batch_norm_backward_reduce_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<grid, block, 0, stream>>>
    (input, grad_output, mean, invstd, mean_dy, mean_dy_xmu, grad_weight, grad_bias);
  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(mean_dy_, mean_dy_xmu_, grad_weight_, grad_bias_);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_cuda_template(const Tensor& grad_out_, const Tensor& input_,
                                               const Tensor& mean_, const Tensor& invstd_,
                                               const Tensor& weight_, const Tensor& mean_dy_, const Tensor& mean_dy_xmu_) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());
  auto grad_input_reshaped = at::empty_like(input_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);

  auto input = input_reshaped.generic_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_input = grad_input_reshaped.generic_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_output = grad_output_reshaped.generic_packed_accessor<input_scalar_t, 3, DefaultPtrTraits, index_t>();
  auto mean = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(invstd_);
  auto weight = packed_accessor_or_dummy<stat_scalar_t, 1, DefaultPtrTraits, index_t>(weight_);
  auto mean_dy = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_dy_);
  auto mean_dy_xmu = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(mean_dy_xmu_);

  auto stream = at::cuda::getCurrentCUDAStream();

  // The kernel is pointwise, but we need to balance reading parameters (save_var/mean,
  // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
  // and good occupancy. Quiet likely, we could go with even more blocks than 1024.
  // The various planes are independent, so we use blocks for them.
  int tf = std::max<int>(getNumThreads(input.size(2)/4),
                         std::min<int>(getNumThreads(input.size(2)), 64));
  int tb = std::max<int>(64/tf, 1);
  dim3 blocks_trans(input.size(1), std::max<int>(1, std::min<int>((256*1024)/input.size(1),
                                                                  (input.size(0)+tb-1)/tb)));
  dim3 threads_trans(tf, tb);
  batch_norm_backward_elemt_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
    (input, grad_output, mean, invstd, weight, mean_dy, mean_dy_xmu, grad_input);
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input_reshaped.view(input_.sizes());
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_update_stats_cuda_template(
        const Tensor& input_, const Tensor& running_mean_, const Tensor& running_var_, double momentum) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_channels = input_.size(1);
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

  auto input_options = input_.options();
  if (input_.scalar_type() == at::ScalarType::Half || input_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  Tensor save_mean_ = at::empty({n_channels}, input_options);
  Tensor save_var_ = at::empty({n_channels}, input_options);

  auto input = input_reshaped.generic_packed_accessor<input_scalar_t, 3, RestrictPtrTraits, index_t>();
  auto running_mean = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<stat_scalar_t, 1, RestrictPtrTraits, index_t>(running_var_);
  auto save_mean = save_mean_.generic_packed_accessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>();
  auto save_var = save_var_.generic_packed_accessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  // for the reduction, we cannot use blocks for the batch dim, but if we have few threads in
  // the feature dimension, we'll use some threads for blocks
  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  // NB: epsilon is unused by the Var transform, so we set it to 0
  batch_norm_collect_statistics_kernel<Var, input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, 0., momentum, running_mean, running_var, save_mean, save_var);
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(save_mean_, save_var_);
}

} } // namespace at::native

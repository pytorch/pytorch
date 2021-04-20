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
// of the data. Then there is a double-shuffling reduction.
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

constexpr int ELEMENTS_PER_ITER = 4; // enables concurrency within each thread to hide latency
constexpr int ELEMENTS_PER_THREAD = 16;
constexpr int OPTIMAL_TILE_W = 32;
constexpr int MAX_H_BLOCK = 128;

__host__ int div_roundup(int x, int y) {
  return lastPow2(1 + (x-1)/y);
}

__host__ void flexible_launch_configs(
      const int reduction,
      const int stride,
      dim3 &block,
      dim3 &grid,
      const bool coop_flag = false) {
  int block_x = std::min(lastPow2(stride), OPTIMAL_TILE_W);
  int block_y = std::min(lastPow2(div_roundup(reduction , ELEMENTS_PER_THREAD)),
                         MAX_BLOCK_SIZE / block_x);
  if (block_x * block_y != MAX_BLOCK_SIZE) {
    block_x = std::min(lastPow2(stride), MAX_BLOCK_SIZE / block_y);
  }

  int grid_x = div_roundup(stride, block_x);
  int grid_y = std::min(div_roundup(reduction, block_y * ELEMENTS_PER_THREAD), MAX_H_BLOCK);
  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}

template<typename T, typename C>
__device__ __forceinline__ void welford_merge_element(C& count,
                                                      T& mean,
                                                      T& m2n,
                                                      const C& count_new,
                                                      const T& mean_new,
                                                      const T& m2n_new) {
      T factor = T(1.0) / ::max(1, (count + count_new));
      T delta0 = mean - mean_new;
      mean = (mean_new * count_new + mean * count) * factor;
      m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
      count += count_new;
}

// merge mean/m2n among threadIdx.y within block
template<typename T, typename C>
__device__ __forceinline__ void welford_merge_block_vertical(C& count,
                                                             T& mean,
                                                             T& m2n,
                                                             C* shmem_count,
                                                             T* shmem_mean,
                                                             T* shmem_m2n) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y/2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset*2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, count_new, mean_new, m2n_new);
    }
  }
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
                                                                                   GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>> >(g, grad_output, plane);

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
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy,
    GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy_xmu,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    GenericPackedTensorAccessor<stat_scalar_t, 1, DefaultPtrTraits, index_t> grad_bias) {

  index_t plane = blockIdx.x;

  stat_accscalar_t r_mean = mean[plane];
  stat_accscalar_t factor = invstd[plane];

  GradOp<input_scalar_t, stat_accscalar_t, GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>> g(r_mean, input, grad_output);
  Float2<input_scalar_t, stat_accscalar_t> res = reduce<Float2<input_scalar_t, stat_accscalar_t>, GradOp<input_scalar_t, stat_accscalar_t,
                                                                                   GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t>> >(g, grad_output, plane);

  if (threadIdx.x == 0) {
    if (grad_weight.size(0) > 0) {
      grad_weight[plane] = static_cast<stat_scalar_t>(res.v2 * factor);
    }
    if (grad_bias.size(0) > 0) {
      grad_bias[plane] = static_cast<stat_scalar_t>(res.v1);
    }
    if (sum_dy.size(0) > 0) {
      sum_dy[plane] = static_cast<stat_accscalar_t>(res.v1);
    }
    if (sum_dy_xmu.size(0) > 0) {
      sum_dy_xmu[plane] = static_cast<stat_accscalar_t>(res.v2);
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
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy,
    const GenericPackedTensorAccessor<stat_accscalar_t, 1, DefaultPtrTraits, index_t> sum_dy_xmu,
    GenericPackedTensorAccessor<input_scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    const int* __restrict__ numel, const int world_size) {

  int64_t div = 0;
  for (int i = 0; i < world_size; i ++) {
    div += numel[i];
  }

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  stat_accscalar_t m_c = mean[plane];
  stat_accscalar_t m_dy_c = sum_dy[plane] / div;
  stat_accscalar_t factor_1_c = invstd[plane];
  stat_accscalar_t factor_2_c = weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane]) : stat_accscalar_t(1);
  factor_2_c *= factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu[plane] / div;

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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // for the reduction, we cannot use blocks for the batch dim, but if we have few threads in
    // the feature dimension, we'll use some threads for blocks
    dim3 blocks(input.size(1));
    tf = getNumThreads(input.size(2));
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

    batch_norm_collect_statistics_kernel<InvStd, input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
      (input, epsilon, momentum, running_mean, running_var, save_mean, save_invstd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, true, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
      (input, output, save_mean, save_invstd, weight, bias, epsilon);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(save_mean_, save_invstd_);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda_template(const Tensor& grad_out_, const Tensor& input_,
                                                                                    const Tensor& mean_, const Tensor& invstd_, const Tensor& weight_,
                                                                                    const bool input_g, const bool weight_g, const bool bias_g) {

  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor sum_dy_;
  Tensor sum_dy_xmu_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (input_g) {
    sum_dy_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    sum_dy_xmu_ = at::empty_like(mean_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
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
  auto sum_dy = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_);
  auto sum_dy_xmu = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_xmu_);

  auto batch_size = input_reshaped.size(0);
  auto feature_size = input_reshaped.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();

  int block_y = std::min<int>(lastPow2(batch_size), MAX_BLOCK_SIZE/C10_WARP_SIZE);
  // We want block_x to be at least a warp width
  int block_x = std::min<int>(std::max<int>(getNumThreads(feature_size), C10_WARP_SIZE), MAX_BLOCK_SIZE/block_y);
  const dim3 block(block_x, block_y);
  const dim3 grid(n_input);

  batch_norm_backward_reduce_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, index_t> <<<grid, block, 0, stream>>>
    (input, grad_output, mean, invstd, sum_dy, sum_dy_xmu, grad_weight, grad_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(sum_dy_, sum_dy_xmu_, grad_weight_, grad_bias_);
}

template<typename input_scalar_t, typename stat_scalar_t, typename index_t>
Tensor batch_norm_backward_elemt_cuda_template(const Tensor& grad_out_, const Tensor& input_,
                                               const Tensor& mean_, const Tensor& invstd_,
                                               const Tensor& weight_, const Tensor& sum_dy_, const Tensor& sum_dy_xmu_, const Tensor& count) {

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
  auto sum_dy = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_);
  auto sum_dy_xmu = packed_accessor_or_dummy<stat_accscalar_t, 1, DefaultPtrTraits, index_t>(sum_dy_xmu_);

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
    (input, grad_output, mean, invstd, weight, sum_dy, sum_dy_xmu, grad_input, count.data_ptr<int>(), count.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(save_mean_, save_var_);
}

// welford kernel for c last tensor calculating mean/biased_variance/unbiased_variance
// original apex name: welford_kernel_c_last
template
   <template<typename T> class VarTransform,
    typename scalar_t,
    typename accscalar_t,
    int PARALLEL_LOADS>
__global__ void
batch_norm_collect_statistics_channels_last_kernel(
      const scalar_t* __restrict__ input,
      accscalar_t* __restrict__ out_mean,
      accscalar_t* __restrict__ out_invstd,
      volatile accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride,
      accscalar_t epsilon) {
  // hide latency with concurrency
  accscalar_t x_mean[PARALLEL_LOADS];
  accscalar_t m_2_n[PARALLEL_LOADS];
  int count[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    x_mean[i] = accscalar_t(0);
    m_2_n[i] = accscalar_t(0);
    count[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_math[PARALLEL_LOADS];
    accscalar_t x_count_inv[PARALLEL_LOADS];
    accscalar_t is_valid[PARALLEL_LOADS];

    // load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_math[j] = input[address_base];
        count[j]++;
        x_count_inv[j] = accscalar_t(1) / count[j];
        is_valid[j] = accscalar_t(1);
      } else {
        x_math[j] = accscalar_t(0);
        x_count_inv[j] = accscalar_t(0);
        is_valid[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

    // calculate mean/m2n with welford
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      accscalar_t delta0 = x_math[j] - x_mean[j];
      x_mean[j] += delta0 * x_count_inv[j];
      accscalar_t delta1 = x_math[j] - x_mean[j];
      m_2_n[j] += delta0 * delta1 * is_valid[j];
    }
  }

  // thread reduction to accumulate mean/m_2_n/count between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    welford_merge_element(count[0], x_mean[0], m_2_n[0], count[j], x_mean[j], m_2_n[j]);
  }

  // release x_mean / m_2_n
  auto mean_th = x_mean[0];
  auto m2_th = m_2_n[0];
  auto count_th = count[0];

  // block-wise reduction with shared memory (since reduction cannot be done within a warp)
  static __shared__ accscalar_t shmem_mean[MAX_BLOCK_SIZE];
  static __shared__ accscalar_t shmem_m2n[MAX_BLOCK_SIZE];
  static __shared__ int shmem_count[MAX_BLOCK_SIZE];

  welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count, shmem_mean, shmem_m2n);

  if (gridDim.y > 1) {
    volatile accscalar_t* staging_mean = staging_data;
    volatile accscalar_t* staging_m2n = &staging_data[stride*gridDim.y];
    volatile int* staging_count = reinterpret_cast<volatile int*>(&staging_m2n[stride*gridDim.y]);

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_mean[address_base] = mean_th;
      staging_m2n[address_base] = m2_th;
      staging_count[address_base] = count_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y-1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      count_th = 0;
      mean_th = accscalar_t(0.0);
      m2_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        int count_new = c_offset < stride ? staging_count[address_base] : 0;
        accscalar_t mean_new = c_offset < stride ? staging_mean[address_base] : accscalar_t(0.0);
        accscalar_t m2n_new = c_offset < stride ? staging_m2n[address_base] : accscalar_t(0.0);

        welford_merge_element(count_th, mean_th, m2_th, count_new, mean_new, m2n_new);
      }

      welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count, shmem_mean, shmem_m2n);
      if (threadIdx.y == 0 && c_offset < stride) {
        out_mean[c_offset] = static_cast<accscalar_t>(mean_th);
        out_invstd[c_offset] = VarTransform<accscalar_t>{}(m2_th/count_th, epsilon);
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      out_mean[c_offset] = static_cast<accscalar_t>(mean_th);
      out_invstd[c_offset] = VarTransform<accscalar_t>{}(m2_th/count_th, epsilon);
    }
  }
}

// elementwise BN kernel
// original apex name: batchnorm_forward_c_last_kernel
template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int PARALLEL_LOADS>
__global__ void batch_norm_transform_input_channels_last_kernel(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ z,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      const layerscalar_t* __restrict__ weight,
      const layerscalar_t* __restrict__ shift,
      scalar_t* __restrict__ out,
      const int reduction_size,
      const int stride,
      const bool fuse_relu) {
  // tensor dimension (m,c)
  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  auto m_c = mean[c_offset];
  auto inv_std_c = static_cast<accscalar_t>(inv_std[c_offset]);
  auto w_c = weight == nullptr ? accscalar_t(1.0) : static_cast<accscalar_t>(weight[c_offset]);
  auto s_c = shift == nullptr ? accscalar_t(0.0) : static_cast<accscalar_t>(shift[c_offset]);

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        auto tmp = w_c * (static_cast<accscalar_t>(input[address_base]) - m_c ) * inv_std_c + s_c;
        if (z != nullptr) {
          tmp += z[address_base];
        }
        out[address_base] = (fuse_relu && tmp <= accscalar_t(0.0) ? scalar_t(0.0) : static_cast<scalar_t>(tmp));
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}

template<typename T>
__device__ __forceinline__ void merge_block_vertical_backward(T& sum_dy,
    T& sum_dy_xmu,
    T* shmem_sum_dy,
    T* shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y/2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset*2) {
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;

      sum_dy += shmem_sum_dy[address];
      sum_dy_xmu += shmem_sum_dy_xmu[address];
    }
  }
}

// batchnorm backward kernel for c last tensor
// original apex name: reduce_bn_c_last_kernel
template
   <typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int PARALLEL_LOADS>
__global__ void batch_norm_backward_reduce_channels_last_kernel(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grad_output,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      accscalar_t* __restrict__ sum_dy_o,
      accscalar_t* __restrict__ sum_dy_xmu_o,
      layerscalar_t* __restrict__ grad_weight,
      layerscalar_t* __restrict__ grad_bias,
      volatile accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride) {

  // hide latency with concurrency
  accscalar_t sum_dy[PARALLEL_LOADS];
  accscalar_t sum_dy_xmu[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    sum_dy[i] = accscalar_t(0);
    sum_dy_xmu[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  auto r_mean = mean[c_offset];
  auto factor = inv_std[c_offset];

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_input[PARALLEL_LOADS];
    accscalar_t x_grad_output[PARALLEL_LOADS];

    // load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_input[j] = input[address_base];
        x_grad_output[j] = grad_output[address_base];
      } else {
        x_input[j] = accscalar_t(0);
        x_grad_output[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

    // calculate sum_dy / sum_dy_xmu
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      sum_dy[j] += x_grad_output[j];
      sum_dy_xmu[j] += x_grad_output[j] * (x_input[j] - r_mean);
    }
  }

  // thread reduction to accumulate sum_dy / sum_dy_xmu between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    sum_dy[0] += sum_dy[j];
    sum_dy_xmu[0] += sum_dy_xmu[j];
  }

  // release array of registers
  auto sum_dy_th = sum_dy[0];
  auto sum_dy_xmu_th = sum_dy_xmu[0];

  // block-wise reduction with shared memory (since reduction cannot be done within a warp)
  static __shared__ accscalar_t shmem_sum_dy[MAX_BLOCK_SIZE];
  static __shared__ accscalar_t shmem_sum_dy_xmu[MAX_BLOCK_SIZE];

  merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);

  if (gridDim.y > 1) {
    volatile accscalar_t* staging_sum_dy = staging_data;
    volatile accscalar_t* staging_sum_dy_xmu = &staging_data[stride*gridDim.y];

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_sum_dy[address_base] = sum_dy_th;
      staging_sum_dy_xmu[address_base] = sum_dy_xmu_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y-1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      sum_dy_th = accscalar_t(0.0);
      sum_dy_xmu_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        sum_dy_th += (c_offset < stride ? staging_sum_dy[address_base] : accscalar_t(0.0));
        sum_dy_xmu_th += (c_offset < stride ? staging_sum_dy_xmu[address_base] : accscalar_t(0.0));
      }

      merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);
      if (threadIdx.y == 0 && c_offset < stride) {
        if (grad_bias != nullptr) {
          grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
        }
        if (grad_weight != nullptr) {
          grad_weight[c_offset] = static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
        }
        //mean_dy[c_offset] = sum_dy_th / reduction_size;
        //mean_dy_xmu[c_offset] = sum_dy_xmu_th / reduction_size;
        sum_dy_o[c_offset] = sum_dy_th;
        sum_dy_xmu_o[c_offset] = sum_dy_xmu_th;
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      if (grad_bias != nullptr) {
        grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
      }
      if (grad_weight != nullptr) {
        grad_weight[c_offset] = static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
      }
      //mean_dy[c_offset] = sum_dy_th / reduction_size;
      //mean_dy_xmu[c_offset] = sum_dy_xmu_th / reduction_size;
      sum_dy_o[c_offset] = sum_dy_th;
      sum_dy_xmu_o[c_offset] = sum_dy_xmu_th;
    }
  }
}

// elementwise BN kernel
// original apex name: batchnorm_backward_c_last_kernel
template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int PARALLEL_LOADS>
__global__ void batch_norm_backward_elemt_channels_last_kernel(
      const scalar_t* __restrict__ grad_output,
      const scalar_t* __restrict__ input,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      const layerscalar_t* __restrict__ weight,
      const accscalar_t* __restrict__ sum_dy,
      const accscalar_t* __restrict__ sum_dy_xmu,
      const int* __restrict__ numel,
      scalar_t* __restrict__ grad_input,
      const int64_t world_size,
      const int reduction_size,
      const int stride) {
  int64_t div = 0;
  for (int i = 0; i < world_size; i++) {
    div += numel[i];
  }
  // tensor dimension (m,c)
  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  auto m_c = mean[c_offset];
  auto m_dy_c = sum_dy[c_offset] / div;
  auto factor_1_c = inv_std[c_offset];
  auto factor_2_c = (weight == nullptr? accscalar_t(1.0) : static_cast<accscalar_t>(weight[c_offset])) * factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu[c_offset] / div;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        grad_input[address_base] = static_cast<scalar_t>(
            (static_cast<accscalar_t>(grad_output[address_base]) - m_dy_c -
            (static_cast<accscalar_t>(input[address_base]) - m_c) * factor_1_c)
            * factor_2_c);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}

template<typename scalar_t>
std::tuple<Tensor, Tensor> batch_norm_stats_channels_last_cuda_template(const Tensor& input, double epsilon) {
  using accscalar_t = at::acc_type<scalar_t, true>;

  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  auto scalar_type = input.scalar_type() == at::kHalf ? at::kFloat : input.scalar_type();
  auto option = input.options().dtype(scalar_type);

  at::Tensor out_invstd = at::empty({stride}, option);
  at::Tensor out_mean = at::empty({stride}, option);

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);

  at::Tensor staging_data;
  at::Tensor semaphores;
  if (grid.y > 1) {
    staging_data = at::empty({4*stride*grid.y}, option);
    semaphores = at::zeros({grid.x}, input.options().dtype(at::kInt));
  }

  accscalar_t* staging_data_ptr = grid.y > 1 ? staging_data.data_ptr<accscalar_t>() : nullptr;
  int* semaphores_ptr = grid.y > 1 ? semaphores.data_ptr<int>() : nullptr;
  batch_norm_collect_statistics_channels_last_kernel<InvStd, scalar_t, accscalar_t, ELEMENTS_PER_ITER>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      out_mean.data_ptr<accscalar_t>(),
      out_invstd.data_ptr<accscalar_t>(),
      staging_data_ptr,
      semaphores_ptr,
      reduction_size,
      stride,
      epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(out_mean, out_invstd);
}

void batch_norm_elemt_channels_last_cuda_template(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& shift,  // bias of BN
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    double epsilon,
    const at::optional<at::Tensor>& z = c10::nullopt,  // bias after BN
    const bool fuse_relu = false) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::kHalf && weight.defined() && weight.scalar_type() == at::kFloat) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "batchnorm_forward", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batch_norm_transform_input_channels_last_kernel<scalar_t, accscalar_t, accscalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data_ptr<scalar_t>(),
          z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr,
          mean.data_ptr<accscalar_t>(),
          inv_std.data_ptr<accscalar_t>(),
          weight.defined() ? weight.data_ptr<accscalar_t>() : nullptr,
          shift.defined() ? shift.data_ptr<accscalar_t>() : nullptr,
          output.data_ptr<scalar_t>(),
          reduction_size,
          stride,
          fuse_relu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  } else {
    if (weight.defined()){
      TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "batchnorm_forward: input.scalar_type() ", input.scalar_type(),
        " is not supported with weight.scalar_type() ", weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "batchnorm_forward", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batch_norm_transform_input_channels_last_kernel<scalar_t, accscalar_t, scalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data_ptr<scalar_t>(),
          z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr,
          mean.data_ptr<accscalar_t>(),
          inv_std.data_ptr<accscalar_t>(),
          weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
          shift.defined() ? shift.data_ptr<scalar_t>(): nullptr,
          output.data_ptr<scalar_t>(),
          reduction_size,
          stride,
          fuse_relu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
batch_norm_backward_reduce_cuda_channels_last_template(const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::Tensor& weight,
    const bool input_g, const bool weight_g, const bool bias_g) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  at::Tensor sumn_dy = at::empty({stride}, mean.options());
  at::Tensor sum_dy_xmu = at::empty({stride}, mean.options());

  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (weight.defined()) {
    grad_weight = at::empty({stride}, weight.options());
    grad_bias = at::empty({stride}, weight.options());
  } else {
    // because I cannot return an uninitialized at::Tensor
    grad_weight = at::empty({0}, mean.options());
    grad_bias = at::empty({0}, mean.options());
  }

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);

  at::Tensor staging_data;
  at::Tensor semaphores;
  if (grid.y > 1) {
    staging_data = at::empty({2*stride*grid.y}, mean.options());
    semaphores = at::zeros({grid.x}, input.options().dtype(at::kInt));
  }
  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::kHalf && weight.defined() && weight.scalar_type() == at::kFloat) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "batchnorm_backward_reduce", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      accscalar_t* staging_data_ptr = grid.y > 1 ? staging_data.data_ptr<accscalar_t>() : nullptr;
      int* semaphores_ptr = grid.y > 1 ? semaphores.data_ptr<int>() : nullptr;
      batch_norm_backward_reduce_channels_last_kernel<scalar_t, accscalar_t, accscalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data_ptr<scalar_t>(),
          grad_output.data_ptr<scalar_t>(),
          mean.data_ptr<accscalar_t>(),
          inv_std.data_ptr<accscalar_t>(),
          sumn_dy.data_ptr<accscalar_t>(),
          sum_dy_xmu.data_ptr<accscalar_t>(),
          weight.defined() ? grad_weight.data_ptr<accscalar_t>() : nullptr,
          weight.defined() ?grad_bias.data_ptr<accscalar_t>() : nullptr,
          staging_data_ptr,
          semaphores_ptr,
          reduction_size,
          stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "batchnorm_backward_reduce: input.scalar_type() ", input.scalar_type(),
        " is not supported with weight.scalar_type() ", weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "batchnorm_backward_reduce", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      accscalar_t* staging_data_ptr = grid.y > 1 ? staging_data.data_ptr<accscalar_t>() : nullptr;
      int* semaphores_ptr = grid.y > 1 ? semaphores.data_ptr<int>() : nullptr;
      batch_norm_backward_reduce_channels_last_kernel<scalar_t, accscalar_t, scalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data_ptr<scalar_t>(),
          grad_output.data_ptr<scalar_t>(),
          mean.data_ptr<accscalar_t>(),
          inv_std.data_ptr<accscalar_t>(),
          sumn_dy.data_ptr<accscalar_t>(),
          sum_dy_xmu.data_ptr<accscalar_t>(),
          weight.defined() ? grad_weight.data_ptr<scalar_t>() : nullptr,
          weight.defined() ? grad_bias.data_ptr<scalar_t>() : nullptr,
          staging_data_ptr,
          semaphores_ptr,
          reduction_size,
          stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }

  return std::make_tuple(sumn_dy, sum_dy_xmu, grad_weight, grad_bias);
}

at::Tensor batch_norm_backward_elemt_channels_last_cuda_template(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::Tensor& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count) {
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  at::Tensor grad_input = at::empty_like(input, input.suggest_memory_format());

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::kHalf && weight.defined() && weight.scalar_type() == at::kFloat) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "batchnorm_backward_element", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batch_norm_backward_elemt_channels_last_kernel<scalar_t, accscalar_t, accscalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          grad_output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          mean.data_ptr<accscalar_t>(),
          inv_std.data_ptr<accscalar_t>(),
          weight.defined() ? weight.data_ptr<accscalar_t>() : nullptr,
          sum_dy.data_ptr<accscalar_t>(),
          sum_dy_xmu.data_ptr<accscalar_t>(),
          count.data_ptr<int>(),
          grad_input.data_ptr<scalar_t>(),
          count.numel(),
          reduction_size,
          stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "batchnorm_backward_element: input.scalar_type() ", input.scalar_type(),
        " is not supported with weight.scalar_type() ", weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "batchnorm_backward_element", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batch_norm_backward_elemt_channels_last_kernel<scalar_t, accscalar_t, scalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          grad_output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          mean.data_ptr<accscalar_t>(),
          inv_std.data_ptr<accscalar_t>(),
          weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
          sum_dy.data_ptr<accscalar_t>(),
          sum_dy_xmu.data_ptr<accscalar_t>(),
          count.data_ptr<int>(),
          grad_input.data_ptr<scalar_t>(),
          count.numel(),
          reduction_size,
          stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }

  return grad_input;
}

} } // namespace at::native

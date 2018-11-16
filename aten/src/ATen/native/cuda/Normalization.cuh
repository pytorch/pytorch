#pragma once

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include "ATen/native/cuda/Normalization.cuh"

namespace at { namespace native {

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;

// take these out when ROCm implements std:: math functions
#include <math.h>
template <typename scalar_t>
static __forceinline__ __device__ scalar_t device_sqrt(scalar_t val);

template <>
__forceinline__ __device__ float device_sqrt(float val) {
  return ::sqrtf(val);
}

template <>
__forceinline__ __device__ double device_sqrt(double val) {
  return ::sqrt(val);
}

#else
constexpr int WARP_SIZE = 32;

template<typename scalar_t>
__forceinline__ __device__ double device_sqrt(scalar_t val) {
  return std::sqrt(val);
}
#endif

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
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
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
// First each warp (of WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than WARP_SIZE**2 threads.
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
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % WARP_SIZE == 0) {
    shared[tid / WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / WARP_SIZE && tid < WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename scalar_t, typename accscalar_t, bool train, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> input,
    PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> output,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> mean_,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> bias,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(var_or_invstd[plane]) + epsilon);
  }

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}


template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> input,
    const accscalar_t epsilon,
    const accscalar_t momentum,
    PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    PackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    PackedTensorAccessor<accscalar_t, 1, RestrictPtrTraits, index_t> save_invstd) {

  __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  accscalar_t* shared_avg_var = (accscalar_t*) &shared_n[WARP_SIZE];

  // first the reductions each thread does separately
  accscalar_t avg = 0;
  accscalar_t var_n = 0;
  int n = 0;
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      accscalar_t v = input[batch][plane][x];
      accscalar_t d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning  
  __syncthreads();
  if (tid % WARP_SIZE == 0) {
    shared_n[tid / WARP_SIZE] = n;
    shared_avg_var[tid / WARP_SIZE * 2] = avg;
    shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid] : 0);
    var_n = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid + 1] : 0);
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    accscalar_t invstd = 0;
    if (var_n != static_cast<accscalar_t>(0) || epsilon != static_cast<accscalar_t>(0)) {
      invstd = static_cast<accscalar_t>(1) / device_sqrt(var_n / N + epsilon);
    }
    save_mean[plane] = avg;
    save_invstd[plane] = invstd;
    if (running_mean.data() != NULL) {
      running_mean[plane] = static_cast<scalar_t>((1 - momentum) * running_mean[plane] + momentum * avg);
    }
    if (running_var.data() != NULL) {
      accscalar_t unbiasedVar = var_n / (N - 1);
      running_var[plane] = static_cast<scalar_t>((1 - momentum) * running_var[plane] + momentum * unbiasedVar);
    }
  }

}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel(
    const PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t> input,
    const PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> grad_bias,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> running_mean,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> running_var,
    const PackedTensorAccessor<accscalar_t, 1, DefaultPtrTraits, index_t> save_mean,
    const PackedTensorAccessor<accscalar_t, 1, DefaultPtrTraits, index_t> save_invstd,
    bool train,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;
  index_t N = grad_output.size(0) * grad_output.size(2);

  accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = static_cast<accscalar_t>(running_mean[plane]);
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(running_var[plane]) + epsilon);
  }

  accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
  accscalar_t norm = accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<scalar_t, accscalar_t, PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t>> g(mean, input, grad_output);
  Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t,
                                                                                   PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t>>>(g, grad_output, plane);
  accscalar_t grad_output_sum = res.v1;
  accscalar_t dot_p = res.v2;

  accscalar_t grad_mean = grad_output_sum * norm;
  accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        scalar_t go = grad_output[batch][plane][x];
        if (train) {
          scalar_t inp = input[batch][plane][x];
          accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<scalar_t>(go * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<scalar_t>(grad_output_sum);
    }
  }
}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

template<typename scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda_template(const Tensor& input_, const Tensor& weight_, const Tensor& bias_,
                                                            const Tensor& running_mean_, const Tensor& running_var_,
                                                            bool train, double momentum, double epsilon) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor save_mean_;
  Tensor save_invstd_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
  auto output_reshaped = at::empty_like(input_reshaped);

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);
  auto input = input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, index_t>();
  auto input_options = input_.options();
  if (input_.type().scalarType() == at::ScalarType::Half) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  if (train) {
    save_mean_ = at::empty({n_input}, input_options);
    save_invstd_ = at::empty({n_input}, input_options);
  } else {
    save_mean_ = at::empty({0}, input_options);
    save_invstd_ = at::empty({0}, input_options);
  }
  auto output = output_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, index_t>();
  auto weight = packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(weight_);
  auto bias = packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(bias_);
  auto running_mean = packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<scalar_t, 1, RestrictPtrTraits, index_t>(running_var_);
  auto save_mean = save_mean_.packed_accessor<accscalar_t, 1, RestrictPtrTraits, index_t>();
  auto save_invstd = save_invstd_.packed_accessor<accscalar_t, 1, RestrictPtrTraits, index_t>();
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
  if (!train) {
    batch_norm_transform_input_kernel<scalar_t, accscalar_t, false, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
      (input, output, running_mean, running_var, weight, bias, epsilon);
  } else {
    // for the reduction, we cannot use blocks for the batch dim, but if we have few threads in
    // the feature dimension, we'll use some threads for blocks
    dim3 blocks(input.size(1));
    tf = getNumThreads(input.size(2));
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
    batch_norm_collect_statistics_kernel<scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
      (input, epsilon, momentum, running_mean, running_var, save_mean, save_invstd);
    batch_norm_transform_input_kernel<scalar_t, accscalar_t, true, index_t> <<<blocks_trans, threads_trans, 0, stream>>>
      (input, output, save_mean, save_invstd, weight, bias, epsilon);
  }
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output_reshaped.view(input_.sizes()), save_mean_, save_invstd_);
}

template<typename scalar_t, typename index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template(const Tensor& grad_out_, const Tensor& input_, const Tensor& weight_,
                                                                     const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
                                                                     bool train, double epsilon, std::array<bool,3> grad_input_mask) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_);
  }

  auto input = input_reshaped.packed_accessor<scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_output = grad_output_reshaped.packed_accessor<scalar_t, 3, DefaultPtrTraits, index_t>();
  auto grad_input = packed_accessor_or_dummy<scalar_t, 3, DefaultPtrTraits, index_t>(grad_input_reshaped);
  auto weight = packed_accessor_or_dummy<scalar_t, 1, DefaultPtrTraits, index_t>(weight_);
  auto grad_weight = packed_accessor_or_dummy<scalar_t, 1, DefaultPtrTraits, index_t>(grad_weight_);
  auto grad_bias = packed_accessor_or_dummy<scalar_t, 1, DefaultPtrTraits, index_t>(grad_bias_);
  auto running_mean = packed_accessor_or_dummy<scalar_t, 1, DefaultPtrTraits, index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<scalar_t, 1, DefaultPtrTraits, index_t>(running_var_);
  auto save_mean = packed_accessor_or_dummy<accscalar_t, 1, DefaultPtrTraits, index_t>(save_mean_);
  auto save_invstd = packed_accessor_or_dummy<accscalar_t, 1, DefaultPtrTraits, index_t>(save_invstd_);

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

  batch_norm_backward_kernel<scalar_t,  accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, grad_output, grad_input, grad_weight, grad_bias, weight, running_mean, running_var,
     save_mean, save_invstd, train, epsilon);
  THCudaCheck(cudaGetLastError());

  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

} } // namespace at::native

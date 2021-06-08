#pragma once

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/util/Half.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/core/DistributionsHelper.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <cstdint>
#include <limits>
#include <utility>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace at {
namespace native {
namespace {

// launch bounds used for kernels utilizing TensorIterator
const uint32_t block_size_bound = 256;
const uint32_t grid_size_bound = 4;
// number of randoms given by distributions like curand_uniform4, curand_uniform2_double
// used in calculating philox offset.
const uint32_t curand4_engine_calls = 4;

// utility function that calculates proper philox_offset
// for distributions utilizing TensorIterator. For distributions using
// TensorIterator, we are using a grid-stride loop with each
// thread yielding one element per thread. For the edge of the grid-stride
// loop, if the tensor size is large, the unroll loop will kick in and the float4
// from curand4 will start getting utilized (for common tensor sizes, we end up
// using rand.x from each thread). Hence, the philox_offset is
// (number of elements per thread * number of engine calls), which makes
// sure that philox offset increment is not less than the number of randoms used
// in each thread.
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(int64_t total_elements) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = block_size_bound;
  const uint32_t unroll = curand4_engine_calls;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  uint32_t blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
      grid.x);
  //number of times random will be generated per thread, to offset philox counter in thc random state
  uint64_t counter_offset = ((numel - 1) / (block_size * grid.x * unroll) + 1)
                                * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);
}

// grid stride loop kernel for distributions
template<typename accscalar_t, int unroll_factor, typename dist_t, typename transform_t>
C10_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel(int numel,
                                                            PhiloxCudaState philox_args,
                                                            const dist_t dist_func,
                                                            const transform_t transform_func) {
  auto seeds = at::cuda::philox::unpack(philox_args);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);

  int rounded_size = ((numel - 1)/(blockDim.x * gridDim.x * unroll_factor)+1) *
      blockDim.x * gridDim.x * unroll_factor;
  for(int linear_index = idx; linear_index < rounded_size; linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
      }
    }
    __syncthreads();
  }
}

/**
 * distribution_nullary_kernel is analogous to gpu_kernel in
 * ATen/native/cuda/Loops.cuh. Like gpu_kernel, it uses
 * TensorIterator to launch a kernel. However, the differences are
 *   - it launches a grid-stride loop based kernel. The kernel is not
 *     generic like elementwise_kernel in Loops.cuh and is specialized
 *     for the distribution kernels here.
 *   - For big size tensors, we can launch multiple kernels recursively
 *     (i.e. if (!iter.can_use_32bit_indexing())) and hence, the philox
 *     offset calculation is done in this function.
 *
 * FIXME: Can we specialize elementwise_kernel and launch_kernel in Loops.cuh
 * to have grid-stride loop kernel and then use that to launch our distribution
 * kernels? Note that we need a grid-stride loop kernel because, we found by testing
 * that it achieves peak effective bandwidth.
 */
template<typename scalar_t,
         typename accscalar_t,
         int unroll_factor,
         typename RNG,
         typename dist_t,
         typename transform_t>
void distribution_nullary_kernel(at::TensorIteratorBase& iter,
                                 RNG gen,
                                 const dist_t& dist_func,
                                 const transform_t transform_func) {
  static_assert(unroll_factor >= 1, "unroll_factor must be >= 1.");
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  auto execution_policy = calc_execution_policy(numel);
  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t, unroll_factor>(sub_iter,
        gen, dist_func, transform_func);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);

  auto stream = at::cuda::getCurrentCUDAStream();
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        scalar_t* out = (scalar_t*)&out_data[stride0 * idx];
        *out = transform_func(rand);
      }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        auto offsets = offset_calc.get(idx);
        scalar_t* out = (scalar_t*)&out_data[offsets[0]];
        *out = transform_func(rand);
      }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Binary kernel
template <typename func_t, typename inp_offset_calc_t, typename out_offset_calc_t>
__global__ void distribution_binary_elementwise_kernel(
    int numel,
    func_t f,
    PhiloxCudaState philox_args,
    typename function_traits<func_t>::result_type *output_data,
    const typename function_traits<func_t>::template arg<1>::type *input_data_1,
    const typename function_traits<func_t>::template arg<2>::type *input_data_2,
    inp_offset_calc_t inp_calc,
    out_offset_calc_t out_calc) {
  auto seeds = at::cuda::philox::unpack(philox_args);

  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;

  input_t_1 inputs_1[thread_work_size];
  input_t_2 inputs_2[thread_work_size];

  int base_index = BLOCK_WORK_SIZE * blockIdx.x;
  int remaining = std::min<int>(numel - base_index, BLOCK_WORK_SIZE);

  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              blockIdx.x * blockDim.x + threadIdx.x,
              std::get<1>(seeds),
              &state);

  // load data into registers
  int thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < thread_work_size; i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int input_idx = thread_idx + base_index;
    auto offsets = inp_calc.get(input_idx);
    inputs_1[i] = input_data_1[offsets[0]];
    inputs_2[i] = input_data_2[offsets[1]];

    thread_idx += num_threads;
  }

  // compute and store
  thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < thread_work_size; i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int input_idx = thread_idx + base_index;
    auto offsets = out_calc.get(input_idx);
    output_data[offsets[0]] = f(state, inputs_1[i], inputs_2[i]);
    thread_idx += num_threads;
  }
}

template <typename func_t>
void distribution_binary_kernel(TensorIterator &iter, PhiloxCudaState philox_args, const func_t &f) {
  static_assert(std::is_same<typename function_traits<func_t>::template arg<0>::type, curandStatePhilox4_32_10_t&>::value, "the first argument of functor must be curandStatePhilox4_32_10_t");
  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;
  using output_t = typename function_traits<func_t>::result_type;

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_binary_kernel(sub_iter, philox_args, f);
    }
    return;
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(iter.can_use_32bit_indexing());

  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  output_t *output_data = static_cast<output_t *>(iter.data_ptr(0));
  const input_t_1 *input_data_1 = static_cast<const input_t_1 *>(iter.data_ptr(1));
  const input_t_2 *input_data_2 = static_cast<const input_t_2 *>(iter.data_ptr(2));

  int64_t grid = (numel + block_work_size - 1) / block_work_size;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (iter.is_contiguous()) {
    distribution_binary_elementwise_kernel<<<grid,num_threads, 0, stream>>>(
        numel, f, philox_args, output_data, input_data_1, input_data_2,
        TrivialOffsetCalculator<2>(), TrivialOffsetCalculator<1>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    distribution_binary_elementwise_kernel<<<grid, num_threads, 0, stream>>>(
        numel, f, philox_args, output_data, input_data_1, input_data_2,
        make_input_offset_calculator<2>(iter), make_output_offset_calculator(iter));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace
}} // namespace at::native


namespace at {
namespace native {
namespace templates {
namespace cuda {

// ==================================================== Random ========================================================

template<typename RNG>
void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel_cuda", [&] {
    if ((
      std::is_same<scalar_t, int64_t>::value ||
      std::is_same<scalar_t, double>::value ||
      std::is_same<scalar_t, float>::value ||
      std::is_same<scalar_t, at::BFloat16>::value) && range >= 1ULL << 32)
    {
      // define lambda to mod with range and add base
      auto random_func = [range, base] __device__ (uint64_t rand) {
        return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
      };
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          uint4 rand_val = curand4(state);
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      auto random_func = [range, base] __device__ (uint32_t rand) {
        return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
      };
      distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand4(state);
        },
        random_func);
    }
   });
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIteratorBase& iter, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cuda", [&] {
    if (std::is_same<scalar_t, int64_t>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, at::BFloat16>::value) {
      auto random_func = [] __device__ (uint64_t rand) {
        return transformation::uniform_int_full_range<scalar_t>(rand);
      };
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          uint4 rand_val = curand4(state);
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      TORCH_CHECK(false, "random_full_64_bits_range_kernel_cuda handles only int64, double, float and bfloat16");
    }
  });
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t base, c10::optional<Generator> gen) {
    random_from_to_kernel(iter, range, base, check_generator<RNG>(gen));
  }
  void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
    random_full_64_bits_range_kernel(iter, check_generator<RNG>(gen));
  }
};

template<typename RNG>
void random_kernel(TensorIteratorBase& iter, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "random_kernel_cuda", [&] {
    if (std::is_same<scalar_t, double>::value || std::is_same<scalar_t, int64_t>::value) {
      auto random_func = [] __device__ (uint64_t rand) {
        return transformation::uniform_int<scalar_t>(rand);
      };
      distribution_nullary_kernel<scalar_t, uint64_t, curand4_engine_calls/2>(iter, gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
          ulonglong2 ret;
          uint4 rand_val = curand4(state);
          ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
          ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
          return ret;
        },
        random_func);
    } else {
      auto random_func = [] __device__ (uint32_t rand) {
        return transformation::uniform_int<scalar_t>(rand);
      };
      distribution_nullary_kernel<scalar_t, uint32_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand4(state);
        },
        random_func);
    }
  });
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIteratorBase& iter, RNG gen) {
    random_kernel(iter, gen);
  }
};

// ====================================================================================================================

template<typename scalar_t, typename accscalar_t, size_t curand4_engine_calls, typename RNG, typename transform_t>
void uniform_and_transform(TensorIteratorBase& iter, RNG gen, transform_t transform) {
  if (std::is_same<scalar_t, double>::value) {
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
      transform);
  } else {
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
      transform);
  }
}

template<typename scalar_t, typename accscalar_t, size_t curand4_engine_calls, typename RNG, typename transform_t>
void normal_and_transform(TensorIteratorBase& iter, RNG gen, transform_t transform) {
  if (std::is_same<scalar_t, double>::value) {
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
      transform);
  } else {
    distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
      transform);
  }
}

// ==================================================== Normal ========================================================

template<typename RNG>
void normal_kernel(Tensor& self, double mean_, double std_, RNG gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "normal_kernel_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda to multiply std and add mean
    auto normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::normal<accscalar_t>(rand, mean, std));
    };
    normal_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, normal_func);
   });
}

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
    normal_kernel(self, mean, std, check_generator<RNG>(gen));
  }
};

// ==================================================== Uniform ========================================================

template<typename RNG>
void uniform_kernel(TensorIteratorBase& iter, double from_, double to_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "uniform_kernel_cuda", [&] {
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto range = static_cast<accscalar_t>(to-from);
    // define lambda to reverse bounds, multiply 'range' and add 'from_'
    auto uniform_func = [range, from] __device__ (accscalar_t rand) {
      // reverse the bounds of curand4 from (0, 1] to [0, 1)
      // Note that this method is from legacy THCTensorRandom and is likely to give
      // you more 0-s, since, the probability of gettings 1-s is higher than 0-s and
      // by reversing the bounds, we are flipping the probabilities of 1-s and 0-s.
      // BEFORE TOUCHING THIS CODE READ: https://github.com/pytorch/pytorch/issues/16706
      auto reverse_bound_rand = rand == static_cast<accscalar_t>(1.0) ? static_cast<accscalar_t>(0.0) : rand;
      return static_cast<scalar_t>(reverse_bound_rand * range + from);
    };
    uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, uniform_func);
   });
}

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIteratorBase& iter, double from, double to, c10::optional<Generator> gen) {
    uniform_kernel(iter, from, to, check_generator<RNG>(gen));
  }
};

// ================================================== LogNormal =======================================================

template<typename RNG>
void log_normal_kernel(TensorIteratorBase& iter, double mean_, double std_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda for log_normal transformation
    auto log_normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::log_normal<accscalar_t>(transformation::normal<accscalar_t>(rand, mean, std)));
    };
    normal_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, log_normal_func);
   });
}

template<typename RNG>
struct LogNormalKernel {
  void operator()(TensorIteratorBase& iter, double mean, double std, c10::optional<Generator> gen) {
    log_normal_kernel(iter, mean, std, check_generator<RNG>(gen));
  }
};

// =================================================== Geometric ======================================================

template<typename RNG>
void geometric_kernel(TensorIteratorBase& iter, double p, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "geometric_cuda", [&] {
    using accscalar_t = at::DiscreteDistributionType<scalar_t>::type;
    // define lambda for geometric transformation
    auto geometric_func = [p] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::geometric<accscalar_t>(rand, p));
    };
    uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, geometric_func);
  });
}

template<typename RNG>
struct GeometricKernel {
  void operator()(TensorIteratorBase& iter, double p, c10::optional<Generator> gen) {
    geometric_kernel(iter, p, check_generator<RNG>(gen));
  }
};

// ================================================== Exponential =====================================================

template<typename RNG>
void exponential_kernel(TensorIteratorBase& iter, double lambda_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "exponential_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto lambda = static_cast<accscalar_t>(lambda_);
    // define lambda for exponential transformation
    auto exponential_func = [lambda] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::exponential<accscalar_t>(rand, lambda));
    };
    uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, exponential_func);
   });
}

template<typename RNG>
struct ExponentialKernel {
  void operator()(TensorIteratorBase& iter, double lambda, c10::optional<Generator> gen) {
    exponential_kernel(iter, lambda, check_generator<RNG>(gen));
  }
};

// ==================================================== Cauchy ========================================================

template<typename RNG>
void cauchy_kernel(TensorIteratorBase& iter, double median_, double sigma_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "cauchy_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto median = static_cast<accscalar_t>(median_);
    auto sigma = static_cast<accscalar_t>(sigma_);
    // define lambda for cauchy transformation
    auto cauchy_func = [median, sigma] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::cauchy<accscalar_t>(rand, median, sigma));
    };
    uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, cauchy_func);
   });
}

template<typename RNG>
struct CauchyKernel {
  void operator()(TensorIteratorBase& iter, double median, double sigma, c10::optional<Generator> gen) {
    cauchy_kernel(iter, median, sigma, check_generator<RNG>(gen));
  }
};

// ==================================================== Bernoulli =====================================================

template<typename scalar_t, typename prob_t>
void bernoulli_tensor_cuda_kernel(
    at::Tensor& ret, const at::Tensor& p,
    PhiloxCudaState philox_args) {
  auto functor = [philox_args] __device__(
          int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
          const prob_t& p1, const prob_t& p2, const prob_t& p3, const prob_t& p4) {
        auto seeds = at::cuda::philox::unpack(philox_args);
        curandStatePhilox4_32_10_t state;
        curand_init(std::get<0>(seeds),
                    blockIdx.x * blockDim.x + threadIdx.x,
                    std::get<1>(seeds),
                    &state);

        // See Note [Register spilling in curand call for CUDA < 10]
        float4 rand = curand_uniform4(&state);
        switch (n) {
          case 4: {
            CUDA_KERNEL_ASSERT(0 <= p4 && p4 <= 1);
            v4 = static_cast<scalar_t>(rand.w <= p4);
            // fallthrough
          }
          case 3: {
            CUDA_KERNEL_ASSERT(0 <= p3 && p3 <= 1);
            v3 = static_cast<scalar_t>(rand.z <= p3);
            // fallthrough
          }
          case 2: {
            CUDA_KERNEL_ASSERT(0 <= p2 && p2 <= 1);
            v2 = static_cast<scalar_t>(rand.y <= p2);
            // fallthrough
          }
          case 1: {
            CUDA_KERNEL_ASSERT(0 <= p1 && p1 <= 1);
            v1 = static_cast<scalar_t>(rand.x <= p1);
          }
        }
      };
  // The template argument `4` below indicates that we want to operate on four
  // element at each time. See NOTE [ CUDA_tensor_applyN helpers ] for details.
  at::cuda::CUDA_tensor_apply2<scalar_t, prob_t, 4, decltype(functor),
                               /*max_threads_per_block=*/512,
                               /*min_blocks_per_sm==*/2>(ret, p, functor);
}

template<typename RNG>
void bernoulli_kernel(Tensor& self, const Tensor& p_, RNG gen) {
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(10);
  }
  auto p_CUDA = p_.to(kCUDA);
  c10::MaybeOwned<Tensor> p = expand_inplace(self, p_CUDA);
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
      using self_t = scalar_t;
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, p->scalar_type(), "bernoulli_tensor_cuda_p_", [&] {
        using p_t = scalar_t;
        return bernoulli_tensor_cuda_kernel<self_t, p_t>(self, *p, rng_engine_inputs);
      });
   });
}

template<typename RNG>
void bernoulli_kernel(TensorIteratorBase& iter, double p, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "bernoulli_scalar_cuda_", [&] {
      using accscalar_t = at::DiscreteDistributionType<scalar_t>::type;
      // define lambda for bernoulli transformation
      auto bernoulli_func = [p] __device__ (accscalar_t rand) {
        return static_cast<scalar_t>(transformation::bernoulli<accscalar_t>(rand, p));
      };
      uniform_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, bernoulli_func);
   });
}

template<typename RNG>
struct BernoulliKernel {
  void operator()(TensorIteratorBase& iter, double p, c10::optional<Generator> gen) {
    bernoulli_kernel(iter, p, check_generator<RNG>(gen));
  }
  void operator()(Tensor& self, const Tensor& p_, c10::optional<Generator> gen) {
    bernoulli_kernel(self, p_, check_generator<RNG>(gen));
  }
};

}}}}

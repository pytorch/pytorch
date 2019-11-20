#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGenerator.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

#include <THC/THCGeneral.h>
#include <THC/THCApply.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

/**
 * Note [Register spilling in curand call for CUDA < 10]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * For CUDA < 10, curandStatePhilox4_32_10_t engine achieves poor performance (60% SOL bandwidth)
 * when called to generate one random number at a time. This is because the line
 *            unsigned ret = (&state->output.x)[state->STATE++];
 * in
 *            QUALIFIERS unsigned int curand(curandStatePhilox4_32_10_t *state)
 * in curand_kernel.h dynamically indexes into state.output, preventing the compiler from ever
 * storing state.output in registers.
 *
 * CUDA 10 fixed this problem. However, for backwards compatibility, in the following kernels
 * we are using curand distributions that utilize curand4 call. curand4 call doesn't have the
 * register spilling problem.
 */

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
                                                            std::pair<uint64_t, uint64_t> seeds,
                                                            const dist_t dist_func,
                                                            const transform_t transform_func) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(
      seeds.first,
      idx,
      seeds.second,
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
         typename dist_t,
         typename transform_t>
void distribution_nullary_kernel(at::TensorIterator& iter,
                                 at::CUDAGenerator* gen,
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
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
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
  } else {
    auto offset_calc = at::native::make_offset_calculator<1>(iter);
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
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void poisson_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& lambda,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
      ret,
      lambda,
      [seeds] __device__(
          scalar_t & ret_val, const scalar_t& lambda) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        ret_val = static_cast<scalar_t>(curand_poisson(&state, lambda));
      });
}

template <typename scalar_t>
void gamma_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& alpha,
    std::pair<uint64_t, uint64_t> seeds) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
      ret,
      alpha,
      [seeds] __device__(
          scalar_t & ret_val, const scalar_t& alpha) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);

        auto uniform_lambda = [&state] __device__ () {
          return curand_uniform(&state);
        };
        BaseSampler<accscalar_t, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

        auto normal_lambda = [&state] __device__ () {
          return curand_normal(&state);
        };
        BaseSampler<accscalar_t, decltype(normal_lambda)> standard_normal(normal_lambda);
        auto sample = sample_gamma<scalar_t, accscalar_t, decltype(uniform_lambda), decltype(normal_lambda)>(alpha, standard_uniform, standard_normal);
        auto min_value = std::numeric_limits<scalar_t>::min();
        ret_val = (min_value > sample) ? min_value : sample;
      });
}

template <typename scalar_t>
void gamma_grad_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& output) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      ret, self, output,
      [] __device__ (scalar_t& ret_val, const scalar_t& self_val, const scalar_t &output_val) {
        ret_val = standard_gamma_grad_one<scalar_t, accscalar_t>(self_val, output_val);
      });
}

template <typename scalar_t>
void dirichlet_grad_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& total) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  at::cuda::CUDA_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
      ret, x, alpha, total,
      [] __device__ (scalar_t& ret_val, const scalar_t& x_val, const scalar_t& alpha_val, const scalar_t& total_val) {
        ret_val = dirichlet_grad_one<scalar_t, accscalar_t>(x_val, alpha_val, total_val);
      });
}

template<typename scalar_t, typename prob_t>
void bernoulli_tensor_cuda_kernel(
    at::Tensor& ret, const at::Tensor& p,
    std::pair<uint64_t, uint64_t> seeds) {
  // The template argument `4` below indicates that we want to operate on four
  // element at each time. See NOTE [ CUDA_tensor_applyN helpers ] for details.
  at::cuda::CUDA_tensor_apply2<scalar_t, prob_t, 4>(
      ret, p,
      [seeds] __device__(
          int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
          const prob_t& p1, const prob_t& p2, const prob_t& p3, const prob_t& p4) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        // See Note [Register spilling in curand call for CUDA < 10]
        float4 rand = curand_uniform4(&state);
        switch (n) {
          case 4: {
            assert(0 <= p4 && p4 <= 1);
            v4 = static_cast<scalar_t>(rand.w <= p4);
            // fallthrough
          }
          case 3: {
            assert(0 <= p3 && p3 <= 1);
            v3 = static_cast<scalar_t>(rand.z <= p3);
            // fallthrough
          }
          case 2: {
            assert(0 <= p2 && p2 <= 1);
            v2 = static_cast<scalar_t>(rand.y <= p2);
            // fallthrough
          }
          case 1: {
            assert(0 <= p1 && p1 <= 1);
            v1 = static_cast<scalar_t>(rand.x <= p1);
          }
        }
      }
    );
}

template<typename scalar_t>
void dirichlet_scalar_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& gamma) {
  auto gamma_sum = gamma.sum(-1, true).expand(ret.sizes());
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(ret, gamma, gamma_sum,
  [] __device__(scalar_t &ret_val, const scalar_t &gamma, const scalar_t &gamma_sum) {
    ret_val = gamma / gamma_sum;
    auto min_value = std::numeric_limits<scalar_t>::min();
    auto max_value = 1 - std::numeric_limits<scalar_t>::epsilon();
    ret_val = (min_value > ret_val) ? min_value : ret_val;
    ret_val = (max_value < ret_val) ? max_value : ret_val;
  });
}

} // namespace

namespace at { namespace native {

Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(20);
  }
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.scalar_type(), "poisson_cuda", [&] {
    poisson_cuda_kernel<scalar_t>(ret, lambda, rng_engine_inputs);
  });
  return ret;
}

Tensor _s_gamma_cuda(const Tensor& alpha, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(10);
  }
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.scalar_type(), "gamma_cuda", [&] {
     gamma_cuda_kernel<scalar_t>(ret, alpha, rng_engine_inputs);
   });
  return ret;
}

Tensor _s_dirichlet_cuda(const Tensor& alpha, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(10);
  }
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.scalar_type(), "dirichlet", [&] {
    Tensor gamma = at::empty(alpha.sizes(), alpha.options());
    gamma_cuda_kernel<scalar_t>(gamma, alpha, rng_engine_inputs);
    dirichlet_scalar_cuda_kernel<scalar_t>(ret, gamma);
  });
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "_standard_gamma_grad_cuda", [&] {
     gamma_grad_cuda_kernel<scalar_t>(ret, self, output);
   });
  return ret;
}

Tensor _dirichlet_grad_cuda(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "_dirichlet_grad_cuda", [&] {
    dirichlet_grad_cuda_kernel<scalar_t>(ret, x, alpha, total);
  });
  return ret;
}

Tensor& bernoulli_tensor_cuda_(Tensor &self, const Tensor& p_, Generator* gen_) {
#ifdef BUILD_NAMEDTENSOR
  NoNamesGuard guard;
#endif
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(10);
  }
  auto p = std::get<0>(expand_inplace(self, p_.to(kCUDA)));
  AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
      using self_t = scalar_t;
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "bernoulli_tensor_cuda_p_", [&] {
        using p_t = scalar_t;
        return bernoulli_tensor_cuda_kernel<self_t, p_t>(self, p, rng_engine_inputs);
      });
   });
  return self;
}

void uniform_kernel_cuda(TensorIterator& iter, double from_, double to_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "uniform_cuda", [&] {
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    TORCH_CHECK(from <= to,
      "uniform_ expects to return a [from, to) range, but found from=", from,
      " > to=", to);
    TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
          "uniform_ expects to-from <= std::numeric_limits<", toString(iter.dtype()),
          ">::max(), but found to=", to, " and from=", from,
          " which result in to-from to exceed the limit");

    using accscalar_t = at::acc_type<scalar_t, true>;
    auto range = static_cast<accscalar_t>(to-from);
    from = static_cast<accscalar_t>(from);
    // define lambda to reverse bounds, multiply 'range' and add 'from_'
    auto uniform_func = [range, from] __device__ (accscalar_t rand) {
      // reverse the bounds of curand4 from (0, 1] to [0, 1)
      // Note that this method is from legacy THCTensorRandom and is likely to give
      // you more 0-s, since, the probability of gettings 1-s is higher than 0-s and
      // by reversing the bounds, we are flipping the probabilities of 1-s and 0-s.
      auto reverse_bound_rand = rand == static_cast<accscalar_t>(1.0) ? static_cast<accscalar_t>(0.0) : rand;
      return static_cast<scalar_t>(reverse_bound_rand * range + from);
    };
    if (std::is_same<scalar_t, double>::value) {
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
        uniform_func);
    } else {
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        uniform_func);
    }
   });
}

void random_kernel_cuda(TensorIterator& iter, uint64_t range, int64_t base, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::Half, iter.dtype(), "random_cuda", [&] {
    if (std::is_same<scalar_t, double>::value || std::is_same<scalar_t, int64_t>::value) {
      // define lambda to mod with range and add base
      auto random_func = [range, base] __device__ (uint64_t rand) {
        return static_cast<int64_t>(rand % range + base);
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
        return static_cast<int32_t>(rand % static_cast<uint32_t>(range) + static_cast<int32_t>(base));
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

void normal_kernel_cuda(TensorIterator& iter, double mean_, double std_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "normal_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda to multiply std and add mean
    auto normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(rand * std + mean);
    };
    if (std::is_same<scalar_t, double>::value) {
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
        normal_func);
    } else {
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
        normal_func);
    }
   });
}

void cauchy_kernel_cuda(TensorIterator& iter, double median_, double sigma_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "cauchy_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto median = static_cast<accscalar_t>(median_);
    auto sigma = static_cast<accscalar_t>(sigma_);
    if (std::is_same<scalar_t, double>::value) {
      // define lambda for cauchy transformation
      auto cauchy_func = [median, sigma] __device__ (accscalar_t rand) {
        return static_cast<scalar_t>(median + sigma *
                ::tan(static_cast<accscalar_t>(M_PI) * (rand-static_cast<accscalar_t>(0.5))));
      };
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
        cauchy_func);
    } else {
      // use __tanf fast approximation for peak bandwidth
      auto cauchy_func = [median, sigma] __device__ (accscalar_t rand) {
        return static_cast<scalar_t>(median + sigma *
                __tanf(static_cast<accscalar_t>(M_PI) * (rand-static_cast<accscalar_t>(0.5))));
      };
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        cauchy_func);
    }
   });
}

void exponential_kernel_cuda(TensorIterator& iter, double lambda_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  // Note that HIP doesn't support std::nextafter in device code.
  auto nextafter_1_0_float = std::nextafter(1.0f, 0.0f);
  auto nextafter_1_0_double = std::nextafter(1.0, 0.0);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "exponential_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto lambda = static_cast<accscalar_t>(lambda_);
    if (std::is_same<scalar_t, double>::value) {
      // define lambda for exponential transformation
      auto exponential_func = [lambda, nextafter_1_0_double] __device__ (accscalar_t rand) {
        accscalar_t sample;
        // curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
        // Hence, squash the 1 to just below 1.
        if(rand == static_cast<accscalar_t>(1.0)) {
          sample = ::log(nextafter_1_0_double);
        } else {
          sample = ::log(rand);
        }
        return static_cast<scalar_t>(static_cast<accscalar_t>(-1.0) / lambda * sample);
      };
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
        exponential_func);
    } else {
      // use __logf fast approximation for peak bandwidth
      auto exponential_func = [lambda, nextafter_1_0_float] __device__ (accscalar_t rand) {
        accscalar_t sample;
        if(rand == static_cast<accscalar_t>(1.0)) {
          sample = __logf(nextafter_1_0_float);
        } else {
          sample = __logf(rand);
        }
        return static_cast<scalar_t>(static_cast<accscalar_t>(-1.0) / lambda * sample);
      };
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        exponential_func);
    }
   });
}

void geometric_kernel_cuda(TensorIterator& iter, double p_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, iter.dtype(), "geometric_cuda", [&] {
    if (std::is_same<scalar_t, double>::value) {
      // define lambda for geometric transformation
      auto geometric_func = [p_] __device__ (double rand) {
        return static_cast<scalar_t>(::ceil(::log(rand) / ::log(static_cast<double>(1.0)-p_)));
      };
      distribution_nullary_kernel<scalar_t, double, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
        geometric_func);
    } else {
      auto p = static_cast<float>(p_);
      auto geometric_func = [p] __device__ (float rand) {
        // use __logf fast approximation for peak bandwidth
        return static_cast<scalar_t>(::ceil(__logf(rand) / __logf(static_cast<float>(1.0)-p)));
      };
      distribution_nullary_kernel<scalar_t, float, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        geometric_func);
    }
   });
}

void log_normal_kernel_cuda(TensorIterator& iter, double mean_, double std_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "log_normal_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    if (std::is_same<scalar_t, double>::value) {
      // define lambda for log_normal transformation
      auto log_normal_func = [mean, std] __device__ (accscalar_t rand) {
        return static_cast<scalar_t>(::exp(rand * std + mean));
      };
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
        log_normal_func);
    } else {
      auto log_normal_func = [mean, std] __device__ (accscalar_t rand) {
        // use __expf fast approximation for peak bandwidth
        return static_cast<scalar_t>(__expf(rand * std + mean));
      };
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
        log_normal_func);
    }
   });
}

void bernoulli_scalar_cuda_kernel(TensorIterator& iter, double p_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::Bool, iter.dtype(), "bernoulli_scalar_cuda_", [&] {
      if (std::is_same<scalar_t, double>::value) {
      // define lambda for bernoulli transformation
      auto bernoulli_func = [p_] __device__ (double rand) {
        return static_cast<scalar_t>(rand <= p_);
      };
      distribution_nullary_kernel<scalar_t, double, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
        bernoulli_func);
    } else {
      auto p = static_cast<float>(p_);
      auto bernoulli_func = [p] __device__ (float rand) {
        return static_cast<scalar_t>(rand <= p);
      };
      distribution_nullary_kernel<scalar_t, float, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        bernoulli_func);
    }
   });
}

Tensor& uniform_cuda_(Tensor& self, double from, double to, Generator* gen) {
  auto iter = TensorIterator::nullary_op(self);
  uniform_kernel_cuda(iter, from, to, gen);
  return self;
}

Tensor& random_cuda_(Tensor& self, Generator* gen) {
  auto iter = TensorIterator::nullary_op(self);
  uint64_t range;
  auto iter_scalar_type = iter.dtype();
  if (isFloatingType(iter_scalar_type)) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter_scalar_type, "random_cuda_range_calc", [&] {
      range = static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter_scalar_type, "random_cuda_range_calc", [&] {
      range = static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1;
    });
  }
  random_kernel_cuda(iter, range, 0, gen);
  return self;
}

Tensor& clamped_random_cuda_(Tensor& self, int64_t from, int64_t to, Generator* gen) {
  TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
  auto iter = TensorIterator::nullary_op(self);
  uint64_t range = to - from;
  random_kernel_cuda(iter, range, from, gen);
  return self;
}

Tensor& capped_random_cuda_(Tensor& self, int64_t to, Generator* gen) {
  return clamped_random_cuda_(self, 0, to, gen);
}

Tensor& normal_cuda_(Tensor& self, double mean, double std, Generator* gen) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);
  auto iter = TensorIterator::nullary_op(self);
  normal_kernel_cuda(iter, mean, std, gen);
  return self;
}

Tensor& normal_out_cuda(Tensor& output, const Tensor& mean, double std, Generator* gen) {
  normal_cuda_(output, 0, std, gen);
  output.add_(mean);
  return output;
}

Tensor& normal_out_cuda(Tensor& output, double mean, const Tensor& std, Generator* gen) {
  normal_cuda_(output, 0, 1, gen);
  auto mean_tensor = at::full({1}, mean, output.options());
  // NB: addcmul_out copies the tensor to be added into the output.
  // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
  // The previous function here was addcmul_out(output, mean_tensor, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean_tensor + mean_tensor * std instead of mean_tensor + output * std
  output.mul_(std).add_(mean_tensor);
  return output;
}

Tensor& normal_out_cuda(Tensor& output, const Tensor& mean, const Tensor& std, Generator* gen) {
  normal_cuda_(output, 0, 1, gen);
  // NB: addcmul_out copies the tensor to be added into the output.
  // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
  // The previous function here was addcmul_out(output, mean, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean + mean * std instead of mean + output * std
  output.mul_(std).add_(mean);
  return output;
}

Tensor normal_cuda(const Tensor& mean, double std, Generator* gen) {
  Tensor ret = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  normal_out_cuda(ret, mean, std, gen);
  return ret;
}

Tensor normal_cuda(double mean, const Tensor& std, Generator* gen) {
  Tensor ret = at::empty_like(std, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  normal_out_cuda(ret, mean, std, gen);
  return ret;
}

Tensor normal_cuda(const Tensor& mean, const Tensor& std, Generator* gen) {
  Tensor ret = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  normal_out_cuda(ret, mean, std, gen);
  return ret;
}

Tensor& cauchy_cuda_(Tensor& self, double median, double sigma, Generator* gen) {
  auto iter = TensorIterator::nullary_op(self);
  cauchy_kernel_cuda(iter, median, sigma, gen);
  return self;
}

Tensor& exponential_cuda_(Tensor& self, double lambda, Generator* gen) {
  auto iter = TensorIterator::nullary_op(self);
  exponential_kernel_cuda(iter, lambda, gen);
  return self;
}

Tensor& geometric_cuda_(Tensor& self, double p, Generator* gen) {
  TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
  auto iter = TensorIterator::nullary_op(self);
  geometric_kernel_cuda(iter, p, gen);
  return self;
}

Tensor& log_normal_cuda_(Tensor& self, double mean, double std, Generator* gen) {
  TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
  auto iter = TensorIterator::nullary_op(self);
  log_normal_kernel_cuda(iter, mean, std, gen);
  return self;
}

Tensor& bernoulli_scalar_cuda_(Tensor &self, double p, Generator* gen) {
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  auto iter = TensorIterator::nullary_op(self);
  bernoulli_scalar_cuda_kernel(iter, p, gen);
  return self;
}

}} // namespace at::native

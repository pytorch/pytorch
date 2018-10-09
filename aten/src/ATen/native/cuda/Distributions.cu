#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/AccumulateType.h"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include "ATen/cuda/PhiloxRNGEngine.h"
#include <utility>
#include <functional>
#include <nvfunctional>

#include "ATen/native/Distributions.h"

#include <THC/THCGeneral.h>
#include <THC/THCApply.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

namespace {

template <typename scalar_t>
void poisson_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& lambda,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
      ret,
      lambda,
      [seeds] __device__(scalar_t & ret_val, const scalar_t& lambda) {
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
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t, 2>(
      ret,
      alpha,
      [seeds] __device__(
        int n, scalar_t& ret_val1, scalar_t& ret_val2, 
               const scalar_t& alpha1, const scalar_t& alpha2) {
        at::cuda::Philox4_32_10 engine_x(
                                  seeds.first,
                                  blockIdx.x * blockDim.x + threadIdx.x,
                                  seeds.second);
        // creating a copy of engine_x such that float2 of normal dist can be utilized
        at::cuda::Philox4_32_10 engine_y = engine_x;

        BaseSampler<accscalar_t> standard_uniform_x([&engine_x] __device__ () {
          return at::cuda::standard_uniform_distribution(engine_x);
        });
        BaseSampler<accscalar_t> standard_uniform_y([&engine_y] __device__ () {
          return at::cuda::standard_uniform_distribution(engine_y);
        });

        BaseSampler<accscalar_t> standard_normal_x([&engine_x] __device__ () {
          return at::cuda::normal_distribution(engine_x).x;
        });
        BaseSampler<accscalar_t> standard_normal_y([&engine_y] __device__ () {
          return at::cuda::normal_distribution(engine_y).y;
        });

        switch (n) {
          case 2: {
            auto sample_y = sample_gamma<scalar_t, accscalar_t>(alpha2, standard_uniform_y, standard_normal_y);
            auto min_value_y = std::numeric_limits<scalar_t>::lowest();
            ret_val2 = (min_value_y > sample_y) ? min_value_y : sample_y;
          }
          case 1: {
            auto sample_x = sample_gamma<scalar_t, accscalar_t>(alpha1, standard_uniform_x, standard_normal_x);
            auto min_value_x = std::numeric_limits<scalar_t>::lowest();
            ret_val1 = (min_value_x > sample_x) ? min_value_x : sample_x;
          }
        }
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

template<typename scalar_t, typename prob_t>
void bernoulli_tensor_cuda_kernel(
    at::Tensor& ret, const at::Tensor& p,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, prob_t>(
      ret, p,
      [seeds] __device__(scalar_t& v1, const prob_t& p1) {
      at::cuda::Philox4_32_10 engine(
                                seeds.first,
                                blockIdx.x * blockDim.x + threadIdx.x,
                                seeds.second);
      auto rand_num = at::cuda::standard_uniform_distribution(engine);
      assert(0 <= p1 && p1 <= 1);
      v1 = static_cast<scalar_t>(rand_num <= p1);
    }
  );
}

template<typename scalar_t>
void bernoulli_scalar_cuda_kernel(
    at::Tensor& ret, double p_,
    std::pair<uint64_t, uint64_t> seeds) {
  float p = static_cast<float>(p_);
  at::cuda::CUDA_tensor_apply1<scalar_t>(
      ret, 
      [seeds, p] __device__(scalar_t& v1) {
      at::cuda::Philox4_32_10 engine(
                                seeds.first,
                                blockIdx.x * blockDim.x + threadIdx.x,
                                seeds.second);
      auto rand_num = at::cuda::standard_uniform_distribution(engine);
      v1 = static_cast<scalar_t>(rand_num <= p);
    }
  );
}

} // namespace

namespace at { namespace native {
Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  auto gen_ = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCUDA));
  uint64_t step = 1;
  uint64_t total_elements = ret.numel();
  // grid calculation from getApplyGrid() in CUDAApplyUtils.cuh
  uint64_t grid_size = (total_elements + (AT_APPLY_THREADS_PER_BLOCK * step) - 1) / (AT_APPLY_THREADS_PER_BLOCK * step);
  #if CUDA_VERSION < 9000
    if (!ret.is_contiguous()) {
      uint64_t blocks_per_sm = AT_APPLY_BLOCKS_PER_SM;
      grid_size = std::min((unsigned int)(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm , grid_size);
    }
  #endif
  // the philox offset calculation here is not exact and the multiplier 30 is just
  // a guess. This is because curand_poisson is using algorithms which use while
  // loops and hence, we cannot predict the number of engine calls that is made
  // by the philox engine.
  auto seeds = gen_->incrementPhiloxOffset(total_elements, grid_size, AT_APPLY_THREADS_PER_BLOCK, 30);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "poisson", [&] {
    poisson_cuda_kernel<scalar_t>(ret, lambda, seeds);
  });
  return ret;
}

Tensor _s_gamma_cuda(const Tensor& alpha, Generator* gen) {
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  auto gen_ = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCUDA));
  uint64_t step = 2;
  uint64_t total_elements = ret.numel();
  // grid calculation from getApplyGrid() in CUDAApplyUtils.cuh
  uint64_t grid_size = (total_elements + (AT_APPLY_THREADS_PER_BLOCK * step) - 1) / (AT_APPLY_THREADS_PER_BLOCK * step);
  #if CUDA_VERSION < 9000
    if (!ret.is_contiguous()) {
      uint64_t blocks_per_sm = AT_APPLY_BLOCKS_PER_SM;
      grid_size = std::min((unsigned int)(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm , grid_size);
    }
  #endif
  // the philox offset calculation here is not exact and the multiplier 40 is just
  // a guess. This is because sample_gamma() (the Marsaglia-Tsang sampler) has an
  // accept-reject cycle which terminates with a probility > 90%. Hence, we cannot 
  // predict the number of engine calls that is made by the philox engine.
  auto seeds = gen_->incrementPhiloxOffset(total_elements, grid_size, AT_APPLY_THREADS_PER_BLOCK, 40);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "gamma", [&] {
    gamma_cuda_kernel<scalar_t>(ret, alpha, seeds);
   });
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "_standard_gamma_grad", [&] {
     gamma_grad_cuda_kernel<scalar_t>(ret, self, output);
   });
  return ret;
}

Tensor& bernoulli_tensor_cuda_(Tensor &self, const Tensor& p_, Generator* gen) {
  auto p = std::get<0>(expand_inplace(self, p_.to(kCUDA)));
  auto gen_ = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCUDA));
  uint64_t step = 1;
  uint64_t total_elements = self.numel();
  // grid calculation from getApplyGrid() in CUDAApplyUtils.cuh
  uint64_t grid_size = (total_elements + (AT_APPLY_THREADS_PER_BLOCK * step) - 1) / (AT_APPLY_THREADS_PER_BLOCK * step);
  #if CUDA_VERSION < 9000
    if (!self.is_contiguous()) {
      uint64_t blocks_per_sm = AT_APPLY_BLOCKS_PER_SM;
      grid_size = std::min((unsigned int)(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm , grid_size);
    }
  #endif
  // number of engine() calls is 1 for uniform
  // no loop unrolling, hence, num_engine_calls = 1
  auto seeds = gen_->incrementPhiloxOffset(total_elements, grid_size, AT_APPLY_THREADS_PER_BLOCK, 1);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "bernoulli_tensor_cuda_self_", [&] {
    const at::Type& p_type = p.type();
    using self_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.type(), "bernoulli_tensor_cuda_p_", [&] {
      using p_t = scalar_t;
      return bernoulli_tensor_cuda_kernel<self_t, p_t>(self, p, seeds);
    });
   });
  return self;
}

Tensor& bernoulli_scalar_cuda_(Tensor &self, double p, Generator* gen) {
  AT_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  auto gen_ = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCUDA));
  uint64_t step = 1;
  uint64_t total_elements = self.numel();
  // grid calculation from getApplyGrid() in CUDAApplyUtils.cuh
  uint64_t grid_size = (total_elements + (AT_APPLY_THREADS_PER_BLOCK * step) - 1) / (AT_APPLY_THREADS_PER_BLOCK * step);
  #if CUDA_VERSION < 9000
    if (!self.is_contiguous()) {
      uint64_t blocks_per_sm = AT_APPLY_BLOCKS_PER_SM;
      grid_size = std::min((unsigned int)(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm , grid_size);
    }   
  #endif
  // number of engine() calls is 1 for uniform
  // no loop unrolling, hence, num_engine_calls = 1
  auto seeds = gen_->incrementPhiloxOffset(total_elements, grid_size, AT_APPLY_THREADS_PER_BLOCK, 1);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "bernoulli_scalar_cuda_", [&] {
    bernoulli_scalar_cuda_kernel<scalar_t>(self, p, seeds);
   });
  return self;
}


}} // namespace at::native

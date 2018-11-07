#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/AccumulateType.h"

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>
#include <nvfunctional>

#include "ATen/native/Distributions.h"

#include <THC/THCGeneral.h>
#include <THC/THCTensorRandom.h>
#include <THC/THCGenerator.hpp>
#include <THC/THCApply.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace {
// increment should be at least the number of curand() random numbers used in
// each thread.
std::pair<uint64_t, uint64_t> next_philox_seed(at::Generator* gen, uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
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
        BaseSampler<accscalar_t> standard_uniform([&state] __device__ () {
          return curand_uniform(&state);
        });
        BaseSampler<accscalar_t> standard_normal([&state] __device__ () {
          return curand_normal(&state);
        });
        auto sample = sample_gamma<scalar_t, accscalar_t>(alpha, standard_uniform, standard_normal);
        auto min_value = std::numeric_limits<scalar_t>::lowest();
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
void bernoulli_scalar_cuda_kernel(
    at::Tensor& ret, double p_,
    std::pair<uint64_t, uint64_t> seeds) {
  float p = static_cast<float>(p_);
  // The template argument `4` below indicates that we want to operate on four
  // element at each time. See NOTE [ CUDA_tensor_applyN helpers ] for details.
  at::cuda::CUDA_tensor_apply1<scalar_t, 4>(
      ret, [seeds, p] __device__(
        int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        float4 rand = curand_uniform4(&state);
        switch (n) {
          case 4: {
            v4 = static_cast<scalar_t>(rand.w <= p);
            // fallthrough
          }
          case 3: {
            v3 = static_cast<scalar_t>(rand.z <= p);
            // fallthrough
          }
          case 2: {
            v2 = static_cast<scalar_t>(rand.y <= p);
            // fallthrough
          }
          case 1: {
            v1 = static_cast<scalar_t>(rand.x <= p);
          }
        }
      }
    );
}

} // namespace

namespace at { namespace native {
Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "poisson", [&] {
    poisson_cuda_kernel<scalar_t>(ret, lambda, next_philox_seed(gen, 20));
  });
  return ret;
}

Tensor _s_gamma_cuda(const Tensor& alpha, Generator* gen) {
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "gamma", [&] {
     gamma_cuda_kernel<scalar_t>(ret, alpha, next_philox_seed(gen, 10));
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
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "bernoulli_tensor_cuda_self_", [&] {
    const at::Type& p_type = p.type();
    using self_t = scalar_t;
    auto seeds = next_philox_seed(gen, 10);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.type(), "bernoulli_tensor_cuda_p_", [&] {
      using p_t = scalar_t;
      return bernoulli_tensor_cuda_kernel<self_t, p_t>(self, p, seeds);
    });
   });
  return self;
}

Tensor& bernoulli_scalar_cuda_(Tensor &self, double p, Generator* gen) {
  AT_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "bernoulli_scalar_cuda_", [&] {
    auto seeds = next_philox_seed(gen, 10);
    bernoulli_scalar_cuda_kernel<scalar_t>(self, p, seeds);
   });
  return self;
}


}} // namespace at::native

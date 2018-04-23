#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>
#include <nvfunctional>

#include "ATen/native/Distributions.h"

#include <TH/THAtomic.h>

#include <THC/THCGeneral.h>
#include <THC/THCTensorRandom.h>
#include <THC/THCGenerator.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>

#include <cstdint>
#include <utility>

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace {
std::pair<uint64_t, uint64_t> next_philox_seed(at::Generator* gen) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().thc_state);
  uint64_t offset = THAtomicAddLong(&gen_->state.philox_seed_offset, 1);
  return std::make_pair(gen_->state.initial_seed, offset);
}

template <typename scalar_t>
void poisson_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& lambda,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, float>(
      ret,
      lambda,
      [seeds] __device__(
          scalar_t & ret_val, const float& lambda, bool early_exit) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        ret_val = scalar_cast<scalar_t>(curand_poisson(&state, lambda));
      });
}

template <typename scalar_t>
void gamma_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& alpha,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, float>(
      ret,
      alpha,
      [seeds] __device__(
          scalar_t & ret_val, const float& alpha, bool early_exit) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
	BaseSampler<float> standard_uniform([&state] __device__ () {
	  return curand_uniform(&state);
        });
	BaseSampler<float> standard_normal([&state] __device__ () {
          return curand_normal(&state);
        });
	auto sample = sample_gamma<float>(alpha, standard_uniform, standard_normal);
	ret_val = ::max(THCNumerics<scalar_t>::min(), scalar_cast<scalar_t>(sample));
      });
}

} // namespace

namespace at { namespace native {
Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = lambda.type().tensor(lambda.sizes());
  auto lambda_ = lambda.toType(ScalarType::Float);
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "poisson", [&] {
     poisson_cuda_kernel<scalar_t>(ret, lambda_, next_philox_seed(gen));
   });
  return ret;
}

Tensor _s_gamma_cuda(const Tensor& alpha, Generator* gen) {
  Tensor ret = alpha.type().tensor(alpha.sizes());
  auto alpha_ = alpha.toType(ScalarType::Float);
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "poisson", [&] {
     gamma_cuda_kernel<scalar_t>(ret, alpha_, next_philox_seed(gen));
   });
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = self.type().tensor(self.sizes());
  AT_DISPATCH_FLOATING_TYPES(self.type(), "_standard_gamma_grad", [&] {
      at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(ret, self, output,
      [] __device__ (scalar_t& ret_val, const scalar_t& self_val, const scalar_t &output_val) {
         ret_val = standard_gamma_grad_one(self_val, output_val);
      }
    );
  });
  return ret;
}

}} // namespace at::native

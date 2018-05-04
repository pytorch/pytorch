#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDATypeConversion.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include <THC/THCNumerics.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

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
#include <THC/THCNumerics.cuh>

#include <cstdint>
#include <utility>

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace {
std::pair<uint64_t, uint64_t> next_philox_seed(at::Generator* gen, uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().thc_state);
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
          scalar_t & ret_val, const scalar_t& lambda, bool early_exit) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        ret_val = scalar_cast<scalar_t>(curand_poisson(&state, scalar_cast<float>(lambda)));
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
          scalar_t & ret_val, const scalar_t& alpha, bool early_exit) {
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
        ret_val = ((THCNumerics<scalar_t>::min() > sample) ? (THCNumerics<scalar_t>::min()) : sample);
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

} // namespace

namespace at { namespace native {
Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = lambda.type().tensor(lambda.sizes());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "poisson", [&] {
    poisson_cuda_kernel<cuda::type<scalar_t>>(ret, lambda, next_philox_seed(gen, 20));
  });
  return ret;
}

Tensor _s_gamma_cuda(const Tensor& alpha, Generator* gen) {
  Tensor ret = alpha.type().tensor(alpha.sizes());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "gamma", [&] {
     gamma_cuda_kernel<cuda::type<scalar_t>>(ret, alpha, next_philox_seed(gen, 10));
   });
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = self.type().tensor(self.sizes());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "_standard_gamma_grad", [&] {
     gamma_grad_cuda_kernel<cuda::type<scalar_t>>(ret, self, output);
   });
  return ret;
}

}} // namespace at::native

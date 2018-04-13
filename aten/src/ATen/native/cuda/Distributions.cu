#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

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

}} // namespace at::native

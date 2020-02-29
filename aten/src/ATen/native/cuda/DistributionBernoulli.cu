#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGenerator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

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

namespace {

template<typename scalar_t, typename prob_t>
void bernoulli_tensor_cuda_kernel(
    at::Tensor& ret, const at::Tensor& p,
    std::pair<uint64_t, uint64_t> seeds) {
  at::TensorIterator iter;
  iter.dont_compute_common_dtype();
  iter.add_output(ret);
  iter.add_input(p);
  iter.build();

  at::native::gpu_kernel(iter,
    [seeds] GPU_LAMBDA (prob_t p) -> scalar_t {
      #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      CUDA_KERNEL_ASSERT(0 <= p && p <= 1);
      float rand = curand_uniform(&state);
      return static_cast<scalar_t>(rand <= p);
      #else
      return scalar_t(0);  // useless
      #endif
    });
}

} // namespace

namespace at { namespace native {

Tensor& bernoulli_tensor_cuda_(Tensor &self, const Tensor& p_, Generator* gen_) {
  NoNamesGuard guard;
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(10);
  }
  auto p = std::get<0>(expand_inplace(self, p_.to(kCUDA)));
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
      using self_t = scalar_t;
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, p.scalar_type(), "bernoulli_tensor_cuda_p_", [&] {
        using p_t = scalar_t;
        return bernoulli_tensor_cuda_kernel<self_t, p_t>(self, p, rng_engine_inputs);
      });
   });
  return self;
}

void bernoulli_scalar_cuda_kernel(TensorIterator& iter, double p_, Generator* gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "bernoulli_scalar_cuda_", [&] {
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

Tensor& bernoulli_scalar_cuda_(Tensor &self, double p, Generator* gen) {
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  auto iter = TensorIterator::nullary_op(self);
  bernoulli_scalar_cuda_kernel(iter, p, gen);
  return self;
}

}} // namespace at::native

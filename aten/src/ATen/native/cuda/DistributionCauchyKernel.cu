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

namespace at { namespace native {

void cauchy_kernel(TensorIterator& iter, double median_, double sigma_, Generator gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "cauchy_cuda", [&] {
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

REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel);

}} // namespace at::native

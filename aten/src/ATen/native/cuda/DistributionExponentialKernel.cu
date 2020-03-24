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

void exponential_kernel(TensorIterator& iter, double lambda_, Generator gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  // Note that HIP doesn't support std::nextafter in device code.
  auto nextafter_1_0_float = std::nextafter(1.0f, 0.0f);
  auto nextafter_1_0_double = std::nextafter(1.0, 0.0);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "exponential_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto lambda = static_cast<accscalar_t>(lambda_);
    if (std::is_same<scalar_t, double>::value) {
      // define lambda for exponential transformation
      auto exponential_func = [lambda, nextafter_1_0_double] __device__ (accscalar_t rand) {
        if (lambda == static_cast<accscalar_t>(0.0)) {
          return static_cast<scalar_t>(0.0);
        }
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
        if (lambda == static_cast<accscalar_t>(0.0)) {
          return static_cast<scalar_t>(0.0);
        }
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

REGISTER_DISPATCH(exponential_stub, &exponential_kernel);

}} // namespace at::native

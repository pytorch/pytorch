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

void log_normal_kernel(TensorIterator& iter, double mean_, double std_, GeneratorHolder gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_cuda", [&] {
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

REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel);

}} // namespace at::native

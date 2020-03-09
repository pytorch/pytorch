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

void geometric_kernel_cuda(TensorIterator& iter, double p_, GeneratorHolder gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "geometric_cuda", [&] {
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

REGISTER_DISPATCH(geometric_stub, &geometric_kernel_cuda);

}} // namespace at::native

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

void normal_kernel_cuda(TensorIterator& iter, double mean_, double std_, GeneratorHolder gen_) {
  auto gen = get_generator_or_default<CUDAGenerator>(gen_, cuda::detail::getDefaultCUDAGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "normal_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda to multiply std and add mean
    auto normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(rand * std + mean);
    };
    if (std::is_same<scalar_t, double>::value) {
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls/2>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
        normal_func);
    } else {
      distribution_nullary_kernel<scalar_t, accscalar_t, curand4_engine_calls>(iter,
        gen,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
        normal_func);
    }
   });
}

Tensor& normal_cuda_(Tensor& self, double mean, double std, GeneratorHolder gen) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);
  auto iter = TensorIterator::nullary_op(self);
  normal_kernel_cuda(iter, mean, std, gen);
  return self;
}

Tensor& normal_out_cuda(Tensor& output, const Tensor& mean, double std, GeneratorHolder gen) {
  normal_cuda_(output, 0, std, gen);
  output.add_(mean);
  return output;
}

Tensor& normal_out_cuda(Tensor& output, double mean, const Tensor& std, GeneratorHolder gen) {
  normal_cuda_(output, 0, 1, gen);
  auto mean_tensor = at::full({}, mean, output.options());
  // NB: addcmul_out copies the tensor to be added into the output.
  // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
  // The previous function here was addcmul_out(output, mean_tensor, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean_tensor + mean_tensor * std instead of mean_tensor + output * std
  output.mul_(std).add_(mean_tensor);
  return output;
}

Tensor& normal_out_cuda(Tensor& output, const Tensor& mean, const Tensor& std, GeneratorHolder gen) {
  bool is_deprecated_th_impl = resize_output_for_normal(output, mean, std);
  normal_cuda_(output, 0, 1, gen);
  // NB: addcmul_out copies the tensor to be added into the output.
  // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
  // The previous function here was addcmul_out(output, mean, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean + mean * std instead of mean + output * std
  if (is_deprecated_th_impl) {
    output.mul_(std.reshape(mean.sizes())).add_(mean);
  }
  else {
    output.mul_(std).add_(mean);
  }
  return output;
}

Tensor normal_cuda(const Tensor& mean, double std, GeneratorHolder gen) {
  Tensor ret = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  normal_out_cuda(ret, mean, std, gen);
  return ret;
}

Tensor normal_cuda(double mean, const Tensor& std, GeneratorHolder gen) {
  Tensor ret = at::empty_like(std, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  normal_out_cuda(ret, mean, std, gen);
  return ret;
}

Tensor normal_cuda(const Tensor& mean, const Tensor& std, GeneratorHolder gen) {
  Tensor ret = at::empty({0}, mean.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  normal_out_cuda(ret, mean, std, gen);
  return ret;
}

}} // namespace at::native

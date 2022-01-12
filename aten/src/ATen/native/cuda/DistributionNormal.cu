#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
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

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

namespace at { namespace native {

void normal_kernel(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::normal_kernel(self, mean, std, generator);
}

REGISTER_DISPATCH(normal_stub, &normal_kernel);

}} // namespace at::native

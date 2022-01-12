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

void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::random_from_to_kernel(iter, range, base, gen);
}

void random_full_64_bits_range_kernel(TensorIteratorBase& iter, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::random_full_64_bits_range_kernel(iter, gen);
}

void random_kernel(TensorIteratorBase& iter, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::random_kernel(iter, gen);
}

REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel);
REGISTER_DISPATCH(random_stub, &random_kernel);
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel);

}} // namespace at::native

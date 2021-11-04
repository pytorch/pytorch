#include <ATen/Dispatch.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

void uniform_kernel(TensorIteratorBase& iter, double from, double to, c10::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  templates::cuda::uniform_kernel(iter, from, to, generator);
}

REGISTER_DISPATCH(uniform_stub, &uniform_kernel);

}} // namespace at::native

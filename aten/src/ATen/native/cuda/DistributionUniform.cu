#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  templates::cuda::uniform_kernel(iter, from, to, generator);
}

REGISTER_DISPATCH(uniform_stub, &uniform_kernel);

} // namespace at::native

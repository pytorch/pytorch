#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::normal_kernel(self, mean, std, generator);
}

REGISTER_DISPATCH(normal_stub, &normal_kernel);

} // namespace at::native

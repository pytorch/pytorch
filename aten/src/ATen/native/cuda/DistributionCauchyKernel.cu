#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::cauchy_kernel(iter, median, sigma, generator);
}

REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel)

} // namespace at::native

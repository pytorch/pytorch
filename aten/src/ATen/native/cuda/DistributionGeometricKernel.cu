#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void geometric_kernel(TensorIteratorBase& iter, double p_, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::geometric_kernel(iter, p_, generator);
}

REGISTER_DISPATCH(geometric_stub, &geometric_kernel)

} // namespace at::native

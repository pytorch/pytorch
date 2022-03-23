#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at { namespace native {

void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, c10::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::log_normal_kernel(iter, mean, std, generator);
}

REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel);

}} // namespace at::native

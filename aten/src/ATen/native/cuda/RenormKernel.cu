#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Normalization.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include <ATen/Dispatch.h>

namespace at {
namespace native{
namespace {

void renorm_scale_factor_impl(TensorIteratorBase& iter, double maxnorm) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "renorm_scale_factor_cpu", [&] {
    const auto maxnorm_s = static_cast<scalar_t>(maxnorm);
    gpu_kernel(
      iter,
      [maxnorm_s] GPU_LAMBDA (scalar_t norm) -> scalar_t {
        const auto eps = static_cast<scalar_t>(1e-7);
        const auto one = static_cast<scalar_t>(1.0);
        return (norm > maxnorm_s) ?
            maxnorm_s / (norm + eps) : one;
      });
  });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(renorm_scale_factor_stub, &renorm_scale_factor_impl);

}}  // namespace at::native

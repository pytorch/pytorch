#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CUDAFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/eq_cuda_dispatch.h>
#include <ATen/ops/equal_native.h>
#endif

namespace at {
namespace native {

bool cuda_equal(const Tensor& self, const Tensor& src) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(
      self.device() == src.device(),
      "Cannot compare two tensors on "
      "different devices. Got: ",
      self.device(),
      " and ",
      src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }
  return at::cuda::eq(self, src).all().item().to<bool>();
}

} // namespace native
} // namespace at

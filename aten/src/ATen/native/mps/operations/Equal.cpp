#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/MPSFunctions.h>
#else
#include <ATen/ops/eq_mps_dispatch.h>
#include <ATen/ops/equal_native.h>
#endif

namespace at {
namespace mps {
TORCH_API at::Tensor eq(const at::Tensor & self, const at::Tensor & other);
} // namespace
namespace native {

bool mps_equal(const Tensor& self, const Tensor &src) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == src.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }
  return at::mps::eq(self, src).all().item().to<bool>();
}

} // namespace native
} // namespace at

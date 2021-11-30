#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

bool cuda_equal(const Tensor& self, const Tensor &src) {
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
  return at::cuda::eq(self, src).all().item().to<bool>();
}

}} // namespace at::native

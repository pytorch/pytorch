#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>

namespace at {
namespace native {

bool cuda_equal_quantized(const Tensor& self, const Tensor& src) {
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
  return at::native::eq(self, src).all().item().to<bool>();
}

} // namespace native
} // namespace at

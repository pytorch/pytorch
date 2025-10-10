#include <ATen/core/op_registration/adaption.h>


namespace c10::impl {

void common_device_check_failure(Device common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
    "Expected all tensors to be on the same device, but got ", argName, " is on ", tensor.device(),
    ", different from other tensors on ", common_device, " (when checking argument in method ", methodName, ")");
}

} // namespace c10::impl

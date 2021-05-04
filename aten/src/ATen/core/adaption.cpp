#include <ATen/core/op_registration/adaption.h>

namespace c10 {
namespace impl {

void common_device_check_failure(optional<Device>& common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
    "Expected all tensors to be on the same device, but "
    "found at least two devices, ", common_device.value(), " and ", tensor.device(), "! "
    "(when checking arugment for argument ", argName, " in method ", methodName, ")");
}

} // namespace impl
} // namespace c10

#include <ATen/DispatchWrapperUtil.h>

namespace c10 {
namespace detail {

void common_device_check_failure(optional<Device>& common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
    "Expected all tensors to be on the same common device, but "
    "found at least two devices, ", common_device.value(), " and ", tensor.device(), "! "
    "(when checking arugment for argument ", argName, " in method ", methodName, ")");
}

void undefined_device_check_failure(at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
    "Tensor is undefined."
    "(when checking arugment for argument ", argName, " in method ", methodName, ")");
}

} // namespace detail
} // namespace c10

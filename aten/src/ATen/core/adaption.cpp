#include <ATen/core/op_registration/adaption.h>

namespace c10 {
namespace impl {

void common_device_check_failure(Device common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
    "Expected all tensors to be on the same device, but "
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    "found at least two devices, ", common_device, " and ", tensor.device(), "! "
    "(when checking argument for argument ", argName, " in method ", methodName, ")");
}

} // namespace impl
} // namespace c10

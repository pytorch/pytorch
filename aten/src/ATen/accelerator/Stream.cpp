#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace at::accelerator {

// NOLINTBEGIN(bugprone-unchecked-optional-access)

void setCurrentStream(c10::Stream stream) {
  const auto device_type = getAccelerator(true).value();
  TORCH_CHECK(
      device_type == stream.device_type(),
      "stream's device type ",
      c10::DeviceTypeName(stream.device_type()),
      " doesn't match the current accelerator ",
      c10::DeviceTypeName(device_type));
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.exchangeStream(stream);
}

c10::Stream getCurrentStream(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getStream({device_type, device_index});
}

// NOLINTEND(bugprone-unchecked-optional-access)

} // namespace at::accelerator

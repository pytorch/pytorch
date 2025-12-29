#include <ATen/accelerator/Accelerator.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace at::accelerator {

// NOLINTBEGIN(bugprone-unchecked-optional-access)

c10::DeviceIndex deviceCount() {
  const auto device_type = getAccelerator(false);
  if (!device_type.has_value()) {
    return static_cast<c10::DeviceIndex>(0);
  }
  c10::impl::VirtualGuardImpl impl(device_type.value());
  return impl.deviceCount();
}

void setDeviceIndex(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.setDevice({device_type, device_index});
}

c10::DeviceIndex getDeviceIndex() {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getDevice().index();
}

void synchronizeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // impl.synchronizeDevice should can be safely called from any device
  impl.synchronizeDevice(device_index);
}

c10::DeviceIndex exchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.exchangeDevice({device_type, device_index}).index();
}

c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // Avoid creating a new context if the context for the given device_index
  // is not initialized.
  impl.uncheckedSetDevice({device_type, device_index});
  return impl.getDevice().index();
}

c10::DeviceCapability getDeviceCapability(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getDeviceCapability({device_type, device_index});
}
// NOLINTEND(bugprone-unchecked-optional-access)

} // namespace at::accelerator

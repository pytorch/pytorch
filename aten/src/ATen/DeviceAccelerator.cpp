#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace at::accelerator {

std::optional<c10::DeviceType> getAccelerator(bool checked) {
#define DETECT_AND_ASSIGN_ACCELERATOR(device_name) \
  if (at::has##device_name()) {                    \
    device_type = k##device_name;                  \
    TORCH_CHECK(                                   \
        !is_accelerator_detected,                  \
        "Cannot have ",                            \
        device_type.value(),                       \
        " with other accelerators.");              \
    is_accelerator_detected = true;                \
  }

  if (is_privateuse1_backend_registered()) {
    // We explicitly allow PrivateUse1 and another device at the same time as we
    // use this for testing. Whenever a PrivateUse1 device is registered, use it
    // first.
    return kPrivateUse1;
  }
  std::optional<c10::DeviceType> device_type = std::nullopt;
  bool is_accelerator_detected = false;
  DETECT_AND_ASSIGN_ACCELERATOR(CUDA)
  DETECT_AND_ASSIGN_ACCELERATOR(MTIA)
  DETECT_AND_ASSIGN_ACCELERATOR(XPU)
  DETECT_AND_ASSIGN_ACCELERATOR(HIP)
  DETECT_AND_ASSIGN_ACCELERATOR(MPS)
  DETECT_AND_ASSIGN_ACCELERATOR(HPU)
  if (checked) {
    TORCH_CHECK(
        device_type, "Cannot access accelerator device when none is available.")
  }
  return device_type;

#undef DETECT_AND_ASSIGN_ACCELERATOR
}

bool isAccelerator(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCUDA:
    case at::kMTIA:
    case at::kXPU:
    case at::kHIP:
    case at::kMPS:
    case at::kHPU:
    case at::kPrivateUse1:
      return true;
    default:
      return false;
  }
}

c10::DeviceIndex deviceCount() {
  const auto device_type = getAccelerator(false);
  if (!device_type.has_value()) {
    return static_cast<c10::DeviceIndex>(0);
  }
  c10::impl::VirtualGuardImpl impl(device_type.value());
  return static_cast<c10::DeviceIndex>(impl.deviceCount());
}

void setDeviceIndex(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.setDevice({device_type, device_index});
}

c10::DeviceIndex getDeviceIndex() {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return static_cast<c10::DeviceIndex>(impl.getDevice().index());
}

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

void synchronizeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // impl.synchronizeDevice should can be safely called from any device
  impl.synchronizeDevice(device_index);
}

} // namespace at::accelerator

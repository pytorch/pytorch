#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>

namespace at {

C10_API std::optional<DeviceType> getAccelerator(bool checked) {
#define CHECK_ACCELERATOR_MUTEX(name)                                          \
  TORCH_CHECK(                                                                 \
      !is_mutex_device_detected, "Cannot have " #name " with other devices."); \
  is_mutex_device_detected = true;

  if (is_privateuse1_backend_registered()) {
    // We explicitly allow PrivateUse1 and another device at the same time as we
    // use this for testing. Whenever a PrivateUse1 device is registered, use it
    // first.
    return kPrivateUse1;
  }
  std::optional<DeviceType> device_type = std::nullopt;
  bool is_mutex_device_detected = false;
  if (at::hasCUDA()) {
    CHECK_ACCELERATOR_MUTEX(CUDA)
    device_type = kCUDA;
  } else if (at::hasMTIA()) {
    CHECK_ACCELERATOR_MUTEX(MTIA)
    device_type = kMTIA;
  } else if (at::hasXPU()) {
    CHECK_ACCELERATOR_MUTEX(XPU)
    device_type = kXPU;
  }
  if (checked) {
    TORCH_CHECK(
        device_type, "Cannot access accelerator device when none is available.")
  }
  return device_type;

#undef CHECK_ACCELERATOR_MUTEX
}

} // namespace at

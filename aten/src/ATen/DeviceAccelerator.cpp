#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
namespace at {

C10_API std::optional<DeviceType> getAccelerator(bool checked) {
#define ASSIGN_ACCELERATOR_AND_CHECK_MUTEX(device_name) \
  if (at::has##device_name()) {                         \
    device_type = k##device_name;                       \
    TORCH_CHECK(                                        \
        !is_mutex_device_detected,                      \
        "Cannot have ",                                 \
        device_type.value(),                            \
        " with other accelerators.");                   \
    is_mutex_device_detected = true;                    \
  }

  if (is_privateuse1_backend_registered()) {
    // We explicitly allow PrivateUse1 and another device at the same time as we
    // use this for testing. Whenever a PrivateUse1 device is registered, use it
    // first.
    return kPrivateUse1;
  }
  std::optional<DeviceType> device_type = std::nullopt;
  bool is_mutex_device_detected = false;
  ASSIGN_ACCELERATOR_AND_CHECK_MUTEX(CUDA)
  ASSIGN_ACCELERATOR_AND_CHECK_MUTEX(MTIA)
  ASSIGN_ACCELERATOR_AND_CHECK_MUTEX(XPU)
  if (checked) {
    TORCH_CHECK(
        device_type, "Cannot access accelerator device when none is available.")
  }
  return device_type;

#undef ASSIGN_ACCELERATOR_AND_CHECK_MUTEX
}

} // namespace at

#include <c10/core/DefaultDevice.h>

namespace c10 {

// For many operators(such as pin_memory), the device argument is `cuda` if
// not given; but for other device, we must have to give extra argument
// `device_type` comparing to `cuda`, so we add an API to set the default
// argument device just once at the begining to keep usage consistent with
// `cuda`.
static c10::DeviceType default_argument_device_type = c10::DeviceType::CUDA;

void set_default_argument_device_type(c10::DeviceType device_type) {
  TORCH_CHECK(
      device_type != c10::DeviceType::CPU,
      "only device type except cpu is supported as default argument device.");
  default_argument_device_type = device_type;
}

c10::DeviceType get_default_argument_device_type() {
  return default_argument_device_type;
}
} // namespace c10

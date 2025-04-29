#pragma once

#include <c10/core/TensorOptions.h>
#include <torch/csrc/Export.h>

// device_lazy_init() is always compiled, even for CPU-only builds.

namespace torch::utils {

/**
 * This mechanism of lazy initialization is designed for each device backend.
 * Currently, CUDA and XPU follow this design. This function `device_lazy_init`
 * MUST be called before you attempt to access any Type(CUDA or XPU) object
 * from ATen, in any way. It guarantees that the device runtime status is lazily
 * initialized when the first runtime API is requested.
 *
 * Here are some common ways that a device object may be retrieved:
 *   - You call getNonVariableType or getNonVariableTypeOpt
 *   - You call toBackend() on a Type
 *
 * It's important to do this correctly, because if you forget to add it you'll
 * get an oblique error message seems like "Cannot initialize CUDA without
 * ATen_cuda library" or "Cannot initialize XPU without ATen_xpu library" if you
 * try to use CUDA or XPU functionality from a CPU-only build, which is not good
 * UX.
 */
TORCH_PYTHON_API void device_lazy_init(at::DeviceType device_type);
TORCH_PYTHON_API void set_requires_device_init(
    at::DeviceType device_type,
    bool value);

inline bool is_device_lazy_init_supported(at::DeviceType device_type) {
  // Add more devices here to enable lazy initialization.
  return (
      device_type == at::DeviceType::CUDA ||
      device_type == at::DeviceType::XPU ||
      device_type == at::DeviceType::HPU ||
      device_type == at::DeviceType::MTIA ||
      device_type == at::DeviceType::PrivateUse1);
}

inline void maybe_initialize_device(at::Device& device) {
  if (is_device_lazy_init_supported(device.type())) {
    device_lazy_init(device.type());
  }
}

inline void maybe_initialize_device(std::optional<at::Device>& device) {
  if (!device.has_value()) {
    return;
  }
  maybe_initialize_device(device.value());
}

inline void maybe_initialize_device(const at::TensorOptions& options) {
  auto device = options.device();
  maybe_initialize_device(device);
}

inline void maybe_initialize_device(
    std::optional<at::DeviceType>& device_type) {
  if (!device_type.has_value()) {
    return;
  }
  maybe_initialize_device(device_type.value());
}

bool is_device_initialized(at::DeviceType device_type);

} // namespace torch::utils

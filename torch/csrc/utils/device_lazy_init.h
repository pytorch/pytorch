#pragma once

#include <c10/core/TensorOptions.h>
#include <torch/csrc/Export.h>

// device_lazy_init() is always compiled, even for CPU-only builds.

namespace torch::utils {

/**
 * Note: [Lazy initialization of device runtime]
 *
 * This lazy initialization mechanism is designed per device backend.
 * Currently, the backends listed in `is_device_lazy_init_supported` follow
 * this design. `device_lazy_init` MUST be called before you access any
 * device-related object from ATen, in any way. It guarantees that the
 * device runtime is lazily initialized on the first runtime API request.
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
 *
 * Lazy-init backends register the fork handler as part of `device_lazy_init`,
 * via `_lazy_init` in their Python module. Backends that don't support lazy
 * init (such as MPS) must register it explicitly via
 * `register_fork_handler_for_device_init`.
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

TORCH_PYTHON_API bool is_device_in_bad_fork(at::DeviceType device_type);

TORCH_PYTHON_API void set_device_in_bad_fork(
    at::DeviceType device_type,
    bool value);

TORCH_PYTHON_API void register_fork_handler_for_device_init(
    at::DeviceType device_type);

inline void maybe_register_fork_handler_for_device_init(
    std::optional<at::DeviceType>& device_type) {
  if (!device_type.has_value()) {
    return;
  }
  register_fork_handler_for_device_init(device_type.value());
}

} // namespace torch::utils

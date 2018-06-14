#include <ATen/DeviceGuard.h>

#include <ATen/Tensor.h>
#include <ATen/TensorMethods.h>
#include <ATen/optional.h>

#include <cstddef>

namespace at {

DeviceGuard::DeviceGuard() = default;

DeviceGuard::DeviceGuard(Device device) {
  set_device(device);
}

DeviceGuard::DeviceGuard(Backend backend, at::optional<int32_t> device_index)
    : DeviceGuard(Device(backend, device_index)) {}

/// Legacy constructor that accepts -1 as the device index and turns it into
/// `at::nullopt`.
DeviceGuard::DeviceGuard(Backend backend, int32_t device_index)
    : DeviceGuard(Device(
          backend,
          device_index == -1 ? nullopt : optional<int32_t>(device_index))) {}

DeviceGuard::DeviceGuard(const Tensor& tensor) {
  set_device_from(tensor);
}

/// Sets the device to the index on which the first tensor in the list is
/// located. If the list is empty, does nothing.
DeviceGuard::DeviceGuard(const TensorList& tensors) {
  if (!tensors.empty()) {
    set_device_from(tensors.front());
  }
}

void DeviceGuard::set_index(int32_t device_index) {
  optional<int32_t> device_index_optional;
  if (device_index != -1) {
    device_index_optional = device_index;
  }
  set_device({at::kCUDA, device_index_optional});
}

void DeviceGuard::set_device_from(const Tensor& tensor) {
  if (tensor.defined()) {
    set_device(Device(tensor));
  }
}

} // namespace at

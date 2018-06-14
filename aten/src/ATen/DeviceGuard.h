#pragma once

#include <ATen/Device.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/optional.h>

#include <cstddef>

namespace at {
/// RAII guard that sets a certain default device in its constructor, and
/// changes it back to the device that was originally active upon destruction.
///
/// The device is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_index` after construction, the
/// destructor will still reset the index to the one that was active at
/// construction time.
///
/// Legacy constructors and setters are kept around that accept -1 as the device
/// index, which is equivalent to `at::nullopt`. All new code should prefer the
/// latter.
struct DeviceGuard {
  /// Default constructor, does nothing.
  DeviceGuard();

  /// Calls `set_device` with the given `Device`. Defined in Device.cpp because
  /// the CUDA device compiler complains otherwise.
  explicit DeviceGuard(Device device);

  /// Convenience constructor that creates a `Device` from the given
  /// arguments and then forwards to the constructor from `Device`.
  DeviceGuard(Backend backend, at::optional<int32_t> device_index);

  /// Legacy constructor that accepts -1 as the device index and turns it into
  /// `at::nullopt`.
  /* deprecated */ DeviceGuard(Backend backend, int32_t device_index);

  /// Sets the device to the index on which the given tensor is located.
  explicit DeviceGuard(const Tensor& tensor);

  /// Sets the device to the index on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit DeviceGuard(const TensorList& tensors);

  /// Resets the device to the index that was active at construction of the
  /// guard.
  ~DeviceGuard() {
    // It should only not have a value if an index was never actually set.
    if (original_device_) {
      // NOTE: When more devices are added, their respective ways of changing
      // the device should be added here.
      if (original_device_->is_cuda()) {
        // Unchecked because we don't want to throw in the destructor.
        detail::DynamicCUDAInterface::unchecked_set_device(
            original_device_->index().value());
      }
    }
  }

  /// Legacy function that sets the backend to CUDA, and accepts -1 as the
  /// device index, which is turned into `at::nullopt`. Use `set_device` for new
  /// code.
  /* deprecated */ void set_index(int32_t device_index);

  /// Sets the device to the given one if its index is not `nullopt`.
  void set_device(Device device) {
    if (!device.has_index()) {
      // Figure out a better strategy here once we really have more than CUDA
      // and CPU devices. For example, the device index may just default to zero
      // if the device type differs from the current device type.
      return;
    }
    if (!original_device_) {
      // Add more ways of swapping the device when more device types are added.
      if (device.is_cuda()) {
        int32_t previous_index = 0;
        detail::DynamicCUDAInterface::get_device(&previous_index);
        original_device_ = Device(Device::Type::CUDA, previous_index);
      } else {
        original_device_ = Device(Device::Type::CPU);
      }
    }
    if (device != original_device_) {
      if (original_device_->is_cuda()) {
        detail::DynamicCUDAInterface::set_device(device.index().value());
      }
    }
    last_device_ = device;
  }

  /// Constructs a `Device` from the given tensor, and then calls `set_device`.
  /// Does nothing if the `tensor` is not defined.
  void set_device_from(const Tensor& tensor);

  /// Returns the device that was set upon construction of the guard.
  const at::optional<Device>& original_device() const noexcept {
    return original_device_;
  }

  // /// Returns the last device that was set via `set_device`, if any.
  const at::optional<Device>& last_device() const noexcept {
    return last_device_;
  }

 private:
  /// The original device that was active at construction of this object.
  at::optional<Device> original_device_;
  at::optional<Device> last_device_;
};
} // namespace at

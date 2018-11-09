#pragma once

#include <ATen/Tensor.h>
#include <c10/Device.h>
#include <ATen/core/ScalarType.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include "c10/util/Optional.h"

#include <cstddef>

namespace at {
/// RAII guard that sets a certain default device in its constructor, and
/// changes it back to the device that was originally active upon destruction.
///
/// The device is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_device` after construction, the
/// destructor will still reset the device to the one that was active at
/// construction time.
struct DeviceGuard {
  /// Default constructor, does nothing.
  DeviceGuard() = default;

  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device) {
    set_device(device);
  }

  explicit DeviceGuard(c10::optional<Device> device_opt) {
    if (device_opt.has_value()) {
      set_device(device_opt.value());
    }
  }

  /// Sets the current device to the device on which the given tensor is located.
  explicit DeviceGuard(const Tensor& tensor) {
    set_device_from(tensor);
  }

  /// Sets the current device to the device on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit DeviceGuard(const TensorList& tensors) {
    if (!tensors.empty()) {
      set_device_from(tensors.front());
    }
  }

  /// Copy is disallowed.
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  /// Move-constructs this `DeviceGuard` from another `DeviceGuard`. The
  /// moved-from `DeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  DeviceGuard(DeviceGuard&& other) noexcept {
    *this = std::move(other);
  }

  /// Move-assigns this `DeviceGuard` from another `DeviceGuard`. The
  /// moved-from `DeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  DeviceGuard& operator=(DeviceGuard&& other) noexcept {
    this->original_index_ = other.original_index_;
    this->last_index_ = other.last_index_;
    // Set other's original index to the unspecified/default state, so that it
    // doesn't also reset the device in its constructor.
    other.original_index_ = -1;
    return *this;
  }

  /// Resets the device to the device that was active at construction of the
  /// guard.
  ~DeviceGuard() {
    // It should only not have a value if an index was never actually set.
    if (original_index_ != -1) {
      // Unchecked because we don't want to throw in the destructor.
      detail::DynamicCUDAInterface::unchecked_set_device(original_index_);
    }
  }

  /// Sets the device to the given one.
  void set_device(at::Device device) {
    if (device.type() == at::kCPU) {
      return;
    }
    AT_ASSERT(device.type() == at::kCUDA);
    auto index = device.index();
    if (index == -1) {
      return;
    }
    AT_ASSERT(index >= 0);
    if (original_index_ == -1) {
      int32_t previous_index = -123;
      detail::DynamicCUDAInterface::get_device(&previous_index);
      original_index_ = previous_index;
      if (index != original_index_) {
        detail::DynamicCUDAInterface::set_device(index);
      }
    } else {
      detail::DynamicCUDAInterface::set_device(index);
    }
    last_index_ = index;
  }

  /// Calls `set_device` with the `Tensor`'s current device, if it is not a
  /// CPU tensor. Does nothing if the `tensor` is not defined.
  void set_device_from(const Tensor& tensor) {
    if (tensor.defined()) {
      set_device(tensor.device());
    }
  }

  /// Returns the device that was set upon construction of the guard.
  at::Device original_device() const noexcept {
    return original_index_ == -1 ? at::kCPU : at::Device(at::kCUDA, original_index_);
  }

  /// Returns the last device that was set via `set_device`, if any.
  at::Device last_device() const noexcept {
    return last_index_ == -1 ? at::kCPU : at::Device(at::kCUDA, last_index_);
  }

 private:
  // This representation only works under the assumption that the DeviceType
  // is only CUDA.  I think a reasonable invariant to assert for DeviceGuard
  // is that once you've "picked" a device type, you can't mix set_device
  // with other device types.

  /// The original device that was active at construction of this object.
  /// If not -1, it is a CUDA device.
  int16_t original_index_ = -1;
  /// The last device that was set via `set_device`.  If not -1, it is a CUDA
  /// device.
  int16_t last_index_ = -1;
};
} // namespace at

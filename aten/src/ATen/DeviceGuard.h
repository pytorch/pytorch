#pragma once

#include <ATen/Device.h>
#include <ATen/Error.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>

#include <cstddef>

namespace at {
/// RAII guard that sets a certain default GPU index in its constructor, and
/// changes it back to the device that was originally active upon destruction.
///
/// The index is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_index` after construction, the
/// destructor will still reset the index to the one that was active at
/// construction time.
struct DeviceGuard {
  /// Default constructor, does nothing.
  DeviceGuard() = default;

  /// Uses the given device's `index()` if it is a CUDA device, else does
  /// nothing.
  explicit DeviceGuard(Device device) {
    if (device.is_cuda()) {
      set_index(device.index());
    }
  }

  /// Calls `set_device` with the given index.
  explicit DeviceGuard(int32_t index) {
    set_index(index);
  }

  /// Sets the device to the index on which the given tensor is located.
  explicit DeviceGuard(const Tensor& tensor) {
    set_index_from(tensor);
  }

  /// Sets the device to the index on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit DeviceGuard(const TensorList& tensors) {
    if (!tensors.empty()) {
      set_index_from(tensors.front());
    }
  }

  /// Resets the device to the index that was active at construction of the
  /// guard.
  ~DeviceGuard() {
    // It should only not have a value if an index was never actually set.
    if (original_index_ != -1) {
      // Unchecked because we don't want to throw in the destructor.
      detail::DynamicCUDAInterface::unchecked_set_device(original_index_);
    }
  }

  /// Sets the device to the given one.
  void set_index(int32_t index) {
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

  /// Calls `set_index` with the `Tensor`'s current device, if it is a CUDA
  /// tensor. Does nothing if the `tensor` is not defined.
  void set_index_from(const Tensor& tensor) {
    if (tensor.defined() && tensor.is_cuda()) {
      set_index(tensor.get_device());
    }
  }

  /// Returns the device that was set upon construction of the guard.
  int32_t original_index() const noexcept {
    return original_index_;
  }

  // /// Returns the last device that was set via `set_device`, if any.
  int32_t last_index() const noexcept {
    return last_index_;
  }

 private:
  /// The original device that was active at construction of this object.
  int32_t original_index_ = -1;
  /// The last index that was set via `set_device`.
  int32_t last_index_ = -1;
};
} // namespace at

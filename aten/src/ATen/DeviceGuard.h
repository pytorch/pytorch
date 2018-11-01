#pragma once

#include <ATen/Tensor.h>
#include <c10/Device.h>
#include <ATen/core/ScalarType.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include "c10/util/Optional.h"
#include <c10/detail/DeviceGuardImplInterface.h>

#include <cstddef>

namespace at {
/// RAII guard that sets a certain default device in its constructor, and
/// changes it back to the device (for that device type) that was originally
/// active upon destruction.
///
/// If the device is changed via this guard to a different one than the
/// active one at construction time, this guard will reset it to the one
/// that was active at the time of construction of the guard.  WARNING: if
/// you change the current device out-of-band, e.g., by directly calling
/// cudaSetDevice(), DeviceGuard is NOT guaranteed to reset it upon
/// exiting this scope.  The contract required by DeviceGuard is that inner code
/// leaves the device in the same state that DeviceGuard set it.  In DEBUG mode,
/// we check for this invariant.
///
/// If a DeviceGuard is constructed without specifying a device type (this
/// can occur if you, e.g., pass a nullopt to the constructor), it behaves as if
/// it were a no-op "CPU" guard; e.g., current_device() reports that the current
/// device is kCPU.  This is different from passing Device(kCUDA, -1), which
/// says to use the current CUDA device; in this case, we will correctly query
/// what the current CUDA device is, but won't change it.
class DeviceGuard {
public:
  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device) {
    init_device(device);
  }

  /// Set the current device to the passed Device, if not nullopt;
  /// otherwise do nothing.
  explicit DeviceGuard(optional<Device> device_opt) {
    if (device_opt.has_value()) {
      init_device(device_opt.value());
    }
  }

  /// Sets the current device to the device on which the given tensor is located.
  explicit DeviceGuard(const Tensor& tensor) {
    init_device_from(tensor);
  }

  /// Sets the current device to the device on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit DeviceGuard(const TensorList& tensors) {
    if (!tensors.empty()) {
      init_device_from(tensors.front());
    }
  }

  /// Copy is disallowed.
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  /// Move-constructs this `DeviceGuard` from another `DeviceGuard`. The
  /// moved-from `DeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  DeviceGuard(DeviceGuard&& other) noexcept {
    // Reuse move assignment implementation
    *this = std::move(other);
  }

  /// Move-assigns this `DeviceGuard` from another `DeviceGuard`. The
  /// moved-from `DeviceGuard` is modified such that its destruction has no
  /// effect (does not reset the device).
  DeviceGuard& operator=(DeviceGuard&& other) noexcept {
    // We cannot use the default move assignment here.  Quoth the standard:
    //
    //    constexpr optional( optional&& other )
    //
    //    If other contains a value, then depending on whether *this contains a
    //    value, the contained value is either direct-initialized or assigned from
    //    *other (2) or std::move(*other) (3). Note that a moved-from optional
    //    still contains a value.
    //
    // Swapping works fine though.
    std::swap(this->impl_, other.impl_);
    std::swap(this->original_device_, other.original_device_);
    std::swap(this->current_device_, other.current_device_);
    return *this;
  }

  /// Resets the device to the device that was active at construction of the
  /// guard.
  ~DeviceGuard() {
#ifdef DEBUG
    // The getDevice call is costly, and also violates noexcept in destructor,
    // so don't test for it outside of DEBUG mode.  If impl_ is nullptr,
    // that indicates CPU; no need to check anything.
    AT_ASSERT(!impl_ || impl_->getDevice() == current_device_);
#endif
    if (original_device_ != current_device_) {
      impl_->uncheckedSetDevice(original_device_);
    }
  }

  /// Returns the device that was set prior to construction of the guard.
  Device original_device() const noexcept {
    return original_device_;
  }

  /// Returns the device that was set after construction of the guard.
  Device current_device() const noexcept {
    return current_device_;
  }

 private:
  void init_device(Device device) {
    if (device.type() == at::kCPU) {
      return;
    }
    impl_ = detail::getDeviceGuardImpl(device.type());
    if (device.index() == -1) {
      original_device_ = impl_->getDevice();
      current_device_ = original_device_;
    } else {
      original_device_ = impl_->exchangeDevice(device);
      current_device_ = device;
    }
  }

  void init_device_from(const Tensor& tensor) {
    if (tensor.defined()) {
      init_device(tensor.device());
    }
  }

  /// The original device that was active at construction of this object,
  /// for the device type that this DeviceGuard is changing.  Defaults to
  /// kCPU if no device type is specified for DeviceGuard.
  Device original_device_ = at::kCPU;

  /// The last device that was set via `set_device`, or the previous
  /// device, if no device was set.  Defaults to kCPU if no device type is
  /// specified for DeviceGuard.

  Device current_device_ = at::kCPU;

  /// Cached pointer to the interface which actually implements the operations
  /// needed for the DeviceGuard.  This is nullptr if the guard is for CPU.
  const detail::DeviceGuardImplInterface* impl_ = nullptr;

  // Member invariants:
  //    !impl_ <==> original_device_ == at::kCPU <==> current_device_ == at::kCPU
};
} // namespace at

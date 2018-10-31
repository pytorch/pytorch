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
/// It's invalid to call `set_device` with devices from different device types;
/// a DeviceGuard only ever handles setting/resetting device for a single
/// device type.  A DeviceGuard always knows what device type it is associated
/// with, which is why we don't provide a nullary constructor (how would it know
/// which device type you want to operate on?)
class DeviceGuard {
public:
  DeviceGuard(DeviceType type) {
    init_device_type(type);
  }

  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device) {
    init_device(device);
  }

  /// Set the current device to the passed Device, if not nullopt;
  /// otherwise do nothing.  It is NOT valid to call set_device
  /// on the resulting device guard.  (See commented out constructor
  /// below if this is ruining your day.)
  explicit DeviceGuard(optional<Device> device_opt) {
    if (device_opt.has_value()) {
      init_device(device_opt.value());
    } else {
      init_device_type(kCPU);
    }
  }

  /*
  /// In principle, this constructor could be useful if you need the
  /// optional<Device> constructor, but you also might want to call
  /// set_device later.  But I don't think anyone will actually need
  /// it in practice.  Feel free to uncomment this if you need it.
  explicit DeviceGuard(DeviceType device_type, optional<Device> device_opt) {
    if (device_opt.has_value()) {
      AT_ASSERT(device_type == device_opt->type());
      init_device(device_opt.value());
    } else {
      init_device_type(device_type);
    }
  }
  */

  /// Sets the current device to the device on which the given tensor is located.
  explicit DeviceGuard(const Tensor& tensor) {
    init_device(tensor.device());
  }

  /// Sets the current device to the device on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit DeviceGuard(const TensorList& tensors) {
    if (!tensors.empty()) {
      init_device(tensors.front().device());
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
    std::swap(this->last_device_, other.last_device_);
    return *this;
  }

  /// Resets the device to the device that was active at construction of the
  /// guard.
  ~DeviceGuard() {
#ifdef DEBUG
    // The getDevice call is costly, and also violates noexcept in destructor,
    // so don't test for it outside of DEBUG mode
    AT_ASSERT(impl_->getDevice() == last_device_);
#endif
    if (original_device_ != last_device_) {
      impl_->uncheckedSetDevice(original_device_);
    }
  }

  /// Calls `set_device` with the `Tensor`'s current device, if it is not a
  /// CPU tensor. Does nothing if the `tensor` is not defined.
  void set_device_from(const Tensor& tensor) {
    if (tensor.defined()) {
      set_device(tensor.device());
    }
  }

  /// Returns the device that was set upon construction of the guard.
  Device original_device() const noexcept {
    return original_device_;
  }

  /// Returns the last device that was set via `set_device`, if any.
  Device last_device() const noexcept {
    return last_device_;
  }

  /// Sets the device to the given one.
  void set_device(at::Device device) {
    AT_ASSERTM(original_device_.type() == device.type(),
               "DeviceGuard was originally used to change the device for ",
               original_device_.type(), ", but set_device() was subsequently ",
               "used to change the device for ", device.type(), ".  To change ",
               "current device for a different device type, you must use a fresh "
               "DeviceGuard.");
    if (device.type() == at::kCPU) return;
    if (device.index() == -1) return;
    AT_ASSERT(device.index() >= 0);
    impl_->setDevice(device);
    last_device_ = device;
  }

 private:
  void init_device_type(DeviceType device_type) {
    if (device_type == at::kCPU) return;
    impl_ = detail::getDeviceGuardImpl(device_type);
    original_device_ = impl_->getDevice();
    last_device_ = original_device_;
  }

  // NB: It would be nice if we could unconditionally reuse
  // the init_device_type() logic here, but that would result in
  // two vcalls on impl_ in the common case, when we only need one.
  void init_device(Device device) {
    if (device.index() == -1) {
      init_device_type(device.type());
      return;
    }
    if (device.type() == at::kCPU) {
      return;
    }
    impl_ = detail::getDeviceGuardImpl(device.type());
    original_device_ = impl_->exchangeDevice(device);
    last_device_ = device;
  }

  /// The original device that was active at construction of this object,
  /// for the device type that this DeviceGuard is changing.
  Device original_device_ = at::kCPU;

  /// The last device that was set via `set_device`, or the previous
  /// device, if no device was set.
  Device last_device_ = at::kCPU;

  // Cached pointer to the interface which actually implements the operations
  // needed for the DeviceGuard.
  const detail::DeviceGuardImplInterface* impl_ = nullptr;

  // Member invariants:
  //    !impl_ <==> original_device_ == at::kCPU <==> last_device_ == at::kCPU
};
} // namespace at

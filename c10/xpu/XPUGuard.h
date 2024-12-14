#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/xpu/impl/XPUGuardImpl.h>

namespace c10::xpu {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for XPU.  It accepts
/// integer indices (interpreting them as XPU devices) and is a little
/// more efficient than DeviceGuard; however, it can only be used
/// from code that links against XPU directly.
struct XPUGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit XPUGuard() = delete;

  /// Set the current XPU device to the passed device index.
  explicit XPUGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current XPU device to the passed device.  Errors if the passed
  /// device is not a XPU device.
  explicit XPUGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  XPUGuard(const XPUGuard&) = delete;
  XPUGuard& operator=(const XPUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  XPUGuard(XPUGuard&& other) = delete;
  XPUGuard& operator=(XPUGuard&& other) = delete;

  /// Sets the XPU device to the given device.  Errors if the given device
  /// is not a XPU device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the XPU device to the given device.  Errors if the given device
  /// is not a XPU device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the XPU device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::XPUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for XPU.  See
/// XPUGuard for when you can use this.
struct OptionalXPUGuard {
  /// Create an uninitialized OptionalXPUGuard.
  explicit OptionalXPUGuard() : guard_() {}

  /// Set the current XPU device to the passed Device, if it is not nullopt.
  explicit OptionalXPUGuard(optional<Device> device_opt) : guard_(device_opt) {}

  /// Set the current XPU device to the passed device index, if it is not
  /// nullopt
  explicit OptionalXPUGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalXPUGuard(const OptionalXPUGuard&) = delete;
  OptionalXPUGuard& operator=(const OptionalXPUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalXPUGuard(OptionalXPUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalXPUGuard& operator=(OptionalXPUGuard&& other) = delete;

  /// Sets the XPU device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a XPU
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the XPU device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a XPU device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the XPU device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original XPU device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::XPUGuardImpl> guard_;
};

} // namespace c10::xpu

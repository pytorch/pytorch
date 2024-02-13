#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/xpu/XPUMacros.h>
#include <c10/xpu/impl/XPUGuardImpl.h>

namespace c10::xpu {

/// A variant of DeviceGuard that is specialized for XPU. It is a little more
/// efficient than DeviceGuard. However, it can only be used from code that
/// links against XPU directly.
struct XPUGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit XPUGuard() = delete;

  /// Set the current XPU device to the passed device index.
  explicit XPUGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current XPU device to the passed device. Errors if the passed
  /// device is not a XPU device.
  explicit XPUGuard(Device device) : guard_(device) {}

  // Copy is not allowed.
  XPUGuard(const XPUGuard&) = delete;
  XPUGuard& operator=(const XPUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state).
  XPUGuard(XPUGuard&& other) = delete;
  XPUGuard& operator=(XPUGuard&& other) = delete;

  /// Sets the XPU device to the given device. Errors if the given device is not
  /// a XPU device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the XPU device to the given device. Errors if the given device is not
  /// a XPU device.
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the XPU device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard.
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via set_device, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::XPUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for XPU. See XPUGuard
/// for when you can use this.
struct OptionalXPUGuard {
  /// Create an uninitialized OptionalXPUGuard.
  explicit OptionalXPUGuard() : guard_() {}

  /// Set the current XPU device to the passed Device, if it is not nullopt.
  explicit OptionalXPUGuard(optional<Device> device_opt) : guard_(device_opt) {}

  /// Set the current XPU device to the passed device index, if it is not
  /// nullopt.
  explicit OptionalXPUGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed.
  OptionalXPUGuard(const OptionalXPUGuard&) = delete;
  OptionalXPUGuard& operator=(const OptionalXPUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalXPUGuard(OptionalXPUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalXPUGuard& operator=(OptionalXPUGuard&& other) = delete;

  /// Sets the XPU device to the given device, initializing the guard if it
  /// is not already initialized. Errors if the given device is not a XPU
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the XPU device to the given device, initializing the guard if it is
  /// not already initialized. Errors if the given device is not a XPU device.
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

/// A variant of StreamGuard that is specialized for XPU. See XPUGuard for when
/// you can use this.
struct XPUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit XPUStreamGuard() = delete;

  /// Set the current XPU device to the device associated with the passed
  /// stream, and set the current XPU stream on that device to the passed
  /// stream. Errors if the Stream is not a XPU stream.
  explicit XPUStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed.
  XPUStreamGuard(const XPUStreamGuard&) = delete;
  XPUStreamGuard& operator=(const XPUStreamGuard&) = delete;

  /// Move is disallowed, as XPUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  XPUStreamGuard(XPUStreamGuard&& other) = delete;
  XPUStreamGuard& operator=(XPUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and the currently
  /// set device to the original device. Then, set the current device to the
  /// device associated with the passed stream, and set the current stream on
  /// that device to the passed stream. Errors if the stream passed is not a
  /// XPU stream.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the XPU stream that was set at the time the guard was constructed.
  XPUStream original_stream() const {
    return XPUStream(XPUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent XPU stream that was set using this device guard,
  /// either from construction, or via set_stream.
  XPUStream current_stream() const {
    return XPUStream(XPUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent XPU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the XPU device that was set at the most recent reset_stream(), or
  /// otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::XPUGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for XPU.  See XPUGuard
/// for when you can use this.
struct OptionalXPUStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalXPUStreamGuard() : guard_() {}

  /// Set the current XPU device to the device associated with the passed
  /// stream, and set the current XPU stream on that device to the passed
  /// stream. Errors if the Stream is not a XPU stream.
  explicit OptionalXPUStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream, if the
  /// passed stream is not nullopt.
  explicit OptionalXPUStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed.
  OptionalXPUStreamGuard(const OptionalXPUStreamGuard&) = delete;
  OptionalXPUStreamGuard& operator=(const OptionalXPUStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalXPUStreamGuard(OptionalXPUStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalXPUStreamGuard& operator=(OptionalXPUStreamGuard&& other) = delete;

  /// Resets the currently set XPU stream to the original stream and the
  /// currently set device to the original device.  Then, set the current device
  /// to the device associated with the passed stream, and set the current
  /// stream on that device to the passed stream. Initializes the guard if it
  /// was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the XPU stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  optional<XPUStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(XPUStream(XPUStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Returns the most recent XPU stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<XPUStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(XPUStream(XPUStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Restore the original XPU device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::XPUGuardImpl> guard_;
};

} // namespace c10::xpu

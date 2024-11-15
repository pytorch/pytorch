#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/xpu/impl/XPUGuardImpl.h>

namespace c10::xpu {

/// A variant of StreamGuard that is specialized for XPU.
/// for when you can use this.
struct XPUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit XPUStreamGuard() = delete;

  /// Set the current XPU device to the device associated with the passed
  /// stream, and set the current XPU stream on that device to the passed
  /// stream. Errors if the Stream is not a XPU stream.
  explicit XPUStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  XPUStreamGuard(const XPUStreamGuard&) = delete;
  XPUStreamGuard& operator=(const XPUStreamGuard&) = delete;

  /// Move is disallowed, as XPUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  XPUStreamGuard(XPUStreamGuard&& other) = delete;
  XPUStreamGuard& operator=(XPUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a XPU stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on XPU, use XPUMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the XPU stream that was set at the time the guard was
  /// constructed.
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

  /// Returns the XPU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::XPUGuardImpl> guard_;
};
} // namespace c10::xpu

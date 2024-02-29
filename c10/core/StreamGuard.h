#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace c10 {

/**
 * A StreamGuard is an RAII class that changes the current device
 * to the device corresponding to some stream, and changes the
 * default stream on that device to be this stream.
 *
 * Use of StreamGuard is HIGHLY discouraged in operator definitions.  In
 * a single operator, you probably don't know enough about the global
 * state of the world to profitably decide how to set streams.  Let
 * the caller handle this appropriately, and just use the current stream
 * in your operator code.
 *
 * This StreamGuard does NOT have an uninitialized state; it is guaranteed
 * to reset the stream and device on exit.  If you are in a situation
 * where you *might* want to setup a stream guard, see OptionalStreamGuard.
 */
struct StreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit StreamGuard() = delete;

  /// Set the current device to the device associated with the passed stream,
  /// and set the current  stream on that device to the passed stream.
  explicit StreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;

  /// Move is disallowed, as StreamGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  StreamGuard(StreamGuard&& other) = delete;
  StreamGuard& operator=(StreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on , use MultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the stream that was set at the time the guard was constructed.
  Stream original_stream() const {
    return guard_.original_stream();
  }

  /// Returns the most recent stream that was set using this device guard,
  /// either from construction, or via set_stream.
  Stream current_stream() const {
    return guard_.current_stream();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::VirtualGuardImpl> guard_;
};

/**
 * An OptionalStreamGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 * See OptionalDeviceGuard for more guidance on how to use this class.
 */
struct OptionalStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalStreamGuard() = default;

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  explicit OptionalStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalStreamGuard(const OptionalStreamGuard&) = delete;
  OptionalStreamGuard& operator=(const OptionalStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalStreamGuard(OptionalStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalStreamGuard& operator=(OptionalStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the stream that was set at the time the guard was most recently
  /// initialized, or nullopt if the guard is uninitialized.
  optional<Stream> original_stream() const {
    return guard_.original_stream();
  }

  /// Returns the most recent  stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<Stream> current_stream() const {
    return guard_.current_stream();
  }

  /// Restore the original  device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::VirtualGuardImpl> guard_{};
};

/**
 * A MultiStreamGuard is an RAII class that sets the current streams of a set of
 * devices all at once, and resets them to their original values on destruction.
 */
struct MultiStreamGuard {
  /// Set the current streams to the passed streams on each of their respective
  /// devices.
  explicit MultiStreamGuard(ArrayRef<Stream> streams) : guard_(streams) {}

  /// Copy is disallowed
  MultiStreamGuard(const MultiStreamGuard&) = delete;
  MultiStreamGuard& operator=(const MultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  MultiStreamGuard(MultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  MultiStreamGuard& operator=(MultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<impl::VirtualGuardImpl> guard_;
};

} // namespace c10

#pragma once

#include <c10/detail/InlineDeviceGuard.h>

namespace c10 {
namespace detail {

/**
 * A StreamGuard is an RAII class that changes the current device
 * to the device corresponding to some stream, and changes the
 * default stream on that device to be this stream.
 *
 * InlineStreamGuard is a helper class for implementing StreamGuards.
 * See InlineDeviceGuard for guidance on how to use this class.
 */
template <typename T>
class InlineStreamGuard : private InlineDeviceGuard<T> {
public:
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit InlineStreamGuard() = delete;

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  explicit InlineStreamGuard(Stream stream)
    : InlineDeviceGuard<T>(stream.device())
    , original_stream_of_original_device_(this->impl_.getStream(original_device()))
    , original_stream_of_current_device_(this->impl_.exchangeStream(stream))
    , current_stream_(stream)
    {}

  /// This constructor exists purely for testing
  template <typename U=T, typename=typename std::enable_if<std::is_same<U, VirtualGuardImpl>::value>::type>
  explicit InlineStreamGuard(Stream stream, const DeviceGuardImplInterface* impl)
    : InlineDeviceGuard<T>(stream.device(), impl ? impl : getDeviceGuardImpl(stream.device_type()))
    , original_stream_of_original_device_(this->impl_.getStream(original_device()))
    , original_stream_of_current_device_(this->impl_.exchangeStream(stream))
    , current_stream_(stream)
    {}

  /// Copy is disallowed
  InlineStreamGuard(const InlineStreamGuard<T>&) = delete;
  InlineStreamGuard<T>& operator=(const InlineStreamGuard<T>&) = delete;

  /// Move is disallowed, as StreamGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  InlineStreamGuard(InlineStreamGuard<T>&& other) = delete;
  InlineStreamGuard& operator=(InlineStreamGuard<T>&& other) = delete;

  ~InlineStreamGuard() {
    this->impl_.exchangeStream(original_stream_of_current_device_);
  }

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
  /// on CUDA, use CUDAMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    // TODO: make a version that takes an impl argument.  Unfortunately,
    // that will require SFINAE because impl is only valid for the
    // VirtualGuardImpl specialization.
    if (stream.device() == this->current_device()) {
      this->impl_.exchangeStream(stream);
      current_stream_ = stream;
    } else {
      // Destruct and reconstruct the StreamGuard in-place
      this->impl_.exchangeStream(original_stream_of_current_device_);
      this->reset_device(stream.device());
      original_stream_of_current_device_ = this->impl_.exchangeStream(stream);
      current_stream_ = stream;
    }
  }

  // It's not clear if set_device should also reset the current stream
  // if the device is unchanged; therefore, we don't provide it.
  // The situation is somewhat clearer with reset_device, but it's still
  // a pretty weird thing to do, so haven't added this either.

  /// Returns the stream of the original device prior to this guard.  Subtly,
  /// the stream returned here is the original stream of the *original*
  /// device; i.e., it's the stream that your computation *would* have
  /// been put on, if it hadn't been for this meddling stream guard.
  /// This is usually what you want.
  Stream original_stream() const {
    return original_stream_of_original_device_;
  }

  /// Returns the most recent stream that was set using this device guard,
  /// either from construction, or via set_stream.
  Stream current_stream() const {
    return current_stream_;
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return InlineDeviceGuard<T>::current_device();
  }

  /// Returns the device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return InlineDeviceGuard<T>::original_device();
  }

private:
  Stream original_stream_of_original_device_; // what the user probably cares about
  Stream original_stream_of_current_device_; // what we need to restore
  Stream current_stream_;
};

/**
 * An OptionalStreamGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 * See InlineOptionalDeviceGuard for more guidance on how to use this class.
 */
template <typename T>
class InlineOptionalStreamGuard {
public:
  /// Creates an uninitialized stream guard.
  explicit InlineOptionalStreamGuard()
    : guard_() // See Note [Explicit initialization of optional fields]
    {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit InlineOptionalStreamGuard(optional<Stream> stream_opt)
    : guard_() {
    if (stream_opt.has_value()) {
      guard_.emplace(stream_opt.value());
    }
  }

  /// All constructors of StreamGuard are valid for OptionalStreamGuard
  template <typename... Args>
  explicit InlineOptionalStreamGuard(Args&&... args)
    : guard_(in_place, std::forward<Args>(args)...) {}

  // See Note [Move construction for RAII guards is tricky]
  InlineOptionalStreamGuard(InlineOptionalStreamGuard<T>&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  InlineOptionalStreamGuard& operator=(InlineOptionalStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the OptionalStreamGuard if it was not previously initialized.
  void reset_stream(Stream stream) {
    if (guard_.has_value()) {
      guard_->reset_stream(stream);
    } else {
      guard_.emplace(stream);
    }
  }

  /// Returns the stream that was set at the time the guard was most recently
  /// initialized, or nullopt if the guard is uninitialized.
  optional<Stream> original_stream() const {
    return guard_.has_value() ? make_optional(guard_->original_stream()) : nullopt;
  }

  /// Returns the most recent stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Stream> current_stream() const {
    return guard_.has_value() ? make_optional(guard_->current_stream()) : nullopt;
  }

  /// Restore the original device and stream, resetting this guard to uninitialized state.
  void reset() {
    guard_.reset();
  }

private:
  optional<InlineStreamGuard<T>> guard_;
};

}} // namespace c10::detail

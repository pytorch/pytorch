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
 */
template <typename T>
class InlineStreamGuard : private InlineDeviceGuard<T> {
public:
  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  explicit InlineStreamGuard(Stream stream)
    : InlineDeviceGuard<T>(stream.device())
    , original_stream_(this->impl_.exchangeStream(stream))
    , current_stream_(stream)
    {}

  // This constructor exists purely for testing
  template <typename U=T, typename=std::enable_if<std::is_same<U, VirtualGuardImpl>::value >>
  explicit InlineStreamGuard(Stream stream, const DeviceGuardImplInterface* impl)
    : InlineDeviceGuard<T>(stream.device(), impl)
    , original_stream_(this->impl_.exchangeStream(stream))
    , current_stream_(stream)
    {}

  /// Copy is disallowed
  InlineStreamGuard(const InlineStreamGuard<T>&) = delete;
  InlineStreamGuard<T>& operator=(const InlineStreamGuard<T>&) = delete;

  /// Move is disallowed, as DeviceGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  InlineStreamGuard(InlineStreamGuard<T>&& other) = delete;
  InlineStreamGuard& operator=(InlineStreamGuard<T>&& other) = delete;

  ~InlineStreamGuard() {
    this->impl_.exchangeStream(original_stream_);
  }

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.  If
  /// a device change occurred, *reset* the current stream on the previous
  /// device to its original value.
  ///
  /// (If you need to remember streams, consider using CUDAMultiStreamGuard.)
  void set_stream(Stream stream) {
    if (stream.device() == this->original_device()) {
      this->impl_.exchangeStream(stream);
      current_stream_ = stream;
    } else {
      // Destruct and reconstruct the StreamGuard in-place
      this->impl_.exchangeStream(original_stream_);
      this->set_device(stream.device());
      original_stream_ = this->impl_.exchangeStream(stream);
      current_stream_ = stream;
    }
  }

  // We could probably provide set_device, but be careful: you need to restore
  // the stream on the current device before moving to the new device.  (It's
  // also weird to think about what set_device should do if it's a no-op;
  // probably shouldn't reset the current stream... it's just weird that
  // the current stream changes in this case.)

  /// Returns the stream that was set at the time the guard was constructed.
  Stream original_stream() const {
    return original_stream_;
  }

  /// Returns the most recent stream that was set using this device guard,
  /// either from construction, or via set_stream.
  Stream current_stream() const {
    return current_stream_;
  }

  Device current_device() const {
    return InlineDeviceGuard<T>::current_device();
  }

  Device original_device() const {
    return InlineDeviceGuard<T>::original_device();
  }

private:
  Stream original_stream_;
  Stream current_stream_;
};

/**
 * A MaybeStreamGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 */
template <typename T>
class InlineMaybeStreamGuard {
public:
  // Note [Explicit initialization of optional fields]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Explicit initialization of optional fields
  // required to workaround an nvcc bug; see https://github.com/pytorch/pytorch/issues/12117


  /// Default constructor, reads the current device so that
  /// we may reset the device to the current device on destruction.
  explicit InlineMaybeStreamGuard()
    : guard_() // See Note [Explicit initialization of optional fields]
    {}

  /// Set the current device to the passed Device
  explicit InlineMaybeStreamGuard(optional<Device> device_opt)
    : guard_() {
    if (device_opt.has_value()) {
      guard_.emplace(device_opt.value());
    }
  }

  /// All constructors of StreamGuard are valid for MaybeStreamGuard
  template <typename... Args>
  explicit InlineMaybeStreamGuard(Args&&... args)
    : guard_(in_place, std::forward<Args>(args)...) {}

  // See Note [Move construction for RAII guards is tricky]
  InlineMaybeStreamGuard(InlineMaybeStreamGuard<T>&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  InlineMaybeStreamGuard& operator=(InlineMaybeStreamGuard&& other) = delete;

  void set_stream(Stream stream) {
    if (guard_.has_value()) {
      guard_->set_stream(stream);
    } else {
      guard_.emplace(stream);
    }
  }

  /// Returns the stream that was set at the time the guard was initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Stream> original_stream() const {
    return guard_.has_value() ? make_optional(guard_->original_stream()) : nullopt;
  }

  /// Returns the most recent stream that was set using this stream guard,
  /// either from construction, or via set_stream, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Stream> current_stream() const {
    return guard_.has_value() ? make_optional(guard_->current_stream()) : nullopt;
  }

private:
  optional<InlineStreamGuard<T>> guard_;
};

}} // namespace c10::detail

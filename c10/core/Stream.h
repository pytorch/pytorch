#pragma once

#include <c10/core/Device.h>

namespace c10 {

/// An index representing a specific stream.  A StreamId is not independently
/// meaningful without knowing the Device it is associated with; try to
/// use Stream rather than StreamId directly.
///
/// StreamIds are opaque; they are assigned by some DeviceType-specific
/// numbering system which is not visible to the user.  HOWEVER, we
/// guarantee that StreamId 0 is always a valid stream, and corresponds
/// to some sort of "default" stream.
using StreamId = int64_t;

// NB: I decided not to call the above StreamIndex to avoid confusion with
// DeviceIndex.  This way, you access device index with index(), and stream id
// with id()

/**
 * A stream is a software mechanism used to synchronize launched kernels
 * without requiring explicit synchronizations between kernels.  The basic
 * model is that every kernel launch is associated with a stream: every
 * kernel on the same stream is implicitly synchronized so that if I launch
 * kernels A and B on the same stream, A is guaranteed to finish before B
 * launches.  If I want B to run concurrently with A, I must schedule
 * it on a different stream.
 *
 * The Stream class is a backend agnostic value class representing a stream
 * which I may schedule a kernel on.  Every stream is associated with a device,
 * which is recorded in stream, which is used to avoid confusion about which
 * device a stream refers to.
 *
 * Streams are explicitly thread-safe, in the sense that it is OK to pass
 * a Stream from one thread to another, and kernels queued from two different
 * threads will still get serialized appropriately.  (Of course, the
 * time when the kernels get queued is undetermined unless you synchronize
 * host side ;)
 *
 * Stream does NOT have a default constructor.  Streams are for expert
 * users; if you want to use Streams, we're going to assume you know
 * how to deal with C++ template error messages if you try to
 * resize() a vector of Streams.
 *
 * Known instances of streams in backends:
 *
 *  - cudaStream_t (CUDA)
 *  - hipStream_t (HIP)
 *  - cl_command_queue (OpenCL)  (NB: Caffe2's existing OpenCL integration
 *    does NOT support command queues.)
 *
 * Because this class is device agnostic, it cannot provide backend-specific
 * functionality (e.g., get the cudaStream_t of a CUDA stream.)  There are
 * wrapper classes which provide this functionality, e.g., CUDAStream.
 */
class C10_API Stream final {
 private:
  Device device_;
  StreamId id_;

 public:
  enum Unsafe { UNSAFE };
  enum Default { DEFAULT };

  /// Unsafely construct a stream from a Device and a StreamId.  In
  /// general, only specific implementations of streams for a
  /// backend should manufacture Stream directly in this way; other users
  /// should use the provided APIs to get a stream.  In particular,
  /// we don't require backends to give any guarantees about non-zero
  /// StreamIds; they are welcome to allocate in whatever way they like.
  explicit Stream(Unsafe, Device device, StreamId id)
      : device_(device), id_(id) {}

  /// Construct the default stream of a Device.  The default stream is
  /// NOT the same as the current stream; default stream is a fixed stream
  /// that never changes, whereas the current stream may be changed by
  /// StreamGuard.
  explicit Stream(Default, Device device) : device_(device), id_(0) {}

  bool operator==(const Stream& other) const noexcept {
    return this->device_ == other.device_ && this->id_ == other.id_;
  }
  bool operator!=(const Stream& other) const noexcept {
    return !(*this == other);
  }

  Device device() const noexcept {
    return device_;
  }
  DeviceType device_type() const noexcept {
    return device_.type();
  }
  DeviceIndex device_index() const noexcept {
    return device_.index();
  }
  StreamId id() const noexcept {
    return id_;
  }

  // Enqueues a wait instruction in the stream's work queue.
  // This instruction is a no-op unless the event is marked
  // for recording. In that case the stream stops processing
  // until the event is recorded.
  template <typename T>
  void wait(const T& event) const {
    event.block(*this);
  }

  // Return whether all asynchronous work previously enqueued on this stream
  // has completed running on the device.
  bool query() const;

  // Wait (by blocking the calling thread) until all asynchronous work enqueued
  // on this stream has completed running on the device.
  void synchronize() const;

  // The purpose of this function is to more conveniently permit binding
  // of Stream to and from Python.  Without packing, I have to setup a whole
  // class with two fields (device and stream id); with packing I can just
  // store a single uint64_t.
  //
  // The particular way we pack streams into a uint64_t is considered an
  // implementation detail and should not be relied upon.
  uint64_t pack() const noexcept {
    // Are you here because this static assert failed?  Make sure you ensure
    // that the bitmasking code below is updated accordingly!
    static_assert(sizeof(DeviceType) == 1, "DeviceType is not 8-bit");
    static_assert(sizeof(DeviceIndex) == 1, "DeviceIndex is not 8-bit");
    static_assert(sizeof(StreamId) == 8, "StreamId is not 64-bit");
    // Concat these together into a 64-bit integer
    // See Note [Hazard when concatenating signed integers]
    uint64_t bits = static_cast<uint64_t>(static_cast<uint8_t>(device_type()))
            << 56 |
        static_cast<uint64_t>(static_cast<uint8_t>(device_index())) << 48 |
        // Remove the sign extension part of the 64-bit address because
        // the id might be used to hold a pointer.
        (static_cast<uint64_t>(id()) & ((1ull << 48) - 1));
    TORCH_INTERNAL_ASSERT(
        static_cast<DeviceIndex>((bits >> 48) & 0xFFull) == device_index(),
        "DeviceIndex is not correctly packed");
    TORCH_INTERNAL_ASSERT(
        static_cast<DeviceType>((bits >> 56)) == device_type(),
        "DeviceType is not correctly packed");
    // Re-extend the sign of stream_id for checking
    uint64_t mask = (1ull << 47);
    TORCH_INTERNAL_ASSERT(
        static_cast<StreamId>(((bits & 0xFFFFFFFFFFFFull) ^ mask) - mask) ==
            id(),
        "DeviceType is not correctly packed");
    return bits;
  }

  static Stream unpack(uint64_t bits) {
    // Re-extend the sign of stream_id
    uint64_t mask = (1ull << 47);
    const auto stream_id =
        (static_cast<StreamId>(bits & 0xFFFFFFFFFFFFull) ^ mask) - mask;
    bits >>= 48;
    const auto device_index = static_cast<DeviceIndex>(bits & 0xFFull);
    bits >>= 8;
    const auto device_type = static_cast<DeviceType>(bits);
    TORCH_CHECK(isValidDeviceType(device_type));
    // Unfortunately, we can't check if the StreamId is valid here; it
    // will be checked upon first use.
    return Stream(UNSAFE, Device(device_type, device_index), stream_id);
  }

  // I decided NOT to provide setters on this class, because really,
  // why would you change the device of a stream?  Just construct
  // it correctly from the beginning dude.
};

C10_API std::ostream& operator<<(std::ostream& stream, const Stream& s);

} // namespace c10

namespace std {
template <>
struct hash<c10::Stream> {
  size_t operator()(c10::Stream s) const noexcept {
    return std::hash<uint64_t>{}(s.pack());
  }
};
} // namespace std

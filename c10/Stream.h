#pragma once

#include <c10/Device.h>

namespace c10 {

/// An index representing a specific stream.  A StreamId is not independently
/// meaningful without knowing the Device it is associated with; try to
/// use Stream rather than StreamId directly.
using StreamId = int32_t;

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
class Stream {
private:
  Device device_;
  StreamId id_;
public:
  explicit Stream(Device device, StreamId id)
    : device_(device)
    , id_(id) {}

  bool operator==(const Stream& other) const noexcept {
    return this->device_ == other.device_ && this->id_ == other.id_;
  }
  bool operator!=(const Stream& other) const noexcept {
    return !(*this == other);
  }

  Device device() const noexcept { return device_; }
  DeviceType device_type() const noexcept { return device_.type(); }
  DeviceIndex device_index() const noexcept { return device_.index(); }
  StreamId id() const noexcept { return id_; }

  // I decided NOT to provide setters on this class, because really,
  // why would you change the device of a stream?  Just construct
  // it correctly from the beginning dude.
};

C10_API std::ostream& operator<<(std::ostream& stream, const Stream& s);

} // namespace c10

namespace at {
  using c10::StreamId;
  using c10::Stream;
}

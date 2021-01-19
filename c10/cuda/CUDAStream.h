#pragma once

#include <cstdint>
#include <utility>

#include <cuda_runtime_api.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>
#include <c10/core/Stream.h>

/*
* Stream pool note.
*
* A CUDAStream is an abstraction of an actual cuStream on the GPU. CUDAStreams
* are backed by cuStreams, but they use several pools to minimize the costs
* associated with creating, retaining, and destroying cuStreams.
*
* There are three pools per device, and a device's pools are lazily created.
*
* The first pool contains only the default stream. When the default stream
* is requested it's returned.
*
* The second pool is the "low priority" or "default priority" streams. In
* HIP builds there is no distinction between streams in this pool and streams
* in the third pool (below). There are 32 of these streams per device, and
* when a stream is requested one of these streams is returned round-robin.
* That is, the first stream requested is at index 0, the second at index 1...
* to index 31, then index 0 again.
*
* This means that if 33 low priority streams are requested, the first and
* last streams requested are actually the same stream (under the covers)
* and kernels enqueued on them cannot run concurrently.
*
* The third pool is the "high priority" streams. The third pool acts like
* the second pool except the streams are created with a higher priority.
*
* These pools suggest that stream users should prefer many short-lived streams,
* as the cost of acquiring and releasing streams is effectively zero. If
* many longer-lived streams are required in performance critical scenarios
* then the functionality here may need to be extended to allow, for example,
* "reserving" a subset of the pool so that other streams do not accidentally
* overlap the performance critical streams.
*
* Note: although the notion of "current stream for device" is thread local
* (every OS thread has a separate current stream, as one might expect),
* the stream pool is global across all threads; stream 0 is always stream 0
* no matter which thread you use it on.  Multiple threads can synchronize
* on the same stream.  Although the CUDA documentation is not very clear
* on the matter, streams are thread safe; e.g., it is safe to enqueue
* a kernel on the same stream from two different threads.
*/

namespace c10 {
namespace cuda {

// Value object representing a CUDA stream.  This is just a wrapper
// around c10::Stream, but it comes with a little extra CUDA-specific
// functionality (conversion to cudaStream_t), and a guarantee that
// the wrapped c10::Stream really is a CUDA stream.
class C10_CUDA_API CUDAStream {
public:

  enum Unchecked { UNCHECKED };

  /// Construct a CUDAStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a CUDA stream.
  explicit CUDAStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
  }

  /// Construct a CUDAStream from a Stream with no error checking.
  /// This constructor uses the "named" constructor idiom, and can
  /// be invoked as: CUDAStream(CUDAStream::UNCHECKED, stream)
  explicit CUDAStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const CUDAStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const CUDAStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to cudaStream_t.
  operator cudaStream_t() const { return stream(); }

  /// Implicit conversion to Stream (a.k.a., forget that the stream is a
  /// CUDA stream).
  operator Stream() const { return unwrap(); }

  /// Get the CUDA device index that this stream is associated with.
  DeviceIndex device_index() const { return stream_.device_index(); }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a CUDA device.
  Device device() const { return Device(DeviceType::CUDA, device_index()); }

  /// Return the stream ID corresponding to this particular stream.
  StreamId id() const { return stream_.id(); }

  bool query() const {
    DeviceGuard guard{stream_.device()};
    cudaError_t err = cudaStreamQuery(stream());

    if (err == cudaSuccess) {
      return true;
    } else if (err != cudaErrorNotReady) {
      C10_CUDA_CHECK(err);
    }

    return false;
  }

  void synchronize() const {
    DeviceGuard guard{stream_.device()};
    C10_CUDA_CHECK(cudaStreamSynchronize(stream()));
  }

  int priority() const {
      DeviceGuard guard{stream_.device()};
      int priority = 0;
      C10_CUDA_CHECK(cudaStreamGetPriority(stream(), &priority));
      return priority;
  }

  /// Explicit conversion to cudaStream_t.
  cudaStream_t stream() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const { return stream_; }

  /// Reversibly pack a CUDAStream into a uint64_t representation.  This may
  /// be helpful when storing a CUDAStream in a C struct, where you cannot
  /// conveniently place the CUDAStream object itself (which is morally
  /// equivalent, but unfortunately is not POD due to the fact that it
  /// has constructors.)
  ///
  /// The CUDAStream can be unpacked using unpack().  The format of
  /// the uint64_t is unspecified and may be changed.
  uint64_t pack() const noexcept {
    return stream_.pack();
  }

  // Unpack a CUDAStream from the uint64_t representation generated by pack().
  static CUDAStream unpack(uint64_t bits) {
    return CUDAStream(Stream::unpack(bits));
  }

  static std::tuple<int, int> priority_range() {
      // Note: this returns the range of priority **supported by PyTorch**, not
      // the range of priority **supported by CUDA**. The former is a subset of
      // the latter. Currently PyTorch only supports 0 and -1, which are "low" and
      // "high" priority.
      int least_priority, greatest_priority;
      C10_CUDA_CHECK(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
      TORCH_INTERNAL_ASSERT(least_priority >= 0, "Unexpected CUDA stream priority range");
      TORCH_INTERNAL_ASSERT(greatest_priority <= -1, "Unexpected CUDA stream priority range");
      return std::make_tuple(0, -1);
  }

  // Deleted for now; use CUDAEvent::block instead
  // void synchronize_with(const CUDAEvent& event) const;

private:
  Stream stream_;
};

/**
 * Get a new stream from the CUDA stream pool.  You can think of this
 * as "creating" a new stream, but no such creation actually happens;
 * instead, streams are preallocated from the pool and returned in a
 * round-robin fashion.
 *
 * You can request a stream from the high priority pool by setting
 * isHighPriority to true, or a stream for a specific device by setting device
 * (defaulting to the current CUDA stream.)
 */
TORCH_API CUDAStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

/**
 * Get the default CUDA stream, for the passed CUDA device, or for the
 * current device if no device index is passed.  The default stream is
 * where most computation occurs when you aren't explicitly using
 * streams.
 */
TORCH_API CUDAStream getDefaultCUDAStream(DeviceIndex device_index = -1);

/**
 * Get the current CUDA stream, for the passed CUDA device, or for the
 * current device if no device index is passed.  The current CUDA stream
 * will usually be the default CUDA stream for the device, but it may
 * be different if someone called 'setCurrentCUDAStream' or used 'StreamGuard'
 * or 'CUDAStreamGuard'.
 */
TORCH_API CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1);

/**
 * Set the current stream on the device of the passed in stream to be
 * the passed in stream.  Yes, you read that right: this function
 * has *nothing* to do with the current device: it toggles the current
 * stream of the device of the passed stream.
 *
 * Confused?  Avoid using this function; prefer using 'CUDAStreamGuard' instead
 * (which will switch both your current device and current stream in the way you
 * expect, and reset it back to its original state afterwards).
 */
TORCH_API void setCurrentCUDAStream(CUDAStream stream);

C10_API std::ostream& operator<<(std::ostream& stream, const CUDAStream& s);

} // namespace cuda
} // namespace at

namespace std {
  template <>
  struct hash<c10::cuda::CUDAStream> {
    size_t operator()(c10::cuda::CUDAStream s) const noexcept {
      return std::hash<c10::Stream>{}(s.unwrap());
    }
  };
} // namespace std

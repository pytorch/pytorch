#pragma once

#include <cstdint>
#include <utility>

#include "cuda_runtime_api.h"

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <c10/util/Exception.h>
#include <c10/Stream.h>

/*
* A CUDAStream interface. See CUDAStream.cpp for implementation details.
*
* Includes the CUDAStream convenience class and a pointer-based stream API.
*
* The ATen/cuda/CUDAContext interface should be preferred when working with streams.
*/

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
*/

struct CUDAStreamInternals;

namespace at {
namespace cuda {

struct CUDAEvent;

namespace detail {

// Pointer-based API (for internal use, backwards compatibility with C-based API)
AT_CUDA_API CUDAStreamInternals* CUDAStream_getDefaultStream(int64_t device = -1);

AT_CUDA_API CUDAStreamInternals* CUDAStream_getStreamFromPool(
  const bool isHighPriority = false
, int64_t device = -1);

AT_CUDA_API CUDAStreamInternals* CUDAStream_getCurrentStream(int64_t device = -1);

AT_CUDA_API void CUDAStream_setStream(CUDAStreamInternals* internals);
AT_CUDA_API void CUDAStream_uncheckedSetStream(CUDAStreamInternals* internals);

AT_CUDA_API cudaStream_t CUDAStream_stream(const CUDAStreamInternals*);
AT_CUDA_API int64_t CUDAStream_device(const CUDAStreamInternals*);

} // namespace detail

// RAII for a CUDA stream
// Allows use as a cudaStream_t, copying, moving, and metadata access.
struct AT_CUDA_API CUDAStream {

  enum Unchecked { UNCHECKED };

  explicit CUDAStream(const CUDAStreamInternals*);

  explicit CUDAStream(Stream stream) : stream_(stream) {
    AT_CHECK(stream_.device_type() == DeviceType::CUDA);
  }

  explicit CUDAStream(Unchecked, Stream stream) : stream_(stream) {}

  // Implicit conversion to cudaStream_t
  operator cudaStream_t() const { return stream(); }
  operator Stream() const { return unwrap(); }

  // Getters
  int64_t device_index() const { return stream_.device_index(); }
  Device device() const { return Device(DeviceType::CUDA, device_index()); }
  cudaStream_t stream() const { return detail::CUDAStream_stream(internals()); }
  CUDAStreamInternals* internals() const;

  Stream unwrap() const { return stream_; }

  void synchronize_with(const CUDAEvent& event) const;

private:
  Stream stream_;
};

} // namespace cuda
} // namespace at

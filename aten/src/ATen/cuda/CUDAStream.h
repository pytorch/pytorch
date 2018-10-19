#pragma once

#include <cstdint>
#include <utility>

#include "cuda_runtime_api.h"

#include <ATen/cuda/ATenCUDAGeneral.h>

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

// Pointer-based API (for internal use)
AT_CUDA_API CUDAStreamInternals* CUDAStream_getDefaultStream(int64_t device = -1);

AT_CUDA_API CUDAStreamInternals* CUDAStream_createStream(
  const bool isHighPriority = false
, int64_t device = -1);

AT_CUDA_API CUDAStreamInternals* CUDAStream_getCurrentStream(int64_t device = -1);

AT_CUDA_API void CUDAStream_setStream(CUDAStreamInternals* internals);
AT_CUDA_API void CUDAStream_uncheckedSetStream(CUDAStreamInternals* internals);

AT_CUDA_API cudaStream_t CUDAStream_stream(CUDAStreamInternals*);
AT_CUDA_API int64_t CUDAStream_device(CUDAStreamInternals*);

} // namespace detail

// RAII for a CUDA stream
// Allows use as a cudaStream_t, copying, moving, and metadata access.
struct AT_CUDA_API CUDAStream {

  // Constructors
  CUDAStream() = default;
  /* implicit */ CUDAStream(CUDAStreamInternals* internals_in)
  : internals_{internals_in} { }

  // Returns true if the CUDAStream is not null.
  explicit operator bool() const noexcept { return internals_ != nullptr; }

  // Implicit conversion to cudaStream_t
  operator cudaStream_t() const { return detail::CUDAStream_stream(internals_); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CUDAStream& left, const CUDAStream& right) {
    return left.internals_ < right.internals_;
  }

  // Getters
  int64_t device() const { return detail::CUDAStream_device(internals_); }
  cudaStream_t stream() const { return detail::CUDAStream_stream(internals_); }
  CUDAStreamInternals* internals() const { return internals_; }

  void synchronize_with(const CUDAEvent& event) const;

private:
  CUDAStreamInternals* internals_ = nullptr;
};

} // namespace cuda
} // namespace at

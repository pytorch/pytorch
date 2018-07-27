#pragma once

#include <cstdint>
#include <utility>

#include "cuda_runtime_api.h"

#include <ATen/ATenGeneral.h>

/*
* A CUDAStream interface. See CUDAStream.cpp for implementation details.
*
* Includes the CUDAStream convenience class and a pointer-based stream API.
*
* The ATen/cuda/CUDAContext interface should be preferred when working with streams.
*/

// Forward-declares internals
struct CUDAStreamInternals;

namespace at {
namespace cuda {

namespace detail {

// Pointer-based API (for internal use)
AT_API CUDAStreamInternals* CUDAStream_getDefaultStream(int64_t device = -1);

AT_API CUDAStreamInternals* CUDAStream_createStream(
  const bool isHighPriority = false
, int64_t device = -1);

AT_API CUDAStreamInternals* CUDAStream_getCurrentStream(int64_t device = -1);

AT_API void CUDAStream_setStream(CUDAStreamInternals* internals);
AT_API void CUDAStream_uncheckedSetStream(CUDAStreamInternals* internals);

AT_API cudaStream_t CUDAStream_stream(CUDAStreamInternals*);
AT_API int64_t CUDAStream_device(CUDAStreamInternals*);

} // namespace detail

// RAII for a CUDA stream
// Allows use as a cudaStream_t, copying, moving, and metadata access.
struct CUDAStream {

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

private:
  CUDAStreamInternals* internals_ = nullptr;
};

} // namespace cuda
} // namespace at

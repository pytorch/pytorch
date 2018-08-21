#pragma once

#include <cstdint>
#include <utility>

#include "cuda_runtime_api.h"

#include <ATen/ATenGeneral.h>

/*
* A CUDA stream interface with no CUDA build dependency.
*
* Includes the CUDAStream RAII class and a pointer-based stream API.
*
* The ATen Context interface should be preferred when working with streams.
*/

struct CUDAStreamInternals;

namespace at {
namespace cuda {

struct CUDAEvent;

namespace detail {

// Pointer-based API (for internal use)
// Note: ATen/Context is preferred to work with streams safely
AT_API CUDAStreamInternals* CUDAStream_getDefaultStreamOnDevice(int64_t device);
AT_API CUDAStreamInternals* CUDAStream_getDefaultStream();

AT_API CUDAStreamInternals* CUDAStream_createAndRetainWithOptions(int32_t flags, int32_t priority);

AT_API CUDAStreamInternals* CUDAStream_getAndRetainCurrentStreamOnDevice(int64_t device);
AT_API CUDAStreamInternals* CUDAStream_getAndRetainCurrentStream();

// Note: these Unsafe gets should NEVER be used and are only here for legacy
// purposes. Once those uses are gone they should be removed.
AT_API CUDAStreamInternals* CUDAStream_getCurrentStreamOnDeviceUnsafe(int64_t device);
AT_API CUDAStreamInternals* CUDAStream_getCurrentStreamUnsafe();

AT_API void CUDAStream_setStreamOnDevice(int64_t device, CUDAStreamInternals* internals);
AT_API void CUDAStream_uncheckedSetStreamOnDevice(
    int64_t device,
    CUDAStreamInternals* internals);
AT_API void CUDAStream_setStream(CUDAStreamInternals* internals);

AT_API cudaStream_t CUDAStream_stream(CUDAStreamInternals*);
AT_API int64_t CUDAStream_device(CUDAStreamInternals*);

AT_API bool CUDAStream_retain(CUDAStreamInternals*);
AT_API void CUDAStream_free(CUDAStreamInternals*&);
AT_API void CUDAStream_uncheckedFree(CUDAStreamInternals*&);

} // namespace detail

// RAII for a CUDA stream
// Allows use as a cudaStream_t, copying, moving, and metadata access.
struct CUDAStream {
  // Constants
  static constexpr int32_t DEFAULT_FLAGS = cudaStreamNonBlocking;
  static constexpr int32_t DEFAULT_PRIORITY = 0;

  // Constructors
  CUDAStream() = default;
  /* implicit */ CUDAStream(CUDAStreamInternals* internals, bool retain = false)
      : internals_{internals} {
    if (retain) {
      detail::CUDAStream_retain(internals_);
    }
  }

  // Destructor
  ~CUDAStream() { detail::CUDAStream_uncheckedFree(internals_); }

  // Copy constructor
  AT_API CUDAStream(const CUDAStream& other);

  // Move constructor
  AT_API CUDAStream(CUDAStream&& other);

  // Assignment operator
  CUDAStream& operator=(CUDAStream other) noexcept {
    std::swap(internals_, other.internals_);
    return *this;
  }

  // Returns true if the CUDAStream is not null.
  explicit operator bool() const noexcept {
    return internals_ != nullptr;
  }

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

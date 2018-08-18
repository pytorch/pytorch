#pragma once

#include <cstdint>
#include <utility>

#include "cuda_runtime_api.h"

#include <ATen/ATenGeneral.h>
#include <ATen/Error.h>

/*
* A CUDA event interface with no CUDA build dependency.
*
* Includes the CUDAEvent RAII class and a pointer-based event API.
*/

struct CUDAEventInternals;

namespace at {
namespace cuda {

struct CUDAStream;

namespace detail {

// Pointer-based API (for internal use)
// Note: ATen/Context is preferred to work with streams safely
AT_API CUDAEventInternals* CUDAEvent_create(unsigned int flags);
AT_API void CUDAEvent_retain(CUDAEventInternals* internals);
AT_API void CUDAEvent_uncheckedFree(CUDAEventInternals* internals);
AT_API cudaEvent_t CUDAEvent_event(CUDAEventInternals* internals);
AT_API int64_t CUDAEvent_device(CUDAEventInternals* internals);

} // namespace detail

struct CUDAEvent {
  // Constants
  static constexpr unsigned int DEFAULT_FLAGS = cudaEventDisableTiming;

  // Constructors
  CUDAEvent(unsigned int flags = DEFAULT_FLAGS)
    : internals_(detail::CUDAEvent_create(flags)) {}

  ~CUDAEvent() { detail::CUDAEvent_uncheckedFree(internals_); }

  CUDAEvent(const CUDAEvent& other) {
    detail::CUDAEvent_retain(other.internals_);
    internals_ = other.internals_;
  }

  CUDAEvent(CUDAEvent&& other) {
    std::swap(internals_, other.internals_);
  }

  CUDAEvent& operator=(CUDAEvent other) noexcept {
    std::swap(internals_, other.internals_);
    return *this;
  }

  operator cudaEvent_t() const { return detail::CUDAEvent_event(internals_); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CUDAEvent& left, const CUDAEvent& right) {
    return left.internals_ < right.internals_;
  }

  int64_t device() const { return detail::CUDAEvent_device(internals_); }
  cudaEvent_t event() const { return detail::CUDAEvent_event(internals_); }
  CUDAEventInternals* internals() const { return internals_; }

  void record() const; // Record on the current stream
  void record(const CUDAStream& stream) const;

private:
  CUDAEventInternals* internals_;
};

} // namespace cuda
} // namespace at


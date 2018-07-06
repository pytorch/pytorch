#pragma once

#include <cstdint>
#include <utility>

/*
* A CUDA stream interface with no CUDA build dependency.
* 
* Includes the CUDAStream RAII class and a pointer-based stream API.
* 
* The ATen Context interface should be preferred when working with streams.
*/

// Forward-declares cudaStream_t to avoid depending on CUDA in CPU builds
// Note: this is the internal CUDA runtime typedef for cudaStream_t
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

// Forward-declares internals
struct CUDAStreamInternals;

namespace at {

namespace detail {

// Pointer-based API (for internal use)
// Note: ATen/Context is preferred to work with streams safely
CUDAStreamInternals* CUDAStream_getDefaultStreamOnDevice(int64_t device);
CUDAStreamInternals* CUDAStream_getDefaultStream();

CUDAStreamInternals* CUDAStream_createAndRetainWithOptions(int32_t flags, int32_t priority);

CUDAStreamInternals* CUDAStream_getAndRetainCurrentStreamOnDevice(int64_t device);
CUDAStreamInternals* CUDAStream_getAndRetainCurrentStream();

// Note: these Unsafe gets should NEVER be used and are only here for legacy
// purposes. Once those uses are gone they should be removed.
CUDAStreamInternals* CUDAStream_getCurrentStreamOnDeviceUnsafe(int64_t device);
CUDAStreamInternals* CUDAStream_getCurrentStreamUnsafe();

void CUDAStream_setStreamOnDevice(int64_t device, CUDAStreamInternals* internals);
void CUDAStream_setStream(CUDAStreamInternals* internals);

cudaStream_t CUDAStream_stream(CUDAStreamInternals*);
int64_t CUDAStream_device(CUDAStreamInternals*);

bool CUDAStream_retain(CUDAStreamInternals*);
void CUDAStream_free(CUDAStreamInternals*&);

} // namespace detail

// RAII for a CUDA stream
// Allows use as a cudaStream_t, copying, moving, and metadata access.
struct CUDAStream {
  // Constants
  static constexpr int32_t DEFAULT_FLAGS = 1; // = cudaStreamNonBlocking;
  static constexpr int32_t DEFAULT_PRIORITY = 0;

  // Constructors
  CUDAStream() = default;
  CUDAStream(CUDAStreamInternals* internals) : internals_{internals} { }
  
  // Destructor
  ~CUDAStream() { detail::CUDAStream_free(internals_); }

  // Copy constructor
  CUDAStream(const CUDAStream& other);

  // Move constructor
  CUDAStream(CUDAStream&& other);  

  // Assignment operator
  CUDAStream& operator=(CUDAStream other) {
    std::swap(internals_, other.internals_);
    return *this;
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

private:
  CUDAStreamInternals* internals_ = nullptr;
};

} // namespace at

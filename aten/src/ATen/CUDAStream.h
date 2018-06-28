#pragma once

#include <cstdint>
 
/*
* A CPU-friendly CUDAStream interface.
* 
* Includes the CUDAStream RAII class and and internal stream API.
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

// Pointer-based API (for internal use)
// Consumers of streams should generally use ATen/Context to work with them safely
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
int CUDAStream_device(CUDAStreamInternals*);

bool CUDAStream_retain(CUDAStreamInternals*);
void CUDAStream_free(CUDAStreamInternals*&);

CUDAStreamInternals* CUDAStream_copy(CUDAStreamInternals*);
void CUDAStream_move(CUDAStreamInternals*& src, CUDAStreamInternals*& dst);

// RAII for a cuda stream
// Allows copying, moving, use as a cudaStream_t, and access to relevant
// metadata. Holding a CUDAStream ensures the underlying stream stays live.
struct CUDAStream {
  // Constants
  static constexpr int32_t DEFAULT_FLAGS = 1; // = cudaStreamNonBlocking;
  static constexpr int32_t DEFAULT_PRIORITY = 0;

  // Constructors
  CUDAStream() = default;
  CUDAStream(CUDAStreamInternals* internals) : internals_{internals} { }
  
  // Destructor
  ~CUDAStream() { CUDAStream_free(internals_); }

  // Copy constructor and copy-assignment operator
  CUDAStream(const CUDAStream& other) {
    CUDAStream_free(internals_);
    internals_ = CUDAStream_copy(other.internals_);
  }
  CUDAStream& operator=(const CUDAStream& other) {
    CUDAStream_free(internals_);
    internals_ = CUDAStream_copy(other.internals_);
    return *this;
  }

  // Move constructor and move-assignment operator
  CUDAStream(CUDAStream&& other) {
    CUDAStream_move(other.internals_, internals_);
  }  
  CUDAStream& operator=(CUDAStream&& other) {
    CUDAStream_move(other.internals_, internals_);
    return *this;
  }

  // Implicit conversion to cudaStream_t
  operator cudaStream_t() const { return CUDAStream_stream(internals_); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CUDAStream& left, const CUDAStream& right) {
    return left.internals_ < right.internals_;
  }

  // Getters
  int device() const { return CUDAStream_device(internals_); }
  cudaStream_t stream() const { return CUDAStream_stream(internals_); }
  CUDAStreamInternals* internals() const { return internals_; }

private:
  CUDAStreamInternals* internals_ = nullptr;
};

} // namespace at
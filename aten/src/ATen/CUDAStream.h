#pragma once

#include <cstdint>

/*
* A CPU-friendly CUDAStream interface.
* 
* Includes the CUDAStream RAII class and and internal stream API.
* 
* The ATen Context and CUDAHooks interfaces should be preferred when working
* with streams.
*/

// Forward-declares cudaStream_t to avoid depending on CUDA in CPU builds
// Note: this is the internal CUDA runtime typedef for cudaStream_t
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace at {

// Forward-declares internals
struct CUDAStreamInternals;

// Pointer-based API (for internal use)
// Consumers of streams should generally use CUDAHooks to work with them safely
// Note: the Unsafe gets should NEVER be used and are only here for legacy
// purposes. Once those are gone they should be removed.
CUDAStreamInternals* CUDAStream_getDefaultStreamOnDevice(int64_t device);
CUDAStreamInternals* CUDAStream_getDefaultStream();
CUDAStreamInternals* CUDAStream_createAndRetainWithOptions(int32_t flags, int32_t priority);
CUDAStreamInternals* CUDAStream_getAndRetainCurrentStreamOnDevice(int64_t device);
CUDAStreamInternals* CUDAStream_getCurrentStreamOnDeviceUnsafe(int64_t device);
CUDAStreamInternals* CUDAStream_getAndRetainCurrentStream();
CUDAStreamInternals* CUDAStream_getCurrentStreamUnsafe();
void CUDAStream_setStreamOnDevice(int64_t device, CUDAStreamInternals* internals);
void CUDAStream_setStream(CUDAStreamInternals* internals);
cudaStream_t CUDAStream_stream(CUDAStreamInternals*);
int CUDAStream_device(CUDAStreamInternals*);
bool CUDAStream_retain(CUDAStreamInternals*);
void CUDAStream_free(CUDAStreamInternals*);

// RAII for a cuda stream
struct CUDAStream {
  // Constants
  static constexpr int32_t DEFAULT_FLAGS = 1; // = cudaStreamNonBlocking;
  static constexpr int32_t DEFAULT_PRIORITY = 0;

  // Constructors
  CUDAStream(CUDAStreamInternals* internals) : internals_{internals} { }
  
  // Destructor
  ~CUDAStream() { CUDAStream_free(internals_); }

  // Copy constructor and copy-assignment operator
  CUDAStream(const CUDAStream& other);
  CUDAStream& operator=(const CUDAStream& other);

  // Move constructor and move-assignment operator
  CUDAStream(CUDAStream&& other);
  CUDAStream& operator=(CUDAStream&& other);

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

  void copyInternal(const CUDAStream& other);
  void moveInternal(CUDAStream&& other);
};

} // namespace at
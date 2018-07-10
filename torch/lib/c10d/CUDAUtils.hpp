#pragma once

typedef struct CUDAStreamInternals THCStream;

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

namespace c10d {

// RAII wrapper for current CUDA device.
class CUDADevice {
 public:
  CUDADevice(int device);

  CUDADevice() {}

  ~CUDADevice();

  void setDevice(int device);

 protected:
  int originalDevice_ = -1;
};

// RAII wrapper for CUDA events.
class CUDAEvent {
 public:
  CUDAEvent(cudaEvent_t event) : event_(event) {}

  CUDAEvent() : CUDAEvent(nullptr) {}

  ~CUDAEvent();

  static CUDAEvent create(unsigned int flags = cudaEventDefault);

  // Must not be copyable.
  CUDAEvent& operator=(const CUDAEvent&) = delete;
  CUDAEvent(const CUDAEvent&) = delete;

  // Must be move constructable.
  CUDAEvent(CUDAEvent&& other) {
    std::swap(event_, other.event_);
  }

  // Must be move assignable.
  CUDAEvent& operator=(CUDAEvent&& other) {
    std::swap(event_, other.event_);
    return *this;
  }

  cudaEvent_t getEvent() const {
    return event_;
  }

 protected:
  cudaEvent_t event_;
};

// RAII wrapper for CUDA streams.
//
// This wrapper uses THCStream instead of cudaStream_t because we need
// to interact with the THC API for selecting the current stream.
// Doing this without having a THCStream pointer is cumbersome.
//
class CUDAStream {
 public:
  CUDAStream(THCStream* stream) : stream_(stream) {}

  CUDAStream() : CUDAStream(nullptr) {}

  ~CUDAStream();

  static CUDAStream create();

  // Must not be copyable.
  CUDAStream& operator=(const CUDAStream&) = delete;
  CUDAStream(const CUDAStream&) = delete;

  // Must be move constructable.
  CUDAStream(CUDAStream&& other) {
    std::swap(stream_, other.stream_);
  }

  // Must be move assignable.
  CUDAStream& operator=(CUDAStream&& other) {
    std::swap(stream_, other.stream_);
    return *this;
  }

  cudaStream_t getStream() const;

  THCStream* getTHCStream() {
    return stream_;
  }

 protected:
  THCStream* stream_;
};

} // namespace c10d

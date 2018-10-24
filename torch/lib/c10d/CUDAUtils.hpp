#pragma once

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

namespace c10d {

// RAII wrapper for CUDA events.
class CUDAEvent {
 public:
  CUDAEvent(cudaEvent_t event, int device) : device_(device), event_(event) {}

  CUDAEvent() : CUDAEvent(nullptr, 0) {}

  ~CUDAEvent() noexcept(false);

  static CUDAEvent create(unsigned int flags = cudaEventDefault);

  // Must not be copyable.
  CUDAEvent& operator=(const CUDAEvent&) = delete;
  CUDAEvent(const CUDAEvent&) = delete;

  // Must be move constructable.
  CUDAEvent(CUDAEvent&& other) {
    std::swap(event_, other.event_);
    std::swap(device_, other.device_);
  }

  // Must be move assignable.
  CUDAEvent& operator=(CUDAEvent&& other) {
    std::swap(event_, other.event_);
    std::swap(device_, other.device_);
    return *this;
  }

  cudaEvent_t getEvent() const {
    return event_;
  }

  int getDevice() const {
    return device_;
  }

 protected:
  int device_;
  cudaEvent_t event_;
};

} // namespace c10d

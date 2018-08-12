#pragma once

#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/Exceptions.h"
#include "ATen/DeviceGuard.h"
#include "ATen/Error.h"

#include "cuda_runtime_api.h"

#include <cstdint> 

namespace at { namespace cuda { 

/*
A CUDAEvent is an RAII for a cudaEvent. It provides easy to use mechanisms
for recording and blocking streams on its completion.

This class is intended for internal use, and not as a replacement for 
events in the Python API. 

CUDAEvent is NOT thread safe. 
*/
struct CUDAEvent {
  CUDAEvent() = default;

  // CUDAEvents are not copyable
  CUDAEvent& operator=(const CUDAEvent&) = delete;
  CUDAEvent(const CUDAEvent&) = delete;

  // CUDAEvents are movable. 
  CUDAEvent& operator=(CUDAEvent&& other) { 
    moveHelper(std::move(other)); 
    return *this;
  }
  CUDAEvent(CUDAEvent&& other) { moveHelper(std::move(other)); }

  void recordOnce(CUDAStream stream) {
    if (!is_created) {
      create(stream);
      AT_CUDA_CHECK(cudaEventRecord(event_, stream));
    }
  }

  void record(CUDAStream stream) {
    if (is_created) {
      AT_ASSERT(device_ == stream.device());
    } else {
      create(stream);
    }

    AT_CUDA_CHECK(cudaEventRecord(event_, stream));
  } 

  void block(CUDAStream stream) { 
    AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
  }

  int64_t device() { return device_; }

  bool happened() { return (cudaEventQuery(event_) == cudaSuccess); }

  ~CUDAEvent() { 
    if (is_created) {
      at::DeviceGuard device_guard{device_};
      cudaEventDestroy(event_); // Unchecked
    }
  }
  
private:
  bool is_created = false;
  cudaEvent_t event_;
  int64_t device_ = -1;

  void moveHelper(CUDAEvent&& other) {
    std::swap(is_created, other.is_created);
    std::swap(event_, other.event_);
    std::swap(device_, other.device_);
  }

  void create(CUDAStream stream) {
    at::DeviceGuard device_guard{stream.device()};
    AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));

    is_created = true;
    device_ = stream.device();
  }
};

} // namespace cuda
} // namespace at

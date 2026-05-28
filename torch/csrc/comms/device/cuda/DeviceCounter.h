// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <torch/csrc/comms/device/cuda/CudaApi.hpp>

namespace torch::comms {

// RAII wrapper around a uint64_t counter for GPU-side atomic increments
// with CPU-side reads.
//
// Uses mapped pinned host memory so reads are direct pointer dereferences
// with no CUDA API calls — safe to call from any thread.
//
// Construction is capture-safe: switches to relaxed capture mode internally.
// Destruction frees the underlying memory.
class DeviceCounter {
 public:
  static cudaError_t create(CudaApi* api, std::unique_ptr<DeviceCounter>& out);

  ~DeviceCounter();

  DeviceCounter(const DeviceCounter&) = delete;
  DeviceCounter& operator=(const DeviceCounter&) = delete;
  DeviceCounter(DeviceCounter&&) = delete;
  DeviceCounter& operator=(DeviceCounter&&) = delete;

  uint64_t read() const;
  cudaError_t increment(cudaStream_t stream, uint64_t amount = 1ULL);

  // Reset the counter to zero. CPU-side write to mapped pinned memory; no
  // CUDA API calls. Caller must ensure no in-flight GPU work atomic-adds
  // to this counter when this is called.
  void reset() {
    if (counter_) {
      *counter_ = 0;
    }
  }

  uint64_t* ptr() const {
    return counter_;
  }

 private:
  DeviceCounter(CudaApi* api, uint64_t* ptr) : api_(api), counter_(ptr) {}

  CudaApi* api_;
  uint64_t* counter_; // pinned host memory, accessible from both CPU and GPU
};

} // namespace torch::comms

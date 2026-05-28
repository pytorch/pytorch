// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/device/cuda/DeviceCounter.h>

#include <torch/csrc/comms/device/cuda/AtomicAddKernel.h>
#include <torch/csrc/comms/utils/CudaRAII.h>

namespace torch::comms {

cudaError_t DeviceCounter::create(
    CudaApi* api,
    std::unique_ptr<DeviceCounter>& out) {
  meta::comms::StreamCaptureModeGuard guard{api, cudaStreamCaptureModeRelaxed};

  void* host_alloc = nullptr;
  cudaError_t err =
      api->hostAlloc(&host_alloc, sizeof(uint64_t), cudaHostAllocDefault);

  if (err != cudaSuccess) {
    return err;
  }

  auto* ptr = static_cast<uint64_t*>(host_alloc);
  *ptr = 0;

  out = std::unique_ptr<DeviceCounter>(new DeviceCounter(api, ptr));

  return cudaSuccess;
}

DeviceCounter::~DeviceCounter() {
  if (counter_) {
    CUDA_CHECK_IGNORE(
        api_, api_->hostFree(counter_), "Failed to free host counter");
  }
}

uint64_t DeviceCounter::read() const {
  return *static_cast<volatile uint64_t*>(counter_);
}

cudaError_t DeviceCounter::increment(cudaStream_t stream, uint64_t amount) {
  return launchAtomicAdd(stream, counter_, amount);
}

} // namespace torch::comms

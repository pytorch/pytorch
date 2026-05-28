// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
namespace meta::comms {

// RAII helper for device buffer pointers
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size);
  ~DeviceBuffer();

  // delete copy constructor
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  // default move constructor

  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void* get() const;

 private:
  void* ptr_{nullptr};
  std::size_t size_{0};
};

// RAII helper for cuda stream
class CudaStream {
 public:
  CudaStream(unsigned int flags = cudaStreamDefault);
  ~CudaStream();

  // delete copy constructor
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  // default move constructor
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  cudaStream_t get() const;

 private:
  cudaStream_t stream_{nullptr};
};

// RAII helper for cuda event
class CudaEvent {
 public:
  CudaEvent();

  ~CudaEvent();

  // delete copy constructor
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  // custom move constructor due to raw pointers won't be automatically moved
  CudaEvent(CudaEvent&& other) noexcept;
  CudaEvent& operator=(CudaEvent&& other) noexcept;

  cudaEvent_t get() const;

 private:
  cudaEvent_t event_{nullptr};
};

// RAII guard that sets the calling thread's CUDA stream capture interaction
// mode and restores the previous mode on destruction.
//
// Two constructors:
//   1. Standalone — calls cudaThreadExchangeStreamCaptureMode directly.
//   2. Mockable — accepts any object with threadExchangeStreamCaptureMode(),
//      captured via template at construction time. No hard dependency on the
//      API type.
//
// Usage:
//   // standalone
//   StreamCaptureModeGuard guard{cudaStreamCaptureModeRelaxed};
//
//   // mockable (e.g. torchcomms CudaApi)
//   StreamCaptureModeGuard guard{api, cudaStreamCaptureModeRelaxed};
class StreamCaptureModeGuard {
 public:
  using ExchangeFn = cudaError_t (*)(void*, cudaStreamCaptureMode*);

  __attribute__((visibility("default"))) explicit StreamCaptureModeGuard(
      cudaStreamCaptureMode desiredMode);

  template <typename Api>
  StreamCaptureModeGuard(Api* api, cudaStreamCaptureMode desiredMode)
      : ctx_(api),
        exchangeFn_([](void* ctx, cudaStreamCaptureMode* mode) {
          return static_cast<Api*>(ctx)->threadExchangeStreamCaptureMode(mode);
        }),
        prevMode_(desiredMode) {
    init();
  }

  __attribute__((visibility("default"))) ~StreamCaptureModeGuard();

  StreamCaptureModeGuard(const StreamCaptureModeGuard&) = delete;
  StreamCaptureModeGuard& operator=(const StreamCaptureModeGuard&) = delete;
  StreamCaptureModeGuard(StreamCaptureModeGuard&&) = delete;
  StreamCaptureModeGuard& operator=(StreamCaptureModeGuard&&) = delete;

 private:
  __attribute__((visibility("default"))) void init();

  void* ctx_{nullptr};
  ExchangeFn exchangeFn_{nullptr};
  cudaStreamCaptureMode prevMode_;
};

} // namespace meta::comms

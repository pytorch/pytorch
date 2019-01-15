#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

namespace at { namespace cuda {

/*
* CUDAEvents are movable not copyable wrappers around CUDA's events.
*
* CUDAEvents are constructed lazily when first recorded unless it is
* reconstructed from a cudaIpcEventHandle_t. The event has a device, and this
* device is acquired from the first recording stream. However, if constructed
* from a handle or ipc_handle() is called before it is ever recorded, the device
* will be acquired from current stream. Later streams that record to the event
* must share this device, but streams on any device can query and wait on the
* event.
*/
struct AT_CUDA_API CUDAEvent {
  // Constants
  static constexpr unsigned int DEFAULT_FLAGS = cudaEventDisableTiming;

  // Constructors
  explicit CUDAEvent(unsigned int flags = DEFAULT_FLAGS)
  : flags_{flags} { }

  // Note: the original event and the reconstructed event now share recorded
  // activities. Users need to make sure the last recording event (either
  // original or reconstructed) must not be destructed when synchronize(),
  // query(), wait(), or block() is called. Otherwise, the behavior will be
  // undefined.
  explicit CUDAEvent(const cudaIpcEventHandle_t* handle) {
    #ifndef __HIP_PLATFORM_HCC__
      device_index_ = getCurrentCUDAStream().device_index();
      CUDAGuard guard(static_cast<int16_t>(device_index_));

      AT_CUDA_CHECK(cudaIpcOpenEventHandle(&event_, *handle));
      is_created_ = true;
    #else
      AT_ERROR("cuIpcOpenEventHandle with HIP is not supported");
    #endif
  }

  // Note: event destruction done on creating device to avoid creating a
  // CUDA context on other devices.
  ~CUDAEvent() {
    try {
      if (is_created_) {
        CUDAGuard device_guard(static_cast<int16_t>(device_index_));
        cudaEventDestroy(event_);
      }
    } catch (...) { /* No throw */ }
  }

  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  CUDAEvent(CUDAEvent&& other) { moveHelper(std::move(other)); }
  CUDAEvent& operator=(CUDAEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  operator cudaEvent_t() const { return event(); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CUDAEvent& left, const CUDAEvent& right) {
    return left.event_ < right.event_;
  }

  bool isCreated() const { return is_created_; }
  int64_t device() const { return device_index_; }
  cudaEvent_t event() const { return event_; }

  // Note: cudaEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }

    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
      return true;
    } else if (err != cudaErrorNotReady) {
      C10_CUDA_CHECK(err);
    }

    return false;
  }

  void record() { record(getCurrentCUDAStream()); }

  void recordOnce(const CUDAStream& stream) {
    if (!was_recorded_) record(stream);
  }

  // Note: cudaEventRecord must be called on the same device as the stream.
  void record(const CUDAStream& stream) {
    CUDAGuard guard(static_cast<int16_t>(stream.device_index()));

    if (is_created_) {
      AT_ASSERT(device_index_ == stream.device_index());
    } else {
      createEvent(stream.device_index());
    }

    AT_CUDA_CHECK(cudaEventRecord(event_, stream));
    was_recorded_ = true;
  }

  // Note: cudaStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual GPU resources associated with it.
  void block(const CUDAStream& stream) const {
    if (is_created_) {
      CUDAGuard guard(static_cast<int16_t>(stream.device_index()));
      AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
    }
  }

  float elapsed_time(const CUDAEvent& other) const {
    AT_CHECK(is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    // raise cudaErrorNotReady if either event is recorded but not yet completed
    AT_CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      CUDAGuard guard(static_cast<int16_t>(device_index_));
      AT_CUDA_CHECK(cudaEventSynchronize(event_));
    }
  }

  void ipc_handle(cudaIpcEventHandle_t * handle) {
    #ifndef __HIP_PLATFORM_HCC__
      if (!is_created_) {
        // this CUDAEvent object was initially constructed from flags but event_
        // is not created yet.
        createEvent(getCurrentCUDAStream().device_index());
      }
      CUDAGuard guard(static_cast<int16_t>(device_index_));
      AT_CUDA_CHECK(cudaIpcGetEventHandle(handle, event_));
    #else
      AT_ERROR("cuIpcGetEventHandle with HIP is not supported");
    #endif
  }

private:
  unsigned int flags_ = DEFAULT_FLAGS;
  bool is_created_ = false;
  bool was_recorded_ = false;
  int64_t device_index_ = -1;
  cudaEvent_t event_;

  void createEvent(const at::DeviceIndex deviceIndex) {
    CUDAGuard guard(static_cast<int16_t>(deviceIndex));
    AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_));
    is_created_ = true;
    device_index_ = deviceIndex;
  }

  void moveHelper(CUDAEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace cuda
} // namespace at

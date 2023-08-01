#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/impl/GPUTrace.h>
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
* device is acquired from the first recording stream. However, if reconstructed
* from a handle, the device should be explicitly specified; or if ipc_handle() is
* called before the event is ever recorded, it will use the current device.
* Later streams that record the event must match this device.
*/
struct TORCH_CUDA_CPP_API CUDAEvent {
  // Constructors
  // Default value for `flags` is specified below - it's cudaEventDisableTiming
  CUDAEvent() noexcept = default;
  CUDAEvent(unsigned int flags) noexcept : flags_{flags} {}

  CUDAEvent(
      DeviceIndex device_index, const cudaIpcEventHandle_t* handle) {
      device_index_ = device_index;
      CUDAGuard guard(device_index_);

      AT_CUDA_CHECK(cudaIpcOpenEventHandle(&event_, *handle));
      is_created_ = true;
  }

  // Note: event destruction done on creating device to avoid creating a
  // CUDA context on other devices.
  ~CUDAEvent() {
    try {
      if (is_created_) {
        CUDAGuard guard(device_index_);
        const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
        if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_deletion(reinterpret_cast<uintptr_t>(event_));
        }
        cudaEventDestroy(event_);
      }
    } catch (...) { /* No throw */ }
  }

  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  CUDAEvent(CUDAEvent&& other) noexcept { moveHelper(std::move(other)); }
  CUDAEvent& operator=(CUDAEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator cudaEvent_t() const { return event(); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CUDAEvent& left, const CUDAEvent& right) {
    return left.event_ < right.event_;
  }

  optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kCUDA, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return is_created_; }
  DeviceIndex device_index() const {return device_index_;}
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
    } else {
      // ignore and clear the error if not ready
      (void)cudaGetLastError();
    }

    return false;
  }

  void record() { record(getCurrentCUDAStream()); }

  void recordOnce(const CUDAStream& stream) {
    if (!was_recorded_) record(stream);
  }

  // Note: cudaEventRecord must be called on the same device as the event.
  void record(const CUDAStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
      " does not match recording stream's device ", stream.device_index(), ".");
    CUDAGuard guard(device_index_);
    AT_CUDA_CHECK(cudaEventRecord(event_, stream));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          reinterpret_cast<uintptr_t>(event_),
          reinterpret_cast<uintptr_t>(stream.stream())
      );
    }
    was_recorded_ = true;
  }

  // Note: cudaStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual GPU resources associated with it.
  void block(const CUDAStream& stream) {
    if (is_created_) {
      CUDAGuard guard(stream.device_index());
      AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(
            reinterpret_cast<uintptr_t>(event_),
            reinterpret_cast<uintptr_t>(stream.stream())
        );
      }
    }
  }

  // Note: cudaEventElapsedTime can be safely called from any device
  float elapsed_time(const CUDAEvent& other) const {
    TORCH_CHECK(is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    // raise cudaErrorNotReady if either event is recorded but not yet completed
    AT_CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  // Note: cudaEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_synchronization(reinterpret_cast<uintptr_t>(event_));
      }
      AT_CUDA_CHECK(cudaEventSynchronize(event_));
    }
  }

  // Note: cudaIpcGetEventHandle must be called on the same device as the event
  void ipc_handle(cudaIpcEventHandle_t * handle) {
      if (!is_created_) {
        // this CUDAEvent object was initially constructed from flags but event_
        // is not created yet.
        createEvent(getCurrentCUDAStream().device_index());
      }
      CUDAGuard guard(device_index_);
      AT_CUDA_CHECK(cudaIpcGetEventHandle(handle, event_));
  }

private:
  unsigned int flags_ = cudaEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
  DeviceIndex device_index_ = -1;
  cudaEvent_t event_{};

  void createEvent(DeviceIndex device_index) {
    device_index_ = device_index;
    CUDAGuard guard(device_index_);
    AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(reinterpret_cast<uintptr_t>(event_));
    }
    is_created_ = true;
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

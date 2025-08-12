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

/*
* `cudaEventExternal` is a torch-specific flag that is used to
* indicate that the CUDAEvent will be used only for synchronization
* with work outside of the cuda graph, rather than creation of
* cross-stream dependencies within a cuda graph. Resources:
* https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#cross-stream-dependencies-and-events
* https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3457b81d1d32c6a00f6132fbc2693d47
* https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0c23426b7252eaa9cef695859991304e
*/
#define cudaEventExternal 0x08

namespace at::cuda {

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
      DeviceIndex device_index, const cudaIpcEventHandle_t* handle) : device_index_(device_index) {
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
          (*interp)->trace_gpu_event_deletion(at::kCUDA, reinterpret_cast<uintptr_t>(event_));
        }
        AT_CUDA_CHECK(cudaEventDestroy(event_));
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

  std::optional<at::Device> device() const {
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

#ifndef USE_ROCM
    // it is an error to use cudaEventRecordExternal when not doing stream capture
    unsigned int flags = (c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None && external_) ? cudaEventRecordExternal : cudaEventRecordDefault;
    AT_CUDA_CHECK(cudaEventRecordWithFlags(event_, stream, flags));
#else
    AT_CUDA_CHECK(cudaEventRecord(event_, stream));
#endif
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(at::kCUDA,
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
#ifndef USE_ROCM
      // it is an error to use cudaEventWaitExternal when not doing stream capture
      unsigned int flags = (c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None && external_) ? cudaEventWaitExternal : cudaEventWaitDefault;
      AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, flags));
#else
      AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_));
#endif
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(at::kCUDA,
            reinterpret_cast<uintptr_t>(event_),
            reinterpret_cast<uintptr_t>(stream.stream())
        );
      }
    }
  }

  // Note: cudaEventElapsedTime can be safely called from any device
  float elapsed_time(const CUDAEvent& other) const {
    TORCH_CHECK_VALUE(
        !(flags_ & cudaEventDisableTiming) && !(other.flags_ & cudaEventDisableTiming),
        "Both events must be created with argument 'enable_timing=True'.");
    TORCH_CHECK_VALUE(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");

    float time_ms = 0;
    // We do not strictly have to set the device index to the same as our event,
    // but if we don't and the current device is not initialized, it will
    // create a new cuda context, which will consume a lot of memory.
    CUDAGuard guard(device_index_);
    // raise cudaErrorNotReady if either event is recorded but not yet completed
    AT_CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  // Note: cudaEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_synchronization(at::kCUDA, reinterpret_cast<uintptr_t>(event_));
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
  bool external_ = false;
  DeviceIndex device_index_ = -1;
  cudaEvent_t event_{};

  void createEvent(DeviceIndex device_index) {
    external_ = (flags_ & cudaEventExternal) != 0;
#ifdef USE_ROCM
    TORCH_CHECK(!external_, "External events are disallowed in rocm");
#endif
    flags_ &= ~cudaEventExternal;
    device_index_ = device_index;
    CUDAGuard guard(device_index_);
    AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(at::kCUDA, reinterpret_cast<uintptr_t>(event_));
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

} // namespace at::cuda

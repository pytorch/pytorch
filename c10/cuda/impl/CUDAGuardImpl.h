#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/PyInterpreter.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <optional>

namespace c10::cuda::impl {

struct CUDAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::CUDA;

  CUDAGuardImpl() = default;
  explicit CUDAGuardImpl(DeviceType t) {
    TORCH_CHECK(
        t == DeviceType::CUDA,
        "CUDAGuardImpl initialized with non-CUDA DeviceType: ",
        t);
  }
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_CHECK(d.is_cuda(), "Expected a CUDA device, but got ", d);
    auto old_device_index = c10::cuda::ExchangeDevice(d.index());
    return Device(DeviceType::CUDA, old_device_index);
  }
  Device getDevice() const override {
    DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    return Device(DeviceType::CUDA, device);
  }
  std::optional<Device> uncheckedGetDevice() const noexcept {
    DeviceIndex device{-1};
    const auto err = C10_CUDA_ERROR_HANDLED(c10::cuda::GetDevice(&device));
    C10_CUDA_CHECK_WARN(err);
    if (err != cudaSuccess) {
      return std::nullopt;
    }
    return Device(DeviceType::CUDA, device);
  }
  void setDevice(Device d) const override {
    TORCH_CHECK(d.is_cuda(), "Expected a CUDA device, but got ", d);
    C10_CUDA_CHECK(c10::cuda::SetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    C10_CUDA_CHECK_WARN(c10::cuda::MaybeSetDevice(d.index()));
  }
  Stream getStream(Device d) const override {
    return getCurrentCUDAStream(d.index()).unwrap();
  }
  Stream getDefaultStream(Device d) const override {
    return getDefaultCUDAStream(d.index());
  }
  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const override {
    CUDAStream cs(s);
    auto old_stream = getCurrentCUDAStream(s.device().index());
    setCurrentCUDAStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  // Event-related functions
  void createEvent(cudaEvent_t* cuda_event, const EventFlag flag) const {
    // Maps PyTorch's Event::Flag to CUDA flag
    auto cuda_flag = cudaEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        cuda_flag = cudaEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
        cuda_flag = cudaEventDefault;
        break;
      default:
        TORCH_CHECK(false, "CUDA event received unknown flag");
    }

    C10_CUDA_CHECK(cudaEventCreateWithFlags(cuda_event, cuda_flag));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
  }

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto cuda_event = static_cast<cudaEvent_t>(event);
    DeviceIndex orig_device{-1};
    C10_CUDA_CHECK_WARN(c10::cuda::GetDevice(&orig_device));
    C10_CUDA_CHECK_WARN(c10::cuda::SetDevice(device_index));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_deletion(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
    C10_CUDA_CHECK_WARN(cudaEventDestroy(cuda_event));
    C10_CUDA_CHECK_WARN(c10::cuda::SetDevice(orig_device));
  }

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(*event);
    CUDAStream cuda_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!cuda_event)
      createEvent(&cuda_event, flag);
    C10_CUDA_CHECK(cudaEventRecord(cuda_event, cuda_stream));
    // Makes the void* point to the (possibly just allocated) CUDA event
    *event = cuda_event;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(cuda_event),
          reinterpret_cast<uintptr_t>(cuda_stream.stream()));
    }

    // Resets device
    setDevice(orig_device);
  }

  void block(void* event, const Stream& stream) const override {
    if (!event)
      return;
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    CUDAStream cuda_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    C10_CUDA_CHECK(cudaStreamWaitEvent(
        cuda_stream,
        cuda_event,
        /*flags (must be zero)=*/0));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_wait(
          c10::kCUDA,
          reinterpret_cast<uintptr_t>(cuda_event),
          reinterpret_cast<uintptr_t>(cuda_stream.stream()));
    }
    setDevice(orig_device);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    // Note: cudaEventQuery can be safely called from any device
    const cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(cuda_event));
    if (err != cudaErrorNotReady) {
      C10_CUDA_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)cudaGetLastError();
    }
    return (err == cudaSuccess);
  }

  // Stream-related functions
  bool queryStream(const Stream& stream) const override {
    CUDAStream cuda_stream{stream};
    return cuda_stream.query();
  }

  void synchronizeStream(const Stream& stream) const override {
    CUDAStream cuda_stream{stream};
    cuda_stream.synchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_synchronization(
          c10::kCUDA, reinterpret_cast<uintptr_t>(cuda_event));
    }
    // Note: cudaEventSynchronize can be safely called from any device
    C10_CUDA_CHECK(cudaEventSynchronize(cuda_event));
  }

  // Note: synchronizeDevice can be safely called from any device
  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    DeviceIndex orig_device{-1};
    C10_CUDA_CHECK(c10::cuda::GetDevice(&orig_device));
    C10_CUDA_CHECK(c10::cuda::SetDevice(device_index));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_device_synchronization(c10::kCUDA);
    }
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    C10_CUDA_CHECK(c10::cuda::SetDevice(orig_device));
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    CUDAStream cuda_stream{stream};
    CUDACachingAllocator::recordStream(data_ptr, cuda_stream);
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    // Even though cudaEventElapsedTime can be safely called from any device, if
    // the current device is not initialized, it will create a new cuda context,
    // which will consume a lot of memory.
    DeviceIndex orig_device{-1};
    C10_CUDA_CHECK(c10::cuda::GetDevice(&orig_device));
    C10_CUDA_CHECK(c10::cuda::SetDevice(device_index));
    cudaEvent_t cuda_event1 = static_cast<cudaEvent_t>(event1);
    cudaEvent_t cuda_event2 = static_cast<cudaEvent_t>(event2);
    float time_ms = 0;
    // raise cudaErrorNotReady if either event is recorded but not yet completed
    C10_CUDA_CHECK(cudaEventElapsedTime(&time_ms, cuda_event1, cuda_event2));
    C10_CUDA_CHECK(c10::cuda::SetDevice(orig_device));
    return static_cast<double>(time_ms);
  }
};

} // namespace c10::cuda::impl

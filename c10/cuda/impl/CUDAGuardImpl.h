#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>

#include <cuda_runtime_api.h>

namespace c10 {
namespace cuda {
namespace impl {

struct CUDAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::CUDA;

  CUDAGuardImpl() {}
  explicit CUDAGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::CUDA);
  }
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_CUDA_CHECK(cudaSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    return Device(DeviceType::CUDA, device);
  }
  c10::optional<Device> uncheckedGetDevice() const noexcept {
    int device;
    auto err = cudaGetDevice(&device);
    C10_CUDA_CHECK_WARN(err);
    if (err != cudaSuccess) {
      return c10::nullopt;
    }
    return Device(DeviceType::CUDA, device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    Device current_device = getDevice();
    if (current_device != d) {
      C10_CUDA_CHECK(cudaSetDevice(d.index()));
    }
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    auto current_device = uncheckedGetDevice();
    if (!current_device.has_value() || current_device.value() != d) {
      C10_CUDA_CHECK_WARN(cudaSetDevice(d.index()));
    }
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentCUDAStream(d.index()).unwrap();
  }
  Stream getDefaultStream(Device d) const override {
    return getDefaultCUDAStream(d.index());
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    CUDAStream cs(s);
    auto old_stream = getCurrentCUDAStream(s.device().index());
    setCurrentCUDAStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  // Event-related functions
  void createEvent(
    cudaEvent_t* cuda_event,
    const EventFlag flag) const {
    // Maps PyTorch's Event::Flag to CUDA flag
    auto cuda_flag = cudaEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
      case EventFlag::CUDA_EVENT_DISABLE_TIMING:
        cuda_flag = cudaEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
      case EventFlag::CUDA_EVENT_DEFAULT:
        cuda_flag = cudaEventDefault;
        break;
      default:
        TORCH_CHECK(false, "CUDA event received unknown flag");
    }

    C10_CUDA_CHECK(cudaEventCreateWithFlags(cuda_event, cuda_flag));
  }

  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override {
    if (!event) return;
    auto cuda_event = static_cast<cudaEvent_t>(event);
    int orig_device;
    C10_CUDA_CHECK_WARN(cudaGetDevice(&orig_device));
    C10_CUDA_CHECK_WARN(cudaSetDevice(device_index));
    C10_CUDA_CHECK_WARN(cudaEventDestroy(cuda_event));
    C10_CUDA_CHECK_WARN(cudaSetDevice(orig_device));
  }

  void record(
    void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
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
    if (!cuda_event) createEvent(&cuda_event, flag);
    C10_CUDA_CHECK(cudaEventRecord(cuda_event, cuda_stream));
    // Makes the void* point to the (possibly just allocated) CUDA event
    *event = cuda_event;

    // Resets device
    setDevice(orig_device);
  }

  void block(
    void* event,
    const Stream& stream) const override {
    if (!event) return;
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    CUDAStream cuda_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    C10_CUDA_CHECK(cudaStreamWaitEvent(
      cuda_stream,
      cuda_event,
      /*flags (must be zero)=*/ 0));
    setDevice(orig_device);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event) return true;
    cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
    const cudaError_t err = cudaEventQuery(cuda_event);
    if (err != cudaErrorNotReady) {
      C10_CUDA_CHECK(err);
    }
    return (err == cudaSuccess);
  }
};

}}} // namespace c10::cuda::impl

#pragma once

#include <ATen/hip/HIPConfig.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// Note [Masquerading as CUDA]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// How it was before caffe2 was removed from public repos. hipify v1.
// ==================================================================
//
// c10_hip is very easy to understand: it is HIPified from c10_cuda,
// and anywhere you said CUDA, the source code now says HIP.  HIPified
// PyTorch is much harder to understand: it is HIPified from regular
// PyTorch, yes, but NO source-to-source translation from CUDA to
// HIP occurs; instead, anywhere we see "CUDA", it actually means "HIP".
// For example, when you use HIPified PyTorch, you say x.cuda() to
// move a tensor onto ROCm device.  We call this situation "HIP
// masquerading as CUDA".
//
// This leads to a very awkward situation when we want to call c10_hip
// code from PyTorch, since c10_hip is expecting things to be called
// HIP, but PyTorch is calling them CUDA (masquerading as HIP).  To
// fix this impedance mismatch, we have MasqueradingAsCUDA variants
// for all c10_hip classes.  These translate between the "HIP" and "CUDA
// masquerading as HIP" worlds.  For example,
// HIPGuardImplMasqueradingAsCUDA (this file) provides something like a
// HIPGuardImpl, but it reports its DeviceType as CUDA (e.g., type()
// returns CUDA, getDevice() reports the current HIP device as a CUDA
// device.)
//
// We should be able to delete all of these classes entirely once
// we switch PyTorch to calling a HIP a HIP.
//
// When you add a new MasqueradingAsCUDA class/function, you need to
// also update the rewrite rules in torch/utils/hipify/cuda_to_hip_mappings.py
//
// By the way, note that the cpp file associated with this also
// *overwrites* the entry in the DeviceGuardImpl registry for CUDA with
// this HIP implementation.
//
// How it is now. caffe2 is removed from public repos. hipify v2.
// ==============================================================
//
// c10_hip is very easy to understand: it is HIPified from c10_cuda,
// and anywhere you used a CUDA API, the source now calls a HIP API.
// Classes and namespaces are not renamed from CUDA to HIP et al.
// Filenames do get renamed from CUDA to HIP. This is the same as how PyTorch
// sources are hipified. It is simpler, better.
//
// However, this leads to a challenge that many downstream projects explicitly
// use these v1 Masquerading headers, classes, and symbols. For the purpose of
// backwards-compatible transitions, we maintain these Masquerading
// implementations but they no longer coerce a HIP device to a CUDA device. New
// code should not use Masquerading implementations but instead use the regular
// CUDA classes, for example the CUDAStream class inside c10/hip/HIPStream.h.
//

<<<<<<< HEAD
struct HIPGuardMasqueradingAsCUDA final : public c10::cuda::CUDAGuard {
  using c10::cuda::CUDAGuard::CUDAGuard;
=======
struct HIPGuardImplMasqueradingAsCUDA final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::CUDA;
  HIPGuardImplMasqueradingAsCUDA() {}
  HIPGuardImplMasqueradingAsCUDA(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::CUDA);
  }
  c10::DeviceType type() const override {
    return c10::DeviceType::CUDA;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_HIP_CHECK(hipSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    int device;
    C10_HIP_CHECK(hipGetDevice(&device));
    return Device(c10::DeviceType::CUDA, device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    C10_HIP_CHECK(hipSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    C10_HIP_CHECK_WARN(hipSetDevice(d.index()));
  }
  Stream getStream(Device d) const override {
    return getCurrentHIPStreamMasqueradingAsCUDA(d.index()).unwrap();
  }
  Stream getDefaultStream(Device d) const override {
    return getDefaultHIPStreamMasqueradingAsCUDA(d.index());
  }
  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPoolMasqueradingAsCUDA(priority, d.index());
  }
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false) const override {
    return getStreamFromPoolMasqueradingAsCUDA(isHighPriority, d.index());
  }
  Stream exchangeStream(Stream s) const override {
    HIPStreamMasqueradingAsCUDA cs(s);
    auto old_stream = getCurrentHIPStreamMasqueradingAsCUDA(s.device().index());
    setCurrentHIPStreamMasqueradingAsCUDA(cs);
    return old_stream.unwrap();
  }
  void* getStreamNativeHandle(const Stream s) const override {
    HIPStreamMasqueradingAsCUDA stream{s};
    return reinterpret_cast<void*>(stream.stream());
  }
  DeviceIndex deviceCount() const noexcept override {
    int deviceCnt;
    hipError_t _err;
    _err = hipGetDeviceCount(&deviceCnt);
    if(_err != hipErrorNoDevice && _err != hipSuccess)
        C10_HIP_CHECK(_err);
    return deviceCnt;
  }

  // Event-related functions
  // Note: hipEventCreateWithFlags should be called on the same device as
  //  the recording stream's device.
  void createEvent(
    hipEvent_t* hip_event,
    const EventFlag flag) const {
    // Maps PyTorch's Event::Flag to HIP flag
    auto hip_flag = hipEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        hip_flag = hipEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
        hip_flag = hipEventDefault;
        break;
      default:
        TORCH_CHECK(false, "HIP event received unknown flag");
    }

    C10_HIP_CHECK(hipEventCreateWithFlags(hip_event, hip_flag));
  }

  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override {
    if (!event) return;
    auto hip_event = static_cast<hipEvent_t>(event);
    int orig_device;
    C10_HIP_CHECK_WARN(hipGetDevice(&orig_device));
    C10_HIP_CHECK_WARN(hipSetDevice(device_index));
    C10_HIP_CHECK_WARN(hipEventDestroy(hip_event));
    C10_HIP_CHECK_WARN(hipSetDevice(orig_device));
  }

  void record(void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
      "Event device index ",
      device_index,
      " does not match recording stream's device index ",
      stream.device_index(),
      ".");

    hipEvent_t hip_event = static_cast<hipEvent_t>(*event);
    HIPStreamMasqueradingAsCUDA hip_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!hip_event) createEvent(&hip_event, flag);
    C10_HIP_CHECK(hipEventRecord(hip_event, hip_stream));
    // Makes the void* point to the (possibly just allocated) HIP event
    *event = hip_event;

    // Resets device
    setDevice(orig_device);
  }

  void block(
    void* event,
    const Stream& stream) const override {
    if (!event) return;
    hipEvent_t hip_event = static_cast<hipEvent_t>(event);
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    C10_HIP_CHECK(hipStreamWaitEvent(
      hip_stream,
      hip_event,
      /*flags (must be zero)=*/ 0));
    setDevice(orig_device);
  }

  bool queryEvent(void* event) const override {
    if (!event) return true;
    hipEvent_t hip_event = static_cast<hipEvent_t>(event);
    const hipError_t err = hipEventQuery(hip_event);
    if (err != hipErrorNotReady) C10_HIP_CHECK(err);
    else {
      // ignore and clear the error if not ready
      (void)hipGetLastError();
    }
    return (err == hipSuccess);
  }

  // Stream-related functions
  bool queryStream(const Stream& stream) const override {
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    return hip_stream.query();
  }

  void synchronizeStream(const Stream& stream) const override {
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    hip_stream.synchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    hipEvent_t hip_event = static_cast<hipEvent_t>(event);
    C10_HIP_CHECK(hipEventSynchronize(hip_event));
  }

  // Note: synchronizeDevice can be safely called from any device
  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    int orig_device{-1};
    C10_HIP_CHECK(hipGetDevice(&orig_device));
    C10_HIP_CHECK(hipSetDevice(device_index));
    C10_HIP_CHECK(hipDeviceSynchronize());
    C10_HIP_CHECK(hipSetDevice(orig_device));
  }

  void recordDataPtrOnStream(
    const c10::DataPtr& data_ptr,
    const Stream& stream) const override {
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA(data_ptr, hip_stream);
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    int orig_device;
    C10_HIP_CHECK(hipGetDevice(&orig_device));
    C10_HIP_CHECK(hipSetDevice(device_index));
    hipEvent_t hip_event1 = static_cast<hipEvent_t>(event1);
    hipEvent_t hip_event2 = static_cast<hipEvent_t>(event2);
    float time_ms = 0;
    // raise hipErrorNotReady if either event is recorded but not yet completed
    C10_HIP_CHECK(hipEventElapsedTime(&time_ms, hip_event1, hip_event2));
    C10_HIP_CHECK(hipSetDevice(orig_device));
    return static_cast<double>(time_ms);
  }
>>>>>>> 925f75e2d93 (Add a unified method data_ptr to c10::Stream and torch.Stream)
};

struct OptionalHIPGuardMasqueradingAsCUDA final : public c10::cuda::OptionalCUDAGuard {
  using c10::cuda::OptionalCUDAGuard::OptionalCUDAGuard;
};

struct HIPStreamGuardMasqueradingAsCUDA final : public c10::cuda::CUDAStreamGuard {
  using c10::cuda::CUDAStreamGuard::CUDAStreamGuard;
};

struct OptionalHIPStreamGuardMasqueradingAsCUDA final : public c10::cuda::OptionalCUDAStreamGuard {
  using c10::cuda::OptionalCUDAStreamGuard::OptionalCUDAStreamGuard;
};

struct HIPMultiStreamGuardMasqueradingAsCUDA final : public c10::cuda::CUDAMultiStreamGuard {
  using c10::cuda::CUDAMultiStreamGuard::CUDAMultiStreamGuard;
};

}} // namespace c10::hip

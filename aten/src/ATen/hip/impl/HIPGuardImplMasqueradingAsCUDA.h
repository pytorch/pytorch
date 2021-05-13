#pragma once

#include <ATen/hip/HIPConfig.h>

// The includes of HIPGuard.h
#include <c10/hip/impl/HIPGuardImpl.h>
#include <c10/hip/HIPMacros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/util/Exception.h>

#include <c10/hip/impl/HIPGuardImpl.h>

#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// Note [Masquerading as CUDA]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
//
//
// By the way, note that the cpp file associated with this also
// *overwrites* the entry in the DeviceGuardImpl registry for CUDA with
// this HIP implementation.

struct HIPGuardImplMasqueradingAsCUDA final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::CUDA;
  HIPGuardImplMasqueradingAsCUDA() {}
  HIPGuardImplMasqueradingAsCUDA(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::CUDA);
  }
  DeviceType type() const override {
    return DeviceType::CUDA;
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
    return Device(DeviceType::CUDA, device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_cuda());
    C10_HIP_CHECK(hipSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    C10_HIP_CHECK_WARN(hipSetDevice(d.index()));
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentHIPStreamMasqueradingAsCUDA(d.index()).unwrap();
  }
  Stream getDefaultStream(Device d) const override {
    return getDefaultHIPStreamMasqueradingAsCUDA(d.index());
  }
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false) const override {
    return getStreamFromPoolMasqueradingAsCUDA(isHighPriority, d.index());
  }
  Stream exchangeStream(Stream s) const noexcept override {
    HIPStreamMasqueradingAsCUDA cs(s);
    auto old_stream = getCurrentHIPStreamMasqueradingAsCUDA(s.device().index());
    setCurrentHIPStreamMasqueradingAsCUDA(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const noexcept override {
    int deviceCnt;
    C10_HIP_CHECK(hipGetDeviceCount(&deviceCnt));
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
      case EventFlag::HIP_EVENT_DISABLE_TIMING:
        hip_flag = hipEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
      case EventFlag::HIP_EVENT_DEFAULT:
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
    return (err == hipSuccess);
  }

  void recordDataPtrOnStream(
    const c10::DataPtr& data_ptr,
    const Stream& stream) const override {
    HIPStreamMasqueradingAsCUDA hip_stream{stream};
    HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA(data_ptr, hip_stream);
  }
};

// All of the guards which have HIPGuardImpl burned in need to also have
// variants using HIPGuardImplMasqueradingAsCUDA.

/// This code is all a direct copy from c10/cuda/HIPGuardMasqueradingAsCUDA.h, but with
/// the correct InlineDeviceGuard burned in.  Sorry about the
/// copy-pasting.

struct HIPGuardMasqueradingAsCUDA {
  explicit HIPGuardMasqueradingAsCUDA() = delete;
  explicit HIPGuardMasqueradingAsCUDA(DeviceIndex device_index) : guard_(device_index) {}
  explicit HIPGuardMasqueradingAsCUDA(Device device) : guard_(device) {}

  HIPGuardMasqueradingAsCUDA(const HIPGuardMasqueradingAsCUDA&) = delete;
  HIPGuardMasqueradingAsCUDA& operator=(const HIPGuardMasqueradingAsCUDA&) = delete;
  HIPGuardMasqueradingAsCUDA(HIPGuardMasqueradingAsCUDA&& other) = delete;
  HIPGuardMasqueradingAsCUDA& operator=(HIPGuardMasqueradingAsCUDA&& other) = delete;

  void set_device(Device device) { guard_.set_device(device); }
  void reset_device(Device device) { guard_.reset_device(device); }
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }
  Device original_device() const { return guard_.original_device(); }
  Device current_device() const { return guard_.current_device(); }

 private:
  c10::impl::InlineDeviceGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct OptionalHIPGuardMasqueradingAsCUDA {
  explicit OptionalHIPGuardMasqueradingAsCUDA() : guard_() {}
  explicit OptionalHIPGuardMasqueradingAsCUDA(optional<Device> device_opt) : guard_(device_opt) {}
  explicit OptionalHIPGuardMasqueradingAsCUDA(optional<DeviceIndex> device_index_opt) : guard_(device_index_opt) {}

  OptionalHIPGuardMasqueradingAsCUDA(const OptionalHIPGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPGuardMasqueradingAsCUDA& operator=(const OptionalHIPGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPGuardMasqueradingAsCUDA(OptionalHIPGuardMasqueradingAsCUDA&& other) = delete;
  OptionalHIPGuardMasqueradingAsCUDA& operator=(OptionalHIPGuardMasqueradingAsCUDA&& other) = delete;

  void set_device(Device device) { guard_.set_device(device); }
  void reset_device(Device device) { guard_.reset_device(device); }
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }
  optional<Device> original_device() const { return guard_.original_device(); }
  optional<Device> current_device() const { return guard_.current_device(); }
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalDeviceGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct HIPStreamGuardMasqueradingAsCUDA {
  explicit HIPStreamGuardMasqueradingAsCUDA() = delete;
  explicit HIPStreamGuardMasqueradingAsCUDA(Stream stream) : guard_(stream) {}
  HIPStreamGuardMasqueradingAsCUDA(const HIPStreamGuardMasqueradingAsCUDA&) = delete;
  HIPStreamGuardMasqueradingAsCUDA& operator=(const HIPStreamGuardMasqueradingAsCUDA&) = delete;
  HIPStreamGuardMasqueradingAsCUDA(HIPStreamGuardMasqueradingAsCUDA&& other) = delete;
  HIPStreamGuardMasqueradingAsCUDA& operator=(HIPStreamGuardMasqueradingAsCUDA&& other) = delete;

  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  HIPStreamMasqueradingAsCUDA original_stream() const {
    return HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, guard_.original_stream());
  }
  HIPStreamMasqueradingAsCUDA current_stream() const {
    return HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, guard_.current_stream());
  }

  Device current_device() const { return guard_.current_device(); }
  Device original_device() const { return guard_.original_device(); }

private:
  c10::impl::InlineStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct OptionalHIPStreamGuardMasqueradingAsCUDA {
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA() : guard_() {}
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA(Stream stream) : guard_(stream) {}
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA(optional<Stream> stream_opt) : guard_(stream_opt) {}

  OptionalHIPStreamGuardMasqueradingAsCUDA(const OptionalHIPStreamGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPStreamGuardMasqueradingAsCUDA& operator=(const OptionalHIPStreamGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPStreamGuardMasqueradingAsCUDA(OptionalHIPStreamGuardMasqueradingAsCUDA&& other) = delete;
  OptionalHIPStreamGuardMasqueradingAsCUDA& operator=(OptionalHIPStreamGuardMasqueradingAsCUDA&& other) = delete;

  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  optional<HIPStreamMasqueradingAsCUDA> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  optional<HIPStreamMasqueradingAsCUDA> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct HIPMultiStreamGuardMasqueradingAsCUDA {
  explicit HIPMultiStreamGuardMasqueradingAsCUDA(ArrayRef<HIPStreamMasqueradingAsCUDA> streams)
    : guard_(unwrapStreams(streams)) {}

  HIPMultiStreamGuardMasqueradingAsCUDA(const HIPMultiStreamGuardMasqueradingAsCUDA&) = delete;
  HIPMultiStreamGuardMasqueradingAsCUDA& operator=(const HIPMultiStreamGuardMasqueradingAsCUDA&) = delete;
  HIPMultiStreamGuardMasqueradingAsCUDA(HIPMultiStreamGuardMasqueradingAsCUDA&& other) = delete;
  HIPMultiStreamGuardMasqueradingAsCUDA& operator=(HIPMultiStreamGuardMasqueradingAsCUDA&& other) = delete;

private:
  c10::impl::InlineMultiStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;

  static std::vector<Stream> unwrapStreams(ArrayRef<HIPStreamMasqueradingAsCUDA> hipStreams) {
    std::vector<Stream> streams;
    streams.reserve(hipStreams.size());
    for (const HIPStreamMasqueradingAsCUDA& hipStream : hipStreams) {
      streams.push_back(hipStream);
    }
    return streams;
  }
};

}} // namespace c10::hip

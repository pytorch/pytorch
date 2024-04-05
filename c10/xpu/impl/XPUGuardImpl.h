#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <vector>

namespace c10::xpu::impl {

struct XPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = kXPU;

  XPUGuardImpl() = default;

  explicit XPUGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == kXPU);
  }

  DeviceType type() const override {
    return kXPU;
  }

  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_xpu());
    const auto old_device_index = c10::xpu::exchange_device(d.index());
    return Device(kXPU, old_device_index);
  }

  Device getDevice() const override {
    const auto device = c10::xpu::current_device();
    return Device(kXPU, device);
  }

  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_xpu());
    c10::xpu::set_device(d.index());
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    c10::xpu::set_device(d.index());
  }

  Stream getStream(Device d) const noexcept override {
    return getCurrentXPUStream(d.index()).unwrap();
  }

  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }

  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    const XPUStream stream(s);
    const auto old_stream = getCurrentXPUStream(s.device().index());
    setCurrentXPUStream(stream);
    return old_stream.unwrap();
  }

  DeviceIndex deviceCount() const noexcept override {
    return c10::xpu::device_count();
  }

  // Event-related functions
  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {}

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

    auto* xpu_event = reinterpret_cast<sycl::event*>(*event);
    const XPUStream xpu_stream{stream};
    *xpu_event = xpu_stream.queue().ext_oneapi_submit_barrier();
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::kXPU,
          reinterpret_cast<uintptr_t>(xpu_event),
          reinterpret_cast<uintptr_t>(&xpu_stream.queue()));
    }
  }

  void block(void* event, const Stream& stream) const override {
    if (!event)
      return;
    auto* xpu_event = reinterpret_cast<sycl::event*>(event);
    std::vector<sycl::event> event_list{*xpu_event};
    const XPUStream xpu_stream(stream);
    xpu_stream.queue().ext_oneapi_submit_barrier(event_list);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_wait(
          c10::kXPU,
          reinterpret_cast<uintptr_t>(xpu_event),
          reinterpret_cast<uintptr_t>(&xpu_stream.queue()));
    }
  }

  bool queryEvent(void* event) const override {
    using namespace sycl::info;
    if (!event)
      return true;
    auto* xpu_event = reinterpret_cast<sycl::event*>(event);
    return xpu_event->get_info<event::command_execution_status>() ==
        event_command_status::complete;
  }

  // Stream-related functions
  bool queryStream(const Stream& stream) const override {
    const XPUStream xpu_stream{stream};
    return xpu_stream.query();
  }

  void synchronizeStream(const Stream& stream) const override {
    const XPUStream xpu_stream{stream};
    xpu_stream.synchronize();
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    const XPUStream xpu_stream{stream};
    XPUCachingAllocator::recordStream(data_ptr, xpu_stream);
  }
};

} // namespace c10::xpu::impl

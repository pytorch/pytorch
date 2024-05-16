#pragma once
#include <ATen/xpu/XPUContext.h>

#include <optional>

namespace at::xpu {

/*
 * XPUEvent are movable not copyable wrappers around SYCL event. XPUEvent are
 * constructed lazily when first recorded. It has a device, and this device is
 * acquired from the first recording stream. Later streams that record the event
 * must match the same device.
 *
 * Currently, XPUEvent does NOT support to export an inter-process event from
 * another process via inter-process comunication(IPC). So it means that
 * inter-process communication for event handles between different processes is
 * not available. This could impact some applications that rely on cross-process
 * synchronization and communication.
 */
struct TORCH_XPU_API XPUEvent {
  // Constructors
  XPUEvent(bool enable_timing = false) noexcept
      : enable_timing_{enable_timing} {}

  ~XPUEvent() {
    if (isCreated()) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_deletion(
            at::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
    }
  }

  XPUEvent(const XPUEvent&) = delete;
  XPUEvent& operator=(const XPUEvent&) = delete;

  XPUEvent(XPUEvent&& other) = default;
  XPUEvent& operator=(XPUEvent&& other) = default;

  operator sycl::event&() const {
    return event();
  }

  std::optional<at::Device> device() const {
    if (isCreated()) {
      return at::Device(at::kXPU, device_index_);
    } else {
      return std::nullopt;
    }
  }

  inline bool isCreated() const {
    return (event_.get() != nullptr);
  }

  DeviceIndex device_index() const {
    return device_index_;
  }

  sycl::event& event() const {
    return *event_;
  }

  bool query() const {
    using namespace sycl::info;
    if (!isCreated()) {
      return true;
    }

    return event().get_info<event::command_execution_status>() ==
        event_command_status::complete;
  }

  void record() {
    record(getCurrentXPUStream());
  }

  void recordOnce(const XPUStream& stream) {
    if (!isCreated()) {
      record(stream);
    }
  }

  void record(const XPUStream& stream) {
    if (!isCreated()) {
      device_index_ = stream.device_index();
      event_ = std::make_unique<sycl::event>(
          stream.queue().ext_oneapi_submit_barrier());
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_creation(
            at::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
    } else {
      TORCH_CHECK(
          device_index_ == stream.device_index(),
          "Event device ",
          device_index_,
          " does not match recording stream's device ",
          stream.device_index(),
          ".");
      event_.reset();
      event_ = std::make_unique<sycl::event>(
          stream.queue().ext_oneapi_submit_barrier());
    }
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          at::kXPU,
          reinterpret_cast<uintptr_t>(event_.get()),
          reinterpret_cast<uintptr_t>(&stream.queue()));
    }
  }

  void block(const XPUStream& stream) {
    if (isCreated()) {
      std::vector<sycl::event> event_list{event()};
      // Make this stream wait until event_ is completed.
      stream.queue().ext_oneapi_submit_barrier(event_list);
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(
            at::kXPU,
            reinterpret_cast<uintptr_t>(event_.get()),
            reinterpret_cast<uintptr_t>(&stream.queue()));
      }
    }
  }

  float elapsed_time(const XPUEvent& end) const {
    TORCH_CHECK(
        isCreated() && end.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && end.query(),
        "Both events must be completed before calculating elapsed time.");
    TORCH_CHECK(
        enable_timing_ && end.enable_timing_,
        "Both events must be created with argument 'enable_timing=True'.");
    auto end_time =
        end.event()
            .get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start_time =
        this->event()
            .get_profiling_info<sycl::info::event_profiling::command_end>();

    auto time_ms = (end_time - start_time) / 1e6;
    return time_ms;
  }

  void synchronize() const {
    if (isCreated()) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_synchronization(
            at::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
      event().wait_and_throw();
    }
  }

 private:
  bool enable_timing_ = false;
  DeviceIndex device_index_ = -1;
  // Only need to track the last event, as events in an in-order queue are
  // executed sequentially.
  std::unique_ptr<sycl::event> event_;
};

} // namespace at::xpu

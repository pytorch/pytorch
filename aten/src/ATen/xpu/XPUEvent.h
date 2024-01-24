#pragma once
#include <ATen/xpu/XPUContext.h>
#include <c10/util/Optional.h>

namespace at::xpu {

/*
 * XPUEvent are movable not copyable wrappers around SYCL event. XPUEvent are
 * constructed lazily when first recorded. It has a device, and this device is
 * acquired from the first recording stream. Later streams that record the event
 * must match the same device.
 */
struct TORCH_XPU_API XPUEvent {
  // Constructors
  XPUEvent() noexcept = default;
  XPUEvent(bool enable_timing) noexcept : enable_timing_{enable_timing} {}

  ~XPUEvent() {}

  XPUEvent(const XPUEvent&) = delete;
  XPUEvent& operator=(const XPUEvent&) = delete;

  XPUEvent(XPUEvent&& other) noexcept {
    moveHelper(std::move(other));
  }

  XPUEvent& operator=(XPUEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator sycl::event&() const {
    return event();
  }

  optional<at::Device> device() const {
    if (isCreated()) {
      return at::Device(at::kXPU, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const {
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
    } else {
      TORCH_CHECK(
          device_index_ == stream.device_index(),
          "Event device ",
          device_index_,
          " does not match recording stream's device ",
          stream.device_index(),
          ".");
      event_.reset();
    }
    event_ = std::make_unique<sycl::event>(
        stream.queue().ext_oneapi_submit_barrier());
  }

  void block(const XPUStream& stream) {
    if (isCreated()) {
      std::vector<sycl::event> event_list{event()};
      // Make this stream wait until event_ is completed.
      stream.queue().ext_oneapi_submit_barrier(event_list);
    }
  }

  float elapsed_time(const XPUEvent& other) const {
    TORCH_CHECK(
        isCreated() && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");
    TORCH_CHECK(
        enable_timing_ && other.enable_timing_,
        "Both events must be created with argument 'enable_timing=True'.");
    // TODO: provides the ability to time the execution of commands in a SYCL
    // queue without enabling profiling on the entire queue
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "elapsed_time is not supported by XPUEvent.");
  }

  void synchronize() const {
    if (isCreated()) {
      event().wait_and_throw();
    }
  }

 private:
  bool enable_timing_ = false;
  DeviceIndex device_index_ = -1;
  // Only need to track the last event, as events in an in-order queue are
  // executed sequentially.
  std::unique_ptr<sycl::event> event_;

  void moveHelper(XPUEvent&& other) {
    std::swap(enable_timing_, other.enable_timing_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace at::xpu

#pragma once
#include <c10/xpu/XPUStream.h>

namespace c10::xpu {

/*
 * XPUEvent are movable not copyable wrappers around SYCL event. XPUEvent are
 * constructed lazily when first recorded. It has a device, and this device is
 * acquired from the first recording stream. Later streams that record the event
 * must match the same device.
 *
 * IPC (inter-process communication) for event handles is supported. If
 * reconstructed from a handle or ipc_handle() is called, the XPUEvent is
 * initialized eagerly instead of lazily.
 */
struct XPUEvent {
  // Constructors
  XPUEvent(bool enable_timing = false, bool enable_ipc = false) noexcept
      : enable_timing_{enable_timing}, enable_ipc_{enable_ipc} {}

#if SYCL_COMPILER_VERSION >= 20260200
  XPUEvent(
      DeviceIndex device_index,
      const sycl::ext::oneapi::experimental::ipc::handle_data_t& handle_data)
      : device_index_(device_index) {
#ifdef _WIN32
    TORCH_CHECK(false, "XPU IPC events are not supported on Windows.");
#endif
    // Events reconstructed from an IPC handle cannot be re-exported via
    // ipc_handle(). So keep `enable_ipc_` false to avoid confusion.
    auto& device = c10::xpu::get_raw_device(device_index);
    reusable_ = device.has(sycl::aspect::ext_oneapi_ipc_event) &&
        device.has(sycl::aspect::ext_oneapi_per_event_profiling);
    TORCH_CHECK(
        reusable_,
        "XPUEvent reconstructed from an IPC handle must be reusable.");
    event_ = std::make_unique<sycl::event>(
        sycl::ext::oneapi::experimental::ipc::event::open(
            handle_data, c10::xpu::get_device_context()));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          c10::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
    }
  }
#endif

  ~XPUEvent() {
    if (isCreated()) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_deletion(
            c10::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
    }
  }

  C10_DISABLE_COPY_AND_ASSIGN(XPUEvent);

  XPUEvent(XPUEvent&& other) = default;
  XPUEvent& operator=(XPUEvent&& other) = default;

  operator sycl::event&() const {
    return event();
  }

  std::optional<c10::Device> device() const {
    if (isCreated()) {
      return c10::Device(c10::kXPU, device_index_);
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
    namespace syclex = sycl::ext::oneapi::experimental;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (!isCreated()) {
      createEvent(stream.device_index());
      if (!reusable_) {
        assignEvent(stream.queue());
      }
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_creation(
            c10::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
    }
    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");

    if (reusable_) {
#if SYCL_COMPILER_VERSION >= 20260200
      syclex::enqueue_signal_event(stream.queue(), *event_);
#endif
    } else {
      reassignEvent(stream.queue());
    }

    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::kXPU,
          reinterpret_cast<uintptr_t>(event_.get()),
          reinterpret_cast<uintptr_t>(&stream.queue()));
    }
  }

  void block(const XPUStream& stream) {
    if (isCreated()) {
      if (reusable_) {
#if SYCL_COMPILER_VERSION >= 20260200
        sycl::ext::oneapi::experimental::enqueue_wait_event(
            stream.queue(), *event_);
#endif
      } else {
        std::vector<sycl::event> event_list{event()};
        // Make this stream wait until event_ is completed.
        stream.queue().ext_oneapi_submit_barrier(event_list);
      }

      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(
            c10::kXPU,
            reinterpret_cast<uintptr_t>(event_.get()),
            reinterpret_cast<uintptr_t>(&stream.queue()));
      }
    }
  }

  double elapsed_time(const XPUEvent& other) const {
    TORCH_CHECK(
        isCreated() && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");
    TORCH_CHECK(
        enable_timing_ && other.enable_timing_,
        "Both events must be created with argument 'enable_timing=True'.");

    using namespace sycl::info::event_profiling;
    // Block until both of the recorded events are completed.
    uint64_t end_time_ns = other.event().get_profiling_info<command_end>();
    uint64_t start_time_ns = event().get_profiling_info<command_end>();
    // Return the elapsed time in milliseconds.
    return 1e-6 *
        (static_cast<double>(end_time_ns) - static_cast<double>(start_time_ns));
  }

  void synchronize() const {
    if (isCreated()) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_synchronization(
            c10::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
      event().wait_and_throw();
    }
  }

#if SYCL_COMPILER_VERSION >= 20260200
  sycl::ext::oneapi::experimental::ipc::handle_data_t ipc_handle() {
    TORCH_CHECK(
        enable_ipc_,
        "XPUEvent ipc_handle() requires the event to be constructed with enable_ipc=True.");
    if (!isCreated()) {
      namespace syclex = sycl::ext::oneapi::experimental;
      createEvent(c10::xpu::current_device());
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_creation(
            c10::kXPU, reinterpret_cast<uintptr_t>(event_.get()));
      }
    }
    TORCH_CHECK(reusable_, "XPUEvent must be reusable to support IPC.");
    return sycl::ext::oneapi::experimental::ipc::event::get(*event_).data();
  }
#endif

 private:
  void assignEvent(sycl::queue& queue) {
    if (enable_timing_) {
      event_ = std::make_unique<sycl::event>(
          sycl::ext::oneapi::experimental::submit_profiling_tag(queue));
    } else {
      event_ = std::make_unique<sycl::event>(queue.ext_oneapi_submit_barrier());
    }
  }

  void reassignEvent(sycl::queue& queue) {
    event_.reset();
    assignEvent(queue);
  }

  void createEvent(c10::DeviceIndex device_index) {
    device_index_ = device_index;
    TORCH_CHECK(
        !enable_ipc_ || !enable_timing_,
        "XPUEvent cannot have both IPC and timing enabled.");
#ifdef _WIN32
    TORCH_CHECK(!enable_ipc_, "XPU IPC events are not supported on Windows.");
#endif
#if SYCL_COMPILER_VERSION >= 20260200
    namespace syclex = sycl::ext::oneapi::experimental;

    auto& device = c10::xpu::get_raw_device(device_index_);
    if (enable_ipc_) {
      TORCH_CHECK(
          device.has(sycl::aspect::ext_oneapi_ipc_event),
          "Requires the ext_oneapi_ipc_event extension, "
          "which is not supported on this device. ",
          "Please upgrade to a newer driver.");
    }
    // Base reusability on per-event profiling support regardless of
    // enable_timing_, to align with c10::Event behavior.
    reusable_ = device.has(sycl::aspect::ext_oneapi_per_event_profiling);
    if (reusable_) {
      event_ = std::make_unique<sycl::event>(syclex::make_event(
          c10::xpu::get_device_context(),
          syclex::properties{
              syclex::enable_ipc{enable_ipc_},
              syclex::enable_profiling{enable_timing_}}));
    }
#else
    TORCH_CHECK(
        !enable_ipc_, "XPU IPC events require SYCL compiler 2026.2 or later.");
#endif
  }

  bool enable_timing_ = false;
  bool enable_ipc_ = false;
  bool reusable_ = false;
  c10::DeviceIndex device_index_ = -1;
  std::unique_ptr<sycl::event> event_;
};

} // namespace c10::xpu

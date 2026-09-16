#ifndef AOTI_RUNTIME_STREAMS_XPU_H
#define AOTI_RUNTIME_STREAMS_XPU_H

#include <cstddef>
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

#include <torch/csrc/inductor/aoti_torch/c/shim_xpu.h>

namespace torch::aot_inductor {

// XPU counterpart of AOTIPerThreadStreamCache (see streams.h). SYCL has no
// driver-level "create a stream on this device" call, so slots > 0 are
// pulled from c10's XPU stream pool via the stable-ABI shim instead of being
// created/destroyed directly; the pool owns the queues for the process
// lifetime, so there is nothing to clean up here.
struct AOTIPerThreadStreamCache {
  std::vector<std::vector<sycl::queue*>> streams_by_device;

  // sycl::queue::ext_oneapi_get_state() is the SYCL-graph analog of
  // cudaStreamIsCapturing: it reports whether this queue is currently
  // recording into a command_graph rather than executing eagerly.
  void check_not_capturing(sycl::queue* caller_stream) {
    namespace syclex = sycl::ext::oneapi::experimental;
    AOTI_RUNTIME_CHECK(
        caller_stream->ext_oneapi_get_state() == syclex::queue_state::executing,
        "AOTI user streams are not supported during XPU graph capture");
  }

  std::vector<sycl::queue*>& streams(int device_idx) {
    AOTI_RUNTIME_CHECK(device_idx >= 0, "AOTI stream cache device index < 0");
    const auto slot = static_cast<std::size_t>(device_idx);
    if (streams_by_device.size() <= slot) {
      streams_by_device.resize(slot + 1);
    }
    return streams_by_device[slot];
  }

  void ensure(std::size_t count, int device_idx, sycl::queue* caller_stream) {
    check_not_capturing(caller_stream);
    auto& device_streams = streams(device_idx);
    const std::size_t old_size = device_streams.size();
    if (old_size >= count) {
      return;
    }
    device_streams.resize(count, nullptr);
    for (std::size_t i = old_size; i < count; ++i) {
      if (i == 0) {
        continue;
      }
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_xpu_stream_from_pool(
          device_idx, reinterpret_cast<void**>(&device_streams[i])));
    }
  }

  sycl::queue* get(
      int stream_idx,
      int device_idx,
      sycl::queue* caller_stream) {
    AOTI_RUNTIME_CHECK(
        stream_idx > 0,
        "AOTI aux stream cache slot 0 is reserved for the caller stream");
    const auto slot = static_cast<std::size_t>(stream_idx);
    ensure(slot + 1, device_idx, caller_stream);
    return streams(device_idx)[slot];
  }
};

// XPU counterpart of AOTIPerThreadEventCache. Mirrors c10::xpu::XPUEvent's
// two record strategies: on toolchains/devices that support per-event
// profiling, a single event is created once via make_event and re-signaled
// on every record_event (cheap, handle-reuse, like cudaEventRecord);
// otherwise each record_event replaces the slot with a fresh event from
// ext_oneapi_submit_barrier, since a plain sycl::event cannot be re-signaled.
struct AOTIPerThreadEventCache {
  struct Slot {
    std::unique_ptr<sycl::event> event;
    bool reusable = false;
  };
  std::vector<std::vector<Slot>> slots_by_device;

  std::vector<Slot>& slots(int device_idx) {
    AOTI_RUNTIME_CHECK(device_idx >= 0, "AOTI event cache device index < 0");
    const auto slot = static_cast<std::size_t>(device_idx);
    if (slots_by_device.size() <= slot) {
      slots_by_device.resize(slot + 1);
    }
    return slots_by_device[slot];
  }

  Slot& slot_for(int event_idx, int device_idx) {
    AOTI_RUNTIME_CHECK(event_idx >= 0, "AOTI event cache event index < 0");
    auto& device_slots = slots(device_idx);
    const auto idx = static_cast<std::size_t>(event_idx);
    if (device_slots.size() <= idx) {
      device_slots.resize(idx + 1);
    }
    return device_slots[idx];
  }

  // Marks slot `event_idx` at the current point of `stream`.
  void record(int event_idx, int device_idx, sycl::queue* stream) {
    namespace syclex = sycl::ext::oneapi::experimental;
    Slot& s = slot_for(event_idx, device_idx);
    if (!s.event) {
#if SYCL_COMPILER_VERSION >= 20260200
      s.reusable =
          stream->get_device().has(sycl::aspect::ext_oneapi_per_event_profiling);
      if (s.reusable) {
        s.event = std::make_unique<sycl::event>(syclex::make_event(
            stream->get_context(),
            syclex::properties{
                syclex::enable_ipc{false}, syclex::enable_profiling{false}}));
      }
#endif
    }
#if SYCL_COMPILER_VERSION >= 20260200
    if (s.reusable) {
      syclex::enqueue_signal_event(*stream, *s.event);
      return;
    }
#endif
    s.event = std::make_unique<sycl::event>(stream->ext_oneapi_submit_barrier());
  }

  // Makes `stream` wait for slot `event_idx`, which must already be recorded.
  void wait(int event_idx, int device_idx, sycl::queue* stream) {
    namespace syclex = sycl::ext::oneapi::experimental;
    Slot& s = slot_for(event_idx, device_idx);
    AOTI_RUNTIME_CHECK(
        s.event != nullptr, "AOTI event cache: wait on an unrecorded event");
#if SYCL_COMPILER_VERSION >= 20260200
    if (s.reusable) {
      syclex::enqueue_wait_event(*stream, *s.event);
      return;
    }
#endif
    stream->ext_oneapi_submit_barrier({*s.event});
  }

  // Blocks the host until slot `event_idx` completes.
  void synchronize(int event_idx, int device_idx) {
    Slot& s = slot_for(event_idx, device_idx);
    if (s.event) {
      s.event->wait_and_throw();
    }
  }
};

} // namespace torch::aot_inductor

#endif // AOTI_RUNTIME_STREAMS_XPU_H

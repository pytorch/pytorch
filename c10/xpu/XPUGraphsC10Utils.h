#pragma once

#include <c10/xpu/XPUStream.h>
#include <iostream>
#include <mutex>

// XPU Graphs utils used by c10 and aten.
using namespace sycl::ext::oneapi::experimental;
namespace c10::xpu {

static_assert(
    int8_t(queue_state::executing) == 0,
    "unexpected int(queue_state::executing) value");
static_assert(
    int8_t(queue_state::recording) == 1,
    "unexpected int(queue_state::recording) value");

enum class CaptureStatus : int8_t {
  Executing = int8_t(queue_state::executing),
  Recording = int8_t(queue_state::recording)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::Executing:
      os << "Executing";
      break;
    case CaptureStatus::Recording:
      os << "Recording";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown XPU graph CaptureStatus", int(status));
  }
  return os;
}

inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  auto state = c10::xpu::getCurrentXPUStream().queue().ext_oneapi_get_state();
  return CaptureStatus(state);
}

// SYCL graph capture only records ops submitted to the queue(s) passed to
// begin_recording(). A queue that is never registered is not part of the
// graph, so any cross-queue wait involving one of its events fails during
// capture ("Graph nodes cannot depend on events from outside the graph").
// This tracks the graph currently being recorded (if any) so components
// that run work on a queue other than the capturing stream -- e.g. a
// communication library's internal collective stream -- can register that
// queue into the same recording session and make ordinary cross-queue
// synchronization (XPUEvent::block(), i.e. ext_oneapi_submit_barrier) work
// during capture, with no change needed to the wait itself.
class C10_XPU_API XPUGraphCaptureRegistry {
 public:
  static XPUGraphCaptureRegistry& get() {
    static XPUGraphCaptureRegistry instance;
    return instance;
  }

  // Called by XPUGraphImpl::capture_begin() once recording has started on
  // the capturing stream. `graph` must outlive the capture.
  void setActiveGraph(command_graph<graph_state::modifiable>* graph) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_graph_ = graph;
  }

  // Called by XPUGraphImpl::capture_end().
  void clearActiveGraph() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_graph_ = nullptr;
  }

  // Registers `queue` into the currently-recording graph, if any. No-op if
  // no capture is active or `queue` is already part of a recording session
  // (begin_recording() puts the queue itself into queue_state::recording,
  // so that state doubles as the "already registered" check).
  void addQueueIfCapturing(sycl::queue& queue) {
    if (queue.ext_oneapi_get_state() == queue_state::recording) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_graph_ != nullptr) {
      active_graph_->begin_recording(queue);
    }
  }

 private:
  std::mutex mutex_;
  command_graph<graph_state::modifiable>* active_graph_ = nullptr;
};

inline void addStreamToCurrentCaptureIfCapturing(const XPUStream& stream) {
  XPUGraphCaptureRegistry::get().addQueueIfCapturing(stream.queue());
}

} // namespace c10::xpu

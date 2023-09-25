#pragma once
#include <torch/csrc/autograd/profiler.h>

namespace c10d {
constexpr int kUnsetTime = -1;

inline int64_t current_time_in_nanos() {
  return torch::profiler::impl::getTime();
}

class TORCH_API Timer {
 private:
  // The timestamp of forward call start time in each iteration.
  int64_t forward_start_time = kUnsetTime;
  // The timestamp of backward computation start and end time in each
  // iteration.
  int64_t backward_compute_start_time = kUnsetTime;
  int64_t backward_compute_end_time = kUnsetTime;
  // The timestamp of first communication call start time in each iteration.
  int64_t backward_comm_start_time = kUnsetTime;
  // The timestamp of last communication call end time in each iteration.
  int64_t backward_comm_end_time = kUnsetTime;

 public:
  enum class Event {
    kForwardStart,
    kBackwardComputeStart,
    kBackwardComputeEnd,
    kBackwardCommStart,
    kBackwardCommEnd,
  };

  // Record the current event, i.e., mark it as having occurred now. Default
  // CPU implementation.
  virtual void record(Event event) {
    getTimeRef(event) = current_time_in_nanos();
  }

  // Return the difference between when two events occurred, in nanoseconds.
  // Or nullopt if one of them hasn't been recorded.
  virtual c10::optional<int64_t> measureDifference(Event start, Event end) = 0;

  virtual ~Timer() = default;

  // Return host-side timestamp, or nullopt if it has not yet been recorded.
  c10::optional<int64_t> getTimestamp(Event event) {
    auto time = getTimeRef(event);
    if (time == kUnsetTime) {
      return c10::nullopt;
    } else {
      return time;
    }
  }

  // Return host-side time member variable corresponding to the given event.
  int64_t& getTimeRef(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start_time;
      case Event::kBackwardComputeStart:
        return backward_compute_start_time;
      case Event::kBackwardComputeEnd:
        return backward_compute_end_time;
      case Event::kBackwardCommStart:
        return backward_comm_start_time;
      case Event::kBackwardCommEnd:
        return backward_comm_end_time;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }
};

TORCH_DECLARE_TYPED_REGISTRY(
    TimerRegistry,
    c10::DeviceType,
    Timer,
    std::unique_ptr,
    c10::Device);
} // namespace c10d

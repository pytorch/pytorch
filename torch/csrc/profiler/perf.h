#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/csrc/profiler/events.h>

#include <c10/util/Exception.h>

namespace torch {
namespace profiler {
namespace impl {
namespace linux_perf {

/*
 * Maximum number of events supported
 * This stems from the hardware limitation on CPU performance counters, and the
 * fact that we don't support time multiplexing just yet.
 * Time multiplexing involves scaling the counter values proportional to
 * the enabled and running time or running the workload multiple times.
 */
constexpr uint8_t MAX_EVENTS = 4;

struct PerfCounter {
  uint64_t value; /* The value of the event */
  uint64_t time_enabled; /* for TIME_ENABLED */
  uint64_t time_running; /* for TIME_RUNNING */
};

/*
 * Basic perf event handler for Android and Linux
 */
class PerfEvent {
 public:
  explicit PerfEvent(std::string& name) : name_(name), fd_(-1) {}

  PerfEvent& operator=(PerfEvent&& other) noexcept {
    if (this != &other) {
      fd_ = other.fd_;
      other.fd_ = -1;
      name_ = std::move(other.name_);
    }
    return *this;
  }

  PerfEvent(PerfEvent&& other) noexcept {
    *this = std::move(other);
  }

  ~PerfEvent();

  /* Setup perf events with the Linux Kernel, attaches perf to this process
   * using perf_event_open(2) */
  void Init();

  /* Stop incrementing hardware counters for this event */
  void Disable() const;

  /* Start counting hardware event from this point on */
  void Enable() const;

  /* Zero out the counts for this event */
  void Reset() const;

  /* Returns PerfCounter values for this event from kernel, on non supported
   * platforms this always returns zero */
  uint64_t ReadCounter() const;

 private:
  /* Name of the event */
  std::string name_;

  int fd_ = -1;
};

class PerfProfiler {
 public:
  /* Configure all the events and track them as individual PerfEvent */
  void Configure(std::vector<std::string>& event_names);

  /* Enable events counting from here */
  void Enable();

  /* Disable counting and fill in the caller supplied container with delta
   * calculated from the start count values since last Enable() */
  void Disable(perf_counters_t&);

 private:
  uint64_t CalcDelta(uint64_t start, uint64_t end) const;

  bool is_enabled_{false};
  std::vector<PerfEvent> events_;
  perf_counters_t start_values_;
};
} // namespace linux_perf
} // namespace impl
} // namespace profiler
} // namespace torch

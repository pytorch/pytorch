#pragma once

#include <torch/csrc/autograd/profiler.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::autograd::profiler;

// Process global profiler state.
class ProcessGlobalProfilerState {
 public:
  explicit ProcessGlobalProfilerState(const ProfilerConfig& config)
      : config_(config) {}
  ~ProcessGlobalProfilerState() = default;

  const ProfilerConfig& config() const {
    return config_;
  }

  void pushProfileRange(thread_event_lists profileRange) {
    std::unique_lock<std::mutex> lock(profileRangesMutex_);

    // NB: When a thread wants to push an entry into the this container,
    // main control logic might have exited the process-global profile range.
    profileRanges_.emplace_back(std::move(profileRange));
  }

  std::vector<thread_event_lists> profileRanges() {
    std::unique_lock<std::mutex> lock(profileRangesMutex_);

    std::vector<thread_event_lists> results;
    results.insert(
        results.begin(),
        std::make_move_iterator(profileRanges_.begin()),
        std::make_move_iterator(profileRanges_.end()));
    profileRanges_.erase(profileRanges_.begin(), profileRanges_.end());
    return results;
  }

 private:
  // Name it profileRanges_ to emphesize on the fact that each element is the
  // results of a profile range. In each profile range, there is a
  // "__profiler_start" mark event that all following events calculate time
  // relative to it, so it's required to call
  // parse_cpu_trace(profileRange) for every profile range.
  std::mutex profileRangesMutex_;
  std::vector<thread_event_lists> profileRanges_;
  ProfilerConfig config_ =
      ProfilerConfig(ProfilerState::Disabled, false, false);
};

// User-facing API.
TORCH_API void enableServerProcessGlobalProfiler(
    const ProfilerConfig& new_config);
TORCH_API std::vector<thread_event_lists> disableServerProcessGlobalProfiler();

// Internal API.
bool serverProcessGlobalProfilerEnabled();

std::shared_ptr<ProcessGlobalProfilerState> serverProcessGlobalProfilerState();

} // namespace rpc
} // namespace distributed
} // namespace torch

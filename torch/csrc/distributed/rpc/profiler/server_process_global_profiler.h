#pragma once

#include <shared_mutex>

#include <torch/csrc/autograd/profiler.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace profiler {
namespace processglobal {

using namespace torch::autograd::profiler;

// Process global profiler state.
//
// This class holds information about a profiling range, from "enable" to
// "disable".
// An instance of this ``State`` will be
// pushed into a global stack, so nested profiling range is supported.
//
// It has 2 members.
// One is ``autograd::profiler::ProfilerConfig``. It's set by user and
// will be copied to thread-local profiler state of RPC threads.
// The other is a container that aggregates recorded
// ``autograd::profiler::Event``s from all thread-local profilers on RPC
// threads.
class State {
 public:
  explicit State(const ProfilerConfig& config) : config_(config) {}
  ~State() = default;

  const ProfilerConfig& config() const {
    return config_;
  }

  void pushResult(thread_event_lists result) {
    std::unique_lock<std::mutex> lock(resultsMutex_);

    // NB: When a thread wants to push an entry into the this container,
    // main control logic might have exited the process-global profile range.
    results_.emplace_back(std::move(result));
  }

  std::vector<thread_event_lists> results();

 private:
  // Each result comes from a profile range. In each profile range, there is a
  // "__profiler_start" marker event that all following events calculate time
  // relative to it, so it's required to call
  // parse_cpu_trace(result) for results of all profile range.
  std::mutex resultsMutex_;
  std::vector<thread_event_lists> results_;
  const ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);
};

class StateStackEntry;

#if defined(__MACH__)
// Compiler error: 'shared_timed_mutex' is unavailable: introduced in
// macOS 10.12
using mutexType = std::mutex;
// Compiler error: 'shared_lock' is unavailable: introduced in
// macOS 10.12
using rLockType = std::unique_lock<std::mutex>;
using wLockType = std::unique_lock<std::mutex>;
#else
using mutexType = std::shared_timed_mutex;
using rLockType = std::shared_lock<std::shared_timed_mutex>;
using wLockType = std::unique_lock<std::shared_timed_mutex>;
#endif

// This is the global stack of ``State``s.
TORCH_API extern std::shared_ptr<StateStackEntry> currentStateStackEntryPtr;
TORCH_API extern mutexType currentStateStackEntryMutex;

// This class is used to implement a stack of ``State``s.
// It has 2 members.
// One is `prevPtr`, a shared_ptr poiniting to previous elememnt in the
// stack.
// The other is ``statePtr``, a shared_ptr pointing to ``State``.
class StateStackEntry {
 public:
  StateStackEntry(
      // NOLINTNEXTLINE(modernize-pass-by-value)
      std::shared_ptr<StateStackEntry> prevPtr,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      std::shared_ptr<State> statePtr)
      : prevPtr_(prevPtr), statePtr_(statePtr) {}

  static void pushRange(std::shared_ptr<State> profilerProcessGlobalStatePtr);
  static std::shared_ptr<State> popRange();

  static std::shared_ptr<StateStackEntry> current() {
    rLockType rlock(currentStateStackEntryMutex);

    return currentStateStackEntryPtr;
  }

  std::shared_ptr<StateStackEntry> prevPtr() const {
    return prevPtr_;
  }

  std::shared_ptr<State> statePtr() const {
    return statePtr_;
  }

 private:
  const std::shared_ptr<StateStackEntry> prevPtr_{nullptr};
  const std::shared_ptr<State> statePtr_{nullptr};
};

// Push the result to ``State``s of current profile range and recursively outer
// profile ranges.
TORCH_API void pushResultRecursive(
    std::shared_ptr<StateStackEntry> stateStackEntryPtr,
    const thread_event_lists& result);

// User-facing API.
//
// Enter a server-side process-global profiling range.
// Profiling range can be neste, so it's ok to call this API for multiple
// times. This enables all RPC threads running server-side request callbacks.
TORCH_API void enableServer(const ProfilerConfig& new_config);
//
// Exit a server-side process-global profiling range.
// Profiling range can be neste, so it's possible that profiler is still on
// after calling this API.
// This enables all RPC threads running server-side request callbacks.
TORCH_API std::vector<thread_event_lists> disableServer();

} // namespace processglobal
} // namespace profiler
} // namespace rpc
} // namespace distributed
} // namespace torch

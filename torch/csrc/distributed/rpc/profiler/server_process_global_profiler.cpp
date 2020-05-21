#include <shared_mutex>

#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

class ProcessGlobalProfilerStateStackEntry;

#if defined(__MACH__)
// Compiler error: 'shared_timed_mutex' is unavailable: introduced in
// macOS 10.12
std::mutex
#else
std::shared_timed_mutex
#endif
    currentProcessGlobalProfilerStateStackEntryMutex;
std::shared_ptr<ProcessGlobalProfilerStateStackEntry>
    currentProcessGlobalProfilerStateStackEntryPtr = nullptr;

class ProcessGlobalProfilerStateStackEntry {
 public:
  static std::shared_ptr<ProcessGlobalProfilerState> current() {
#if defined(__MACH__)
    std::unique_lock<std::mutex>
#else
    std::shared_lock<std::shared_timed_mutex>
#endif
        rlock(currentProcessGlobalProfilerStateStackEntryMutex);

    if (!currentProcessGlobalProfilerStateStackEntryPtr) {
      return nullptr;
    }
    return currentProcessGlobalProfilerStateStackEntryPtr
        ->profilerProcessGlobalStatePtr_;
  }

  static void push(std::shared_ptr<ProcessGlobalProfilerState>
                       profilerProcessGlobalStatePtr) {
#if defined(__MACH__)
    std::unique_lock<std::mutex>
#else
    std::unique_lock<std::shared_timed_mutex>
#endif
        wlock(currentProcessGlobalProfilerStateStackEntryMutex);

    auto previousStateStackEntryPtr =
        currentProcessGlobalProfilerStateStackEntryPtr;
    currentProcessGlobalProfilerStateStackEntryPtr =
        std::make_shared<ProcessGlobalProfilerStateStackEntry>();
    currentProcessGlobalProfilerStateStackEntryPtr
        ->previousProcessGlobalProfilerStateStackEntryPtr_ =
        previousStateStackEntryPtr;
    currentProcessGlobalProfilerStateStackEntryPtr
        ->profilerProcessGlobalStatePtr_ =
        std::move(profilerProcessGlobalStatePtr);
  }

  static std::shared_ptr<ProcessGlobalProfilerState> pop() {
#if defined(__MACH__)
    std::unique_lock<std::mutex>
#else
    std::unique_lock<std::shared_timed_mutex>
#endif
        wlock(currentProcessGlobalProfilerStateStackEntryMutex);

    auto poppedStateStackEntryPtr =
        currentProcessGlobalProfilerStateStackEntryPtr;
    TORCH_INTERNAL_ASSERT(
        poppedStateStackEntryPtr &&
        poppedStateStackEntryPtr->profilerProcessGlobalStatePtr_);
    currentProcessGlobalProfilerStateStackEntryPtr =
        poppedStateStackEntryPtr
            ->previousProcessGlobalProfilerStateStackEntryPtr_;
    return poppedStateStackEntryPtr->profilerProcessGlobalStatePtr_;
  }

 private:
  std::shared_ptr<ProcessGlobalProfilerState> profilerProcessGlobalStatePtr_{
      nullptr};
  std::shared_ptr<ProcessGlobalProfilerStateStackEntry>
      previousProcessGlobalProfilerStateStackEntryPtr_{nullptr};
};

} // namespace

using namespace torch::autograd::profiler;

void enableServerProcessGlobalProfiler(const ProfilerConfig& new_config) {
  TORCH_CHECK(
      new_config.state == ProfilerState::CPU, "RPC module only support CPU");
  auto new_state = std::make_shared<ProcessGlobalProfilerState>(new_config);
  ProcessGlobalProfilerStateStackEntry::push(new_state);
}

std::vector<thread_event_lists> disableServerProcessGlobalProfiler() {
  auto disableServerProcessGlobalProfiler =
      ProcessGlobalProfilerStateStackEntry::pop();
  return disableServerProcessGlobalProfiler->profileRanges();
}

bool serverProcessGlobalProfilerEnabled() {
  return ProcessGlobalProfilerStateStackEntry::current() != nullptr;
}

std::shared_ptr<ProcessGlobalProfilerState> serverProcessGlobalProfilerState() {
  return ProcessGlobalProfilerStateStackEntry::current();
}

} // namespace rpc
} // namespace distributed
} // namespace torch

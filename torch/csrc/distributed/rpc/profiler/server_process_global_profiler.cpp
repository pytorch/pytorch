#include <shared_mutex>

#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>

namespace torch::distributed::rpc {

namespace {

class ProcessGlobalProfilerStateStackEntry;

std::shared_timed_mutex currentProcessGlobalProfilerStateStackEntryMutex;
std::shared_ptr<ProcessGlobalProfilerStateStackEntry>
    currentProcessGlobalProfilerStateStackEntryPtr = nullptr;

class ProcessGlobalProfilerStateStackEntry {
 public:
  static std::shared_ptr<ProcessGlobalProfilerState> current() {
    std::shared_lock<std::shared_timed_mutex> rlock(
        currentProcessGlobalProfilerStateStackEntryMutex);

    if (!currentProcessGlobalProfilerStateStackEntryPtr) {
      return nullptr;
    }
    return currentProcessGlobalProfilerStateStackEntryPtr
        ->profilerProcessGlobalStatePtr_;
  }

  static void push(std::shared_ptr<ProcessGlobalProfilerState>
                       profilerProcessGlobalStatePtr) {
    std::unique_lock<std::shared_timed_mutex> wlock(
        currentProcessGlobalProfilerStateStackEntryMutex);

    auto previousStateStackEntryPtr =
        currentProcessGlobalProfilerStateStackEntryPtr;
    currentProcessGlobalProfilerStateStackEntryPtr =
        std::make_shared<ProcessGlobalProfilerStateStackEntry>();
    currentProcessGlobalProfilerStateStackEntryPtr
        ->previousProcessGlobalProfilerStateStackEntryPtr_ =
        previousStateStackEntryPtr;
    currentProcessGlobalProfilerStateStackEntryPtr
        ->profilerProcessGlobalStatePtr_ = profilerProcessGlobalStatePtr;
  }

  static std::shared_ptr<ProcessGlobalProfilerState> pop() {
    std::unique_lock<std::shared_timed_mutex> wlock(
        currentProcessGlobalProfilerStateStackEntryMutex);

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

} // namespace torch::distributed::rpc

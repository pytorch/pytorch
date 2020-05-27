#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>

#include <shared_mutex>

namespace torch {
namespace distributed {
namespace rpc {
namespace profiler {
namespace processglobal {

namespace {
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

mutexType currentStateStackEntryMutex;
std::shared_ptr<StateStackEntry> currentStateStackEntryPtr = nullptr;

class StateStackEntry {
 public:
  static std::shared_ptr<State> current() {
    rLockType rlock(currentStateStackEntryMutex);

    if (!currentStateStackEntryPtr) {
      return nullptr;
    }
    return currentStateStackEntryPtr->profilerProcessGlobalStatePtr_;
  }

  static void push(std::shared_ptr<State> profilerProcessGlobalStatePtr) {
    wLockType wlock(currentStateStackEntryMutex);

    auto previousStateStackEntryPtr = currentStateStackEntryPtr;
    currentStateStackEntryPtr = std::make_shared<StateStackEntry>();
    currentStateStackEntryPtr->previousStateStackEntryPtr_ =
        previousStateStackEntryPtr;
    currentStateStackEntryPtr->profilerProcessGlobalStatePtr_ =
        std::move(profilerProcessGlobalStatePtr);
  }

  static std::shared_ptr<State> pop() {
    wLockType wlock(currentStateStackEntryMutex);

    auto poppedStateStackEntryPtr = currentStateStackEntryPtr;
    TORCH_INTERNAL_ASSERT(
        poppedStateStackEntryPtr &&
        poppedStateStackEntryPtr->profilerProcessGlobalStatePtr_);
    currentStateStackEntryPtr =
        poppedStateStackEntryPtr->previousStateStackEntryPtr_;
    return poppedStateStackEntryPtr->profilerProcessGlobalStatePtr_;
  }

 private:
  std::shared_ptr<State> profilerProcessGlobalStatePtr_{nullptr};
  std::shared_ptr<StateStackEntry> previousStateStackEntryPtr_{nullptr};
};

} // namespace

using namespace torch::autograd::profiler;

void enableServer(const ProfilerConfig& new_config) {
  auto new_state = std::make_shared<State>(new_config);
  StateStackEntry::push(std::move(new_state));
}

std::vector<thread_event_lists> disableServer() {
  auto disableServer = StateStackEntry::pop();
  return disableServer->profileRanges();
}

bool serverEnabled() {
  return StateStackEntry::current() != nullptr;
}

std::shared_ptr<State> serverState() {
  return StateStackEntry::current();
}

} // namespace processglobal
} // namespace profiler
} // namespace rpc
} // namespace distributed
} // namespace torch

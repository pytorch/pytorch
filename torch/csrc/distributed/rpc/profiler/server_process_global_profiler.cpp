#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace profiler {
namespace processglobal {

using namespace torch::autograd::profiler;

std::vector<thread_event_lists> State::results() {
  std::unique_lock<std::mutex> lock(resultsMutex_);

  std::vector<thread_event_lists> results;
  results.swap(results_);
  return results;
}

mutexType currentStateStackEntryMutex;
std::shared_ptr<StateStackEntry> currentStateStackEntryPtr = nullptr;

void StateStackEntry::pushRange(
    std::shared_ptr<State> profilerProcessGlobalStatePtr) {
  wLockType wlock(currentStateStackEntryMutex);

  auto previousStateStackEntryPtr = currentStateStackEntryPtr;
  currentStateStackEntryPtr = std::make_shared<StateStackEntry>(
      previousStateStackEntryPtr, std::move(profilerProcessGlobalStatePtr));
}

std::shared_ptr<State> StateStackEntry::popRange() {
  wLockType wlock(currentStateStackEntryMutex);

  auto poppedStateStackEntryPtr = currentStateStackEntryPtr;
  TORCH_INTERNAL_ASSERT(
      poppedStateStackEntryPtr && poppedStateStackEntryPtr->statePtr_);
  currentStateStackEntryPtr = poppedStateStackEntryPtr->prevPtr_;
  return poppedStateStackEntryPtr->statePtr_;
}

void pushResultRecursive(
    std::shared_ptr<StateStackEntry> stateStackEntryPtr,
    const thread_event_lists& result) {
  while (stateStackEntryPtr) {
    // Put event_lists into the process-global profiler state.
    stateStackEntryPtr->statePtr()->pushResult(result);
    stateStackEntryPtr = stateStackEntryPtr->prevPtr();
  }
}

void enableServer(const ProfilerConfig& new_config) {
  auto new_state = std::make_shared<State>(new_config);
  StateStackEntry::pushRange(std::move(new_state));
}

std::vector<thread_event_lists> disableServer() {
  auto statePtr = StateStackEntry::popRange();
  return statePtr->results();
}

} // namespace processglobal
} // namespace profiler
} // namespace rpc
} // namespace distributed
} // namespace torch

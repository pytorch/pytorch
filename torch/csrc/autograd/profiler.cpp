#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/function.h"

namespace torch { namespace autograd { namespace profiler {

ProfilerState state = ProfilerState::Disabled;
uint32_t next_thread_id = 0;
std::mutex all_event_lists_mutex;
std::list<std::shared_ptr<RangeEventList>> all_event_lists;
thread_local std::shared_ptr<RangeEventList> event_list;
thread_local int32_t thread_id;

void RecordFunction::pushFunctionRange(Function* fn) {
  pushRange(fn->name());
}

void enableProfiler(ProfilerState new_state) {
  TORCH_ASSERT(new_state != ProfilerState::Disabled);
#ifndef WITH_CUDA
  if (new_state == ProfilerState::NVTX)
    throw std::runtime_error("Can't use NVTX profiler - PyTorch was compiled without CUDA");
#endif
  if (state != ProfilerState::Disabled && new_state != state) {
      throw std::runtime_error("can't change kind of profiling (e.g. NVTX to CPU) while profiler is running");
  }
  state = new_state;
  mark("__start_profile");
}

thread_event_lists disableProfiler() {
  if (state == ProfilerState::Disabled) {
    throw std::runtime_error("can't disable profiler when it's not running");
  }
  ProfilerState old_state = state;
  mark("__stop_profile");
  state = ProfilerState::Disabled;
  if (old_state == ProfilerState::NVTX) {
    return thread_event_lists();
  } else {
    thread_event_lists result;
    std::lock_guard<std::mutex> guard(all_event_lists_mutex);
    for (auto it = all_event_lists.begin(); it != all_event_lists.end();) {
      auto & list = *it;
      result.emplace_back(list->consolidate());
      // GC lists that are not held by any threads
      if (list.use_count() == 1) {
        auto current_it = it;
        ++it;
        all_event_lists.erase(current_it);
      } else {
        ++it;
      }
    }
    return result;
  }
}

}}}

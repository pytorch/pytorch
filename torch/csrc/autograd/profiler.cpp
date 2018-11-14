#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/function.h"

#ifdef USE_CUDA
#include "ATen/cuda/CUDAGuard.h"
#endif

#include <sstream>

namespace torch { namespace autograd { namespace profiler {

ProfilerState state = ProfilerState::Disabled;
uint16_t next_thread_id = 0;
std::mutex all_event_lists_mutex;
std::list<std::shared_ptr<RangeEventList>> all_event_lists;
thread_local std::shared_ptr<RangeEventList> event_list;
thread_local uint16_t thread_id;

RangeEventList& getEventList() {
  if (!event_list) {
    std::lock_guard<std::mutex> guard(all_event_lists_mutex);
    event_list = std::make_shared<RangeEventList>();
    thread_id = next_thread_id++;
    all_event_lists.emplace_front(event_list);
  }
  return *event_list;
}

void mark(std::string name, bool include_cuda /* = true */) {
  if (state == ProfilerState::Disabled) {
    return;
  }
  if (state == ProfilerState::NVTX) {
#ifdef USE_CUDA
    nvtxMarkA(name.c_str());
#else
    throw std::logic_error(
        "mark called with NVTX tracing, but compiled without CUDA");
#endif
  } else {
    getEventList().record(
        EventKind::Mark,
        std::move(name),
        thread_id,
        include_cuda && state == ProfilerState::CUDA);
  }
}

const char* c_str(const char *str) { return str; }
// NB: non-const to disallow temporaries (lifetime issues)
const char* c_str(std::string& str) { return str.c_str(); }

template<typename T>
void pushRangeImpl(T name, const char* msg="", int64_t sequence_nr=-1) {
  if (state == ProfilerState::Disabled) {
    return;
  }
  if (state == ProfilerState::NVTX) {
#ifdef USE_CUDA
    if(sequence_nr >= 0) {
      std::stringstream s;
      s << name << msg << sequence_nr;
      nvtxRangePushA(s.str().c_str());
    } else {
      nvtxRangePushA(c_str(name));
    }
#else
    throw std::logic_error(
        "pushRange called with NVTX tracing, but compiled without CUDA");
#endif
  } else {
    getEventList().record(
        EventKind::PushRange,
        std::move(name),
        thread_id,
        state == ProfilerState::CUDA);
  }
}

void pushRange(std::string name) {
  pushRangeImpl(std::move(name));
}

void popRange() {
  if (state == ProfilerState::Disabled) {
    return;
  }
  if (state == ProfilerState::NVTX) {
#ifdef USE_CUDA
    nvtxRangePop();
#else
    throw std::logic_error(
        "popRange called with NVTX tracing, but compiled without CUDA");
#endif
  } else {
    getEventList().record(
        EventKind::PopRange,
        "",
        thread_id,
        state == ProfilerState::CUDA);
  }
}

RecordFunction::RecordFunction(Function* fn) {
  // typeid(*fn).name() would avoid an additional string allocation.
  // However, typeid(*fn).name() would cause nvtx annotations for all user-defined 
  // (Python-side) custom autograd function backward() methods to have the same name,
  // because they route through the same C++ side class.
  // fn->name() ensures that nvtx annotations for custom function backward() methods
  // receive a relevant, demangled name.
  pushRangeImpl(fn->name(), ", stashed seq=", fn->sequence_nr());
}

RecordFunction::RecordFunction(std::string name) {
  pushRangeImpl(std::move(name));
}

RecordFunction::RecordFunction(const char* name) {
  pushRangeImpl<const char*>(name);
}

RecordFunction::RecordFunction(const char* name, int64_t current_sequence_nr)
{
  pushRangeImpl<const char*>(name, ", seq=", current_sequence_nr);
}

#ifdef USE_CUDA
static void onEachDevice(std::function<void(int)> op) {
  at::cuda::OptionalCUDAGuard device_guard;
  int count;
  TORCH_CUDA_CHECK(cudaGetDeviceCount(&count));
  for(int i = 0; i < count; i++) {
    device_guard.set_index(i);
    op(i);
  }
}
#endif

void enableProfiler(ProfilerState new_state) {
  AT_ASSERT(new_state != ProfilerState::Disabled);
#ifndef USE_CUDA
  if (new_state == ProfilerState::NVTX)
    throw std::runtime_error("Can't use NVTX profiler - PyTorch was compiled without CUDA");
#endif
  if (state != ProfilerState::Disabled && new_state != state) {
      throw std::runtime_error("can't change kind of profiling (e.g. NVTX to CPU) while profiler is running");
  }
  state = new_state;

#ifdef USE_CUDA
  if(state == ProfilerState::CUDA) {
    // event recording appears to have some startup overhead, so we need to
    // to generate some dummy events first before recording syncrhonization events
    for(int i = 0; i < 5; i++) {
      onEachDevice([](int d) {
          mark("__cuda_startup");
          cudaDeviceSynchronize();
      });
    }

    // cuda events must be on the same device, so we need a start event recorded
    // for each gpu. we then use this event to synchronize time on the GPU
    // with the CPU clock.
    onEachDevice([](int d) {
        mark("__cuda_start_event");
    });
  }
#endif
  mark("__start_profile", false);
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

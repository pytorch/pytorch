#pragma once

#ifdef WITH_CUDA
#include <nvToolsExt.h>
#endif
#include <thread>
#include <iostream>
#include <mutex>
#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include <list>
#include <sstream>
#include <forward_list>
#include <tuple>
#include "ATen/ATen.h"
#include "torch/csrc/cuda/cuda_check.h"
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace torch { namespace autograd {

struct Function;

namespace profiler {

constexpr inline std::size_t ceilToMultiple(std::size_t a, std::size_t b) {
  return ((a + b - 1) / b) * b;
}

inline uint64_t getTime() {
  using namespace std::chrono;
  using clock = std::conditional<high_resolution_clock::is_steady, high_resolution_clock, steady_clock>::type;
  return duration_cast<nanoseconds>(clock::now().time_since_epoch()).count();
}

enum class EventKind {
  Mark,
  PushRange,
  PopRange
};

struct Event {
  Event(EventKind kind, std::string name, uint32_t thread_id, bool record_cuda)
  : kind_(kind)
  , name_(std::move(name))
  , thread_id_(thread_id) {
#ifdef WITH_CUDA
    if(record_cuda) {
      TORCH_CUDA_CHECK(cudaGetDevice(&device_));
      TORCH_CUDA_CHECK(cudaEventCreate(&event));
      auto stream = at::globalContext().getCurrentCUDAStream();
      cpu_ns_ = getTime();
      TORCH_CUDA_CHECK(cudaEventRecord(event, stream));
    } else {
      cpu_ns_ = getTime();
    }
#else
    cpu_ns_ = getTime();
#endif
  }
  std::string kind() const {
    switch(kind_) {
      case EventKind::Mark: return "mark";
      case EventKind::PushRange: return "push";
      case EventKind::PopRange: return "pop";
    }
    throw std::runtime_error("unknown EventKind");
  }
  const std::string & name() const {
    return name_;
  }
  uint32_t thread_id() const {
    return thread_id_;
  }
  double cpu_elapsed_us(const Event & e) {
    return (e.cpu_ns_ - cpu_ns_)/(1000.0);
  }
  double cuda_elapsed_us(const Event & e) {
#ifdef WITH_CUDA
    if(!e.has_cuda() || !has_cuda()) {
      throw std::logic_error("Events were not recorded for CUDA");
    }
    if(e.device() != device()) {
      throw std::logic_error("Events are not on the same device");
    }
    TORCH_CUDA_CHECK(cudaEventSynchronize(event));
    TORCH_CUDA_CHECK(cudaEventSynchronize(e.event));
    float ms;
    TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, event, e.event));
    return ms*1000.0;
#else
    throw std::logic_error("CUDA not enabled");
#endif
  }
  bool has_cuda() const {
#ifdef WITH_CUDA
    return event != nullptr;
#else
    return false;
#endif
  }
  int device() const {
    return device_;
  }
private:
  EventKind kind_;
  std::string name_;
  uint32_t thread_id_;
  int64_t cpu_ns_; // signed to allow for negative intervals
#ifdef WITH_CUDA
  cudaEvent_t event = nullptr;
#endif
  int device_ = -1;
};

// a linked-list of fixed sized vectors, to avoid
// a std::vector resize from taking a large amount of time inside
// a profiling  event
struct RangeEventList {
  constexpr static std::size_t MB = 1024 * 1024;
  constexpr static std::size_t event_block_size = 16 * MB;
  constexpr static std::size_t num_block_elements =
    event_block_size / ceilToMultiple(sizeof(Event), alignof(Event));
  static_assert(sizeof(Event[num_block_elements]) <= event_block_size,
                "num_block_elements is calculated incorrectly");
  using block_type = std::vector<Event>;

  void allocBlock() {
    blocks.emplace_front();
    blocks.front().reserve(num_block_elements);
  }

  template<typename... Args>
  void record(Args&&... args) {
    if (blocks.empty() || blocks.front().size() == num_block_elements) {
      allocBlock();
    }
    blocks.front().emplace_back(std::forward<Args>(args)...);
  }

  std::vector<Event> consolidate() {
    std::vector<Event> result;
    for (auto & block : blocks) {
      result.insert(result.begin(),
                    std::make_move_iterator(block.begin()),
                    std::make_move_iterator(block.end()));
    }
    blocks.clear();
    return result;
  }

  std::forward_list<block_type> blocks;
};

enum class ProfilerState {
    Disabled,
    CPU, // CPU-only profiling
    CUDA, // CPU + CUDA events
    NVTX,  // only emit NVTX markers
};

extern ProfilerState state;
extern uint32_t next_thread_id;
extern std::mutex all_event_lists_mutex;
extern std::list<std::shared_ptr<RangeEventList>> all_event_lists;

extern thread_local std::shared_ptr<RangeEventList> event_list;
extern thread_local int32_t thread_id;

inline RangeEventList& getEventList() {
  if (!event_list) {
    std::lock_guard<std::mutex> guard(all_event_lists_mutex);
    event_list = std::make_shared<RangeEventList>();
    thread_id = next_thread_id++;
    all_event_lists.emplace_front(event_list);
  }
  return *event_list;
}

inline void mark(std::string name, bool include_cuda = true) {
  if (state == ProfilerState::NVTX) {
#ifdef WITH_CUDA
    nvtxMarkA(name.c_str());
#else
    throw std::logic_error("mark called with NVTX tracing, but compiled without CUDA");
#endif
  } else {
    getEventList().record(EventKind::Mark, std::move(name), thread_id, include_cuda && state == ProfilerState::CUDA);
  }
}

inline void pushRange(std::string name) {
  if (state == ProfilerState::NVTX) {
#ifdef WITH_CUDA
    nvtxRangePushA(name.c_str());
#else
    throw std::logic_error("pushRange called with NVTX tracing, but compiled without CUDA");
#endif
  } else {
    getEventList().record(EventKind::PushRange, std::move(name), thread_id, state == ProfilerState::CUDA);
  }
}

inline void popRange() {
  if (state == ProfilerState::NVTX) {
#ifdef WITH_CUDA
    nvtxRangePop();
#else
    throw std::logic_error("popRange called with NVTX tracing, but compiled without CUDA");
#endif
  } else {
    getEventList().record(EventKind::PopRange, std::string(), thread_id, state == ProfilerState::CUDA);
  }
}

struct RecordFunction {
  explicit RecordFunction(Function *fn) {
    if (state == ProfilerState::Disabled) return;
    pushFunctionRange(fn);
  }

  explicit RecordFunction(std::string name) {
    if (state == ProfilerState::Disabled) return;
    pushRange(std::move(name));
  }

  explicit RecordFunction(const char *name) {
    if (state == ProfilerState::Disabled) return;
    pushRange(name);
  }

  ~RecordFunction() {
    if (state == ProfilerState::Disabled) return;
    popRange();
  }

  // Needed only because we don't have Function defined yet.
  void pushFunctionRange(Function *fn);
};

using thread_event_lists = std::vector<std::vector<Event>>;
// NOTE: changing profiler modes is **NOT THREAD SAFE**. You should ensure that
// there no autograd functions are being executed when these function are used.
void enableProfiler(ProfilerState state);
thread_event_lists disableProfiler();

} // namespace profiler
}} // namespace torch::autograd

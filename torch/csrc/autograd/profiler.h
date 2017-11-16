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
#ifdef WITH_CUDA
static void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define PROFILER_CUDA_CHECK(result) cudaCheck(result,__FILE__,__LINE__);
#endif

struct Event {
  Event(EventKind kind, std::string name, uint32_t thread_id, bool record_cuda)
  : kind_(kind),
  name_(std::move(name)),
  thread_id_(thread_id),
  cpu_ns(getTime()) {
#ifdef WITH_CUDA
    if(record_cuda) {
      PROFILER_CUDA_CHECK(cudaEventCreate(&event));
      PROFILER_CUDA_CHECK(cudaEventRecord(event, at::globalContext().getCurrentCUDAStream()));
    }
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
    return (e.cpu_ns - cpu_ns)/(1000.0);
  }
  double cuda_elapsed_us(const Event & e) {
#ifdef WITH_CUDA
    if(!e.has_cuda() || !has_cuda()) {
      throw std::logic_error("Events were not recorded for CUDA");
    }
    PROFILER_CUDA_CHECK(cudaEventSynchronize(event));
    PROFILER_CUDA_CHECK(cudaEventSynchronize(e.event));
    float ms;
    PROFILER_CUDA_CHECK(cudaEventElapsedTime(&ms, event, e.event));
    return ms*1000.0;
#else
    throw std::logic_error("CUDA not enabled");
#endif
  }
  bool has_cuda() const {
    return event != nullptr;
  }

private:
  EventKind kind_;
  std::string name_;
  uint32_t thread_id_;
  uint64_t cpu_ns;
#ifdef WITH_CUDA
  cudaEvent_t event = nullptr;
#endif
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

extern bool profiling;
extern bool using_nvprof;
extern bool using_cuda;
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

inline void mark(std::string name) {
  if (using_nvprof) {
#ifdef WITH_CUDA
    nvtxMarkA(name.c_str());
#else
    throw std::logic_error("mark called with use_nvprof=True, but compiled without CUDA");
#endif
  } else {
    getEventList().record(EventKind::Mark, std::move(name), thread_id, using_cuda);
  }
}

inline void pushRange(std::string name) {
  if (using_nvprof) {
#ifdef WITH_CUDA
    nvtxRangePushA(name.c_str());
#else
    throw std::logic_error("pushRange called with use_nvprof=True, but compiled without CUDA");
#endif
  } else {
    getEventList().record(EventKind::PushRange, std::move(name), thread_id, using_cuda);
  }
}

inline void popRange() {
  if (using_nvprof) {
#ifdef WITH_CUDA
    nvtxRangePop();
#else
    throw std::logic_error("popRange called with use_nvprof=True, but compiled without CUDA");
#endif
  } else {
    getEventList().record(EventKind::PopRange, std::string(), thread_id, using_cuda);
  }
}

struct RecordFunction {
  explicit RecordFunction(Function *fn) {
    if (!profiling) return;
    pushFunctionRange(fn);
  }

  explicit RecordFunction(std::string name) {
    if (!profiling) return;
    pushRange(std::move(name));
  }

  explicit RecordFunction(const char *name) {
    if (!profiling) return;
    pushRange(name);
  }

  ~RecordFunction() {
    if (!profiling) return;
    popRange();
  }

  // Needed only because we don't have Function defined yet.
  void pushFunctionRange(Function *fn);
};

using thread_event_lists = std::vector<std::vector<Event>>;
// NOTE: changing profiler modes is **NOT THREAD SAFE**. You should ensure that
// there no autograd functions are being executed when these function are used.
void enableProfiler(bool use_nvprof, bool use_cuda);
thread_event_lists disableProfiler();

} // namespace profiler
}} // namespace torch::autograd

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
#include <forward_list>
#include <tuple>

namespace torch { namespace autograd {

struct Function;

namespace profiler {

constexpr inline std::size_t ceilToMultiple(std::size_t a, std::size_t b) {
  return ((a + b - 1) / b) * b;
}

enum class EventKind {
  Mark,
  PushRange,
  PopRange
};

// NOTE: we don't need a flag saying if an event is a kernel, because it's
// used only for the CPU-side perf recording.
using Event = std::tuple<std::string, uint64_t, EventKind>; // (name, time, kind)

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
extern bool using_cuda;
extern std::mutex all_event_lists_mutex;
extern std::list<std::shared_ptr<RangeEventList>> all_event_lists;
extern thread_local std::shared_ptr<RangeEventList> event_list;

inline RangeEventList& getEventList() {
  if (!event_list) {
    std::lock_guard<std::mutex> guard(all_event_lists_mutex);
    event_list = std::make_shared<RangeEventList>();
    all_event_lists.emplace_front(event_list);
  }
  return *event_list;
}

inline uint64_t getTime() {
  using namespace std::chrono;
  using clock = std::conditional<high_resolution_clock::is_steady, high_resolution_clock, steady_clock>::type;
  return duration_cast<nanoseconds>(clock::now().time_since_epoch()).count();
}

inline void mark(std::string name) {
  if (using_cuda) {
#ifdef WITH_CUDA
    nvtxMarkA(name.c_str());
#else
    throw std::logic_error("mark called with use_cuda=True, but compiled without CUDA");
#endif
  } else {
    getEventList().record(std::move(name), getTime(), EventKind::Mark);
  }
}

inline void pushRange(std::string name) {
  if (using_cuda) {
#ifdef WITH_CUDA
    nvtxRangePushA(name.c_str());
#else
    throw std::logic_error("pushRange called with use_cuda=True, but compiled without CUDA");
#endif
  } else {
    getEventList().record(std::move(name), getTime(), EventKind::PushRange);
  }
}

inline void popRange() {
  if (using_cuda) {
#ifdef WITH_CUDA
    nvtxRangePop();
#else
    throw std::logic_error("popRange called with use_cuda=True, but compiled without CUDA");
#endif
  } else {
    getEventList().record(std::string(), getTime(), EventKind::PopRange);
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
void enableProfiler(bool use_cuda);
thread_event_lists disableProfiler();

} // namespace profiler
}} // namespace torch::autograd

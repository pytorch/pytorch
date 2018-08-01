#pragma once

#ifdef USE_CUDA
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
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/cuda/cuda_check.h"
#ifdef USE_CUDA
#include "ATen/cuda/CUDAContext.h"
#include <cuda_runtime.h>
#endif

namespace torch { namespace autograd {

struct Function;

namespace profiler {

constexpr inline size_t ceilToMultiple(size_t a, size_t b) {
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
#ifdef USE_CUDA
    if(record_cuda) {
      TORCH_CUDA_CHECK(cudaGetDevice(&device_));
      TORCH_CUDA_CHECK(cudaEventCreate(&event));
      auto stream = at::cuda::getCurrentCUDAStream();
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
#ifdef USE_CUDA
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
#ifdef USE_CUDA
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
#ifdef USE_CUDA
  cudaEvent_t event = nullptr;
#endif
  int device_ = -1;
};

// a linked-list of fixed sized vectors, to avoid
// a std::vector resize from taking a large amount of time inside
// a profiling  event
struct RangeEventList {
  constexpr static size_t MB = 1024 * 1024;
  constexpr static size_t event_block_size = 16 * MB;
  constexpr static size_t num_block_elements =
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

TORCH_API RangeEventList& getEventList();
TORCH_API void mark(std::string name, bool include_cuda = true);
TORCH_API void pushRange(std::string name);
TORCH_API void popRange();

struct TORCH_API RecordFunction {
  explicit RecordFunction(Function* fn);

  explicit RecordFunction(std::string name);

  explicit RecordFunction(const char* name);

  ~RecordFunction();

  // Needed only because we don't have Function defined yet.
  void pushFunctionRange(Function *fn);
};

using thread_event_lists = std::vector<std::vector<Event>>;
// NOTE: changing profiler modes is **NOT THREAD SAFE**. You should ensure that
// there no autograd functions are being executed when these function are used.
TORCH_API void enableProfiler(ProfilerState new_state);
TORCH_API thread_event_lists disableProfiler();

} // namespace profiler
}} // namespace torch::autograd

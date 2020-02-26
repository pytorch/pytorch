#pragma once

#include <iostream>
#include <mutex>
#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include <sstream>
#include <forward_list>
#include <tuple>
#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#ifndef _WIN32
#include <ctime>
#endif

#include <torch/csrc/autograd/record_function.h>

typedef struct CUevent_st* CUDAEventStub;

namespace torch { namespace autograd {

struct Node;

namespace profiler {

TORCH_API uint16_t getThreadId();

struct TORCH_API CUDAStubs {
  virtual void record(int* device, CUDAEventStub* event, int64_t* cpu_ns) {
    fail();
  }
  virtual float elapsed(CUDAEventStub event, CUDAEventStub event2) {
    fail();
    return 0.f;
  }
  virtual void nvtxMarkA(const char* name) {
    fail();
  }
  virtual void nvtxRangePushA(const char* name) {
    fail();
  }
  virtual void nvtxRangePop() {
    fail();
  }
  virtual bool enabled() {
    return false;
  }
  virtual void onEachDevice(std::function<void(int)> op) {
    fail();
  }
  virtual void synchronize() {
    fail();
  }
  virtual ~CUDAStubs();

private:
  void fail() {
    AT_ERROR("CUDA used in profiler but not enabled.");
  }
};

TORCH_API void registerCUDAMethods(CUDAStubs* stubs);

constexpr inline size_t ceilToMultiple(size_t a, size_t b) {
  return ((a + b - 1) / b) * b;
}

#if (defined(__MACH__) && !defined(CLOCK_REALTIME)) || defined(C10_IOS)
#include <sys/time.h>
// clock_gettime is not implemented on older versions of OS X (< 10.12).
// If implemented, CLOCK_REALTIME will have already been defined.

// clock_gettime is only available on iOS 10.0 or newer. Unlike OS X, iOS can't rely on
// CLOCK_REALTIME, as it is defined no matter if clock_gettime is implemented or not
#endif

inline int64_t getTime() {
#ifdef _WIN32
  using namespace std::chrono;
  using clock = std::conditional<high_resolution_clock::is_steady, high_resolution_clock, steady_clock>::type;
  return duration_cast<nanoseconds>(clock::now().time_since_epoch()).count();
#elif (defined(__MACH__) && !defined(CLOCK_REALTIME)) || defined(C10_IOS)
  struct timeval now;
  gettimeofday(&now, NULL);
  return static_cast<int64_t>(now.tv_sec) * 1000000000 + static_cast<int64_t>(now.tv_usec) * 1000;
#else
  // clock_gettime is *much* faster than std::chrono implementation on Linux
  struct timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<int64_t>(t.tv_sec) * 1000000000 + static_cast<int64_t>(t.tv_nsec);
#endif
}

// Old GCC versions generate warnings incorrectly
// see https://stackoverflow.com/questions/2463113/g-c0x-enum-class-compiler-warnings
#ifndef _MSC_VER
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wattributes"
#endif
enum class TORCH_API ProfilerState {
    Disabled,
    CPU, // CPU-only profiling
    CUDA, // CPU + CUDA events
    NVTX,  // only emit NVTX markers
};

struct TORCH_API ProfilerConfig {
  ProfilerConfig(ProfilerState state, bool report_input_shapes)
      : state(state), report_input_shapes(report_input_shapes) {}
  ~ProfilerConfig();
  ProfilerState state;
  bool report_input_shapes;
};

enum class TORCH_API EventKind : uint16_t {
  Mark,
  PushRange,
  PopRange
};
#ifndef _MSC_VER
#  pragma GCC diagnostic pop
#endif

struct TORCH_API Event final {
  Event(
      EventKind kind,
      StringView name,
      uint16_t thread_id,
      bool record_cuda,
      std::vector<std::vector<int64_t>>&& shapes = {})
      : name_(std::move(name)),
        kind_(kind),
        thread_id_(thread_id),
        shapes_(shapes) {
    record(record_cuda);
  }

  void record(bool record_cuda);
  std::string kind() const {
    switch(kind_) {
      case EventKind::Mark: return "mark";
      case EventKind::PushRange: return "push";
      case EventKind::PopRange: return "pop";
    }
    throw std::runtime_error("unknown EventKind");
  }
  const char* name() const {
    return name_.str();
  }
  uint16_t thread_id() const {
    return thread_id_;
  }
  std::vector<std::vector<int64_t>> shapes() const {
    return shapes_;
  }
  double cpu_elapsed_us(const Event & e) {
    return (e.cpu_ns_ - cpu_ns_)/(1000.0);
  }
  double cuda_elapsed_us(const Event & e);
  bool has_cuda() const {
    return event != nullptr;
  }
  int device() const {
    return device_;
  }
private:
  // signed to allow for negative intervals, initialized for safety.
  int64_t cpu_ns_ = 0;
  StringView name_;
  EventKind kind_;
  uint16_t thread_id_;
  std::vector<std::vector<int64_t>> shapes_;
  int device_ = -1;
  struct CUevent_st* event = nullptr;
};

// a linked-list of fixed sized vectors, to avoid
// a std::vector resize from taking a large amount of time inside
// a profiling  event
struct RangeEventList {
  // This mutex is used to serialize access when different threads are writing
  // to the same instance of RangeEventList.
  std::mutex mutex_;
  constexpr static size_t MB = 1024 * 1024;
  constexpr static size_t event_block_size = 16 * MB;
  constexpr static size_t num_block_elements =
    event_block_size / ceilToMultiple(sizeof(Event), alignof(Event));
  static_assert(sizeof(Event[num_block_elements]) <= event_block_size,
                "num_block_elements is calculated incorrectly");
  using block_type = std::vector<Event>;

// allocBlock() assumes that mutex_ is held when called, in order to prevent
  // multiple threads' block writes stomping over each other.
  void allocBlock() {
    blocks.emplace_front();
    auto & new_block = blocks.front();
    new_block.reserve(num_block_elements);
    // Materialize all pages in the new block to release jitter when recording events.
    const char * const end_ptr = reinterpret_cast<char*>(new_block.data() + num_block_elements);
    for (volatile const char * ptr = reinterpret_cast<char*>(new_block.data());
         ptr < end_ptr; ptr += 4 * 1024) {
      (*ptr);
    }
  }

  template<typename... Args>
  void record(Args&&... args) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (blocks.empty() || blocks.front().size() == num_block_elements) {
      allocBlock();
    }
    blocks.front().emplace_back(std::forward<Args>(args)...);
  }

  std::vector<Event> consolidate() {
    std::lock_guard<std::mutex> guard(mutex_);
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

TORCH_API RangeEventList& getEventList();
TORCH_API void mark(std::string name, bool include_cuda = true);
TORCH_API void pushRange(std::string name);
TORCH_API void popRange(const StringView& name = StringView(""));

using thread_event_lists = std::vector<std::vector<Event>>;
// NOTE: changing profiler modes is **NOT THREAD SAFE**. You should ensure that
// there no autograd functions are being executed when these function are used.
TORCH_API void enableProfiler(ProfilerConfig);
TORCH_API thread_event_lists disableProfiler();
TORCH_API bool profilerEnabled();


// Usage:
//   {
//     RecordProfile guard("filename.trace");
//     // code you want to profile
//   }
// Then open filename.trace in chrome://tracing
struct TORCH_API RecordProfile {
  RecordProfile(std::ostream& out);
  RecordProfile(const std::string& filename);

  ~RecordProfile();
private:
  void init();
  std::unique_ptr<std::ofstream> file_;
  std::ostream& out_;
  void processEvents(const std::vector<Event*>& events);
};


} // namespace profiler
}} // namespace torch::autograd

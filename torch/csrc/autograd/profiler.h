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
#if defined(C10_IOS) && defined(C10_MOBILE)
#include <sys/time.h> // for gettimeofday()
#endif

#include <ATen/record_function.h>

typedef struct CUevent_st* CUDAEventStub;

namespace torch { namespace autograd {

struct Node;

namespace profiler {

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

inline int64_t getTime() {
#if defined(C10_IOS) && defined(C10_MOBILE)
// clock_gettime is only available on iOS 10.0 or newer. Unlike OS X, iOS can't rely on
// CLOCK_REALTIME, as it is defined no matter if clock_gettime is implemented or not
  struct timeval now;
  gettimeofday(&now, NULL);
  return static_cast<int64_t>(now.tv_sec) * 1000000000 + static_cast<int64_t>(now.tv_usec) * 1000;
#elif defined(_WIN32) || defined(__MACH__)
  using namespace std::chrono;
  using clock = std::conditional<high_resolution_clock::is_steady, high_resolution_clock, steady_clock>::type;
  return duration_cast<nanoseconds>(clock::now().time_since_epoch()).count();
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
  ProfilerConfig(
      ProfilerState state,
      bool report_input_shapes,
      bool profile_memory)
      : state(state),
        report_input_shapes(report_input_shapes),
        profile_memory(profile_memory) {}
  ~ProfilerConfig();
  ProfilerState state;
  bool report_input_shapes;
  bool profile_memory;
};

enum class TORCH_API EventKind : uint16_t {
  Mark,
  PushRange,
  PopRange,
  MemoryAlloc,
};
#ifndef _MSC_VER
#  pragma GCC diagnostic pop
#endif

struct TORCH_API Event final {
  Event(
      EventKind kind,
      at::StringView name,
      uint16_t thread_id,
      bool record_cuda,
      at::RecordFunctionHandle handle = 0,
      std::vector<std::vector<int64_t>>&& shapes = {}, int node_id = -1)
      : name_(std::move(name)),
        kind_(kind),
        thread_id_(thread_id),
        handle_(handle),
        shapes_(shapes), node_id_(node_id) {
    record(record_cuda);
  }

  void setNodeId(int node_id) {
    node_id_ = node_id;
  }

  // Returns IValues corresponding to event structure, to be used for
  // serialization.
  at::IValue getEventIValues() const {
    c10::impl::GenericList eventIValueList(at::AnyType::get());
    eventIValueList.push_back(at::IValue(static_cast<int64_t>(kind_)));
    eventIValueList.push_back(at::IValue(std::string(name_.str())));
    eventIValueList.push_back(at::IValue(thread_id_));
    eventIValueList.push_back(at::IValue(static_cast<double>(handle_)));
    eventIValueList.push_back(at::IValue(node_id_));
    return at::IValue(eventIValueList);
  }

  // Reconstructs an event from IValues given by getEventIValues.
  static Event fromIValue(at::IValue eventIValue) {
    TORCH_INTERNAL_ASSERT(
        eventIValue.isList(),
        "Expected IValue to contain type c10::impl::GenericList");
    auto ivalues = eventIValue.toList();

    Event evt(
      static_cast<EventKind>(ivalues.get(0).toInt()),
      at::StringView(ivalues.get(1).toStringRef()),
      ivalues.get(2).toInt(),
      false, // TODO: record_cuda
      static_cast<at::RecordFunctionHandle>(ivalues.get(3).toDouble()),
      {}, // TODO: record shapes
      ivalues.get(4).toInt()
    );
    return evt;
  }

  void record(bool record_cuda);
  std::string kind() const {
    switch(kind_) {
      case EventKind::Mark: return "mark";
      case EventKind::PushRange: return "push";
      case EventKind::PopRange: return "pop";
      case EventKind::MemoryAlloc: return "memory_alloc";
    }
    throw std::runtime_error("unknown EventKind");
  }

  // Get enum kind of this event.
  EventKind eventKind() const {
    return kind_;
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
    return cuda_event != nullptr;
  }
  int device() const {
    return device_;
  }

  void updateMemoryStats(int64_t alloc_size, c10::Device device) {
    if (device.type() == c10::DeviceType::CUDA ||
        device.type() == c10::DeviceType::HIP) {
      cuda_memory_usage_ = alloc_size;
    } else if (device.type() == c10::DeviceType::CPU ||
        device.type() == c10::DeviceType::MKLDNN ||
        device.type() == c10::DeviceType::IDEEP) {
      cpu_memory_usage_ = alloc_size;
    } else {
      LOG(WARNING) << "Unsupported memory profiling device: " << device;
    }
  }

  int64_t cpu_memory_usage() const {
    return cpu_memory_usage_;
  }

  int64_t cuda_memory_usage() const {
    return cuda_memory_usage_;
  }

  at::RecordFunctionHandle handle() const {
    return handle_;
  }

  // Node ID corresponding to this event.
  int node_id( )const {
    return node_id_;
  }

private:
  // signed to allow for negative intervals, initialized for safety.
  int64_t cpu_ns_ = 0;
  at::StringView name_;
  EventKind kind_;
  uint16_t thread_id_;
  at::RecordFunctionHandle handle_ {0};
  std::vector<std::vector<int64_t>> shapes_;
  int64_t cpu_memory_usage_ = 0;
  int64_t cuda_memory_usage_ = 0;
  int device_ = -1;
  struct CUevent_st* cuda_event = nullptr;
  int node_id_ = 0;
};

// a linked-list of fixed sized vectors, to avoid
// a std::vector resize from taking a large amount of time inside
// a profiling  event
struct RangeEventList {
  RangeEventList() {
    events_.reserve(kReservedCapacity);
  }

  template<typename... Args>
  void record(Args&&... args) {
    std::lock_guard<std::mutex> guard(mutex_);
    events_.emplace_back(std::forward<Args>(args)...);
  }

  std::vector<Event> consolidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Event> result;
    result.insert(
        result.begin(),
        std::make_move_iterator(events_.begin()),
        std::make_move_iterator(events_.end()));
    events_.erase(events_.begin(), events_.end());
    return result;
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
  }

 private:
  // This mutex is used to serialize access when different threads are writing
  // to the same instance of RangeEventList.
  std::mutex mutex_;
  std::vector<Event> events_;

  static const size_t kReservedCapacity = 1024;
};

using thread_event_lists = std::vector<std::vector<Event>>;
// NOTE: profiler mode is thread local, with automatic propagation
// across thread boundary (e.g. at::launch tasks)
TORCH_API void enableProfiler(const ProfilerConfig&);
TORCH_API thread_event_lists disableProfiler();
// adds profiledEvents to the current thread local recorded events. Each event
// will be marked with node ID given by fromNodeId.
TORCH_API void addEventList(std::vector<Event> profiledEvents, int fromNodeId = -1);
// Returns if the profiler is currently enabled in the current thread.
TORCH_API bool profilerEnabled();
// Retrieve the thread_local ProfilerConfig.
TORCH_API ProfilerConfig getProfilerConfig();
// Writes profiled events to a stream.
TORCH_API void writeProfilerEventsToStream(std::ostream& out, const std::vector<Event*>& events);

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

// A guard that enables the profiler, taking in an optional callback to process
// the results
// Usage:
// {
//   TLSProfilerGuard g([](thread_event_lists profilerResults) {
//     // process profilerResuls
//   });
//   Code to profile
// }
struct TORCH_API TLSProfilerGuard {
  explicit TLSProfilerGuard(
      const ProfilerConfig& cfg, c10::optional<std::function<void(thread_event_lists)>> resultCallback =
          c10::nullopt)
      : cb_(std::move(resultCallback)) {
    enableProfiler(
        cfg);
  };
  ~TLSProfilerGuard() {
    thread_event_lists event_lists = disableProfiler();
    if (cb_) {
      (*cb_)(std::move(event_lists));
    }
  }

 private:
  c10::optional<std::function<void(thread_event_lists)>> cb_;
};

} // namespace profiler
}} // namespace torch::autograd

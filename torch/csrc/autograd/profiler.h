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
#include <c10/core/DeviceType.h>

#ifdef USE_KINETO
#include "libkineto.h"
#endif

struct CUevent_st;
typedef std::shared_ptr<CUevent_st> CUDAEventStub;

namespace torch { namespace autograd {

struct Node;

namespace profiler {

struct TORCH_API CUDAStubs {
  virtual void record(int* device, CUDAEventStub* event, int64_t* cpu_ns) {
    fail();
  }
  virtual float elapsed(const CUDAEventStub* event, const CUDAEventStub* event2) {
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

// A struct to control settings of disableProfiler options.
struct TORCH_API ProfilerDisableOptions {
  ProfilerDisableOptions() = default;
  ProfilerDisableOptions(bool shouldCleanupTLSState, bool shouldConsolidate)
      : cleanupTLSState(shouldCleanupTLSState),
        consolidate(shouldConsolidate) {}
  // Whether we should clean up profiler states that are thread local, such as
  // ThreadLocalDebugInfo and thread local RecordFunction callbacks.
  bool cleanupTLSState = true;
  // Whether we should consolidate all currently recorded profiled events. If
  // false, will not consolidate and other threads can continue to write to the
  // event lists.
  bool consolidate = true;
};

enum class C10_API_ENUM ProfilerState {
  Disabled = 0,
  CPU, // CPU-only profiling
  CUDA, // CPU + CUDA events
  NVTX,  // only emit NVTX markers
  KINETO, // use libkineto
  NUM_PROFILER_STATES, // must be the last one
};

enum class C10_API_ENUM ActivityType {
  CPU = 0,
  // CUDA_RUNTIME, // CUDA host events
  CUDA, // CUDA kernels
  NUM_KINETO_ACTIVITIES, // must be the last one
};

struct TORCH_API ProfilerConfig {
  ProfilerConfig(
      ProfilerState state,
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false)
      : state(state),
        report_input_shapes(report_input_shapes),
        profile_memory(profile_memory),
        with_stack(with_stack) {}
  ~ProfilerConfig();
  ProfilerState state;
  bool report_input_shapes;
  bool profile_memory;
  bool with_stack;

  // Returns IValues corresponding to ProfilerConfig struct, to be used for
  // serialization.
  at::IValue toIValue() const;

  // Reconstructs a ProfilerConfig from IValues given by toIValue.
  static ProfilerConfig fromIValue(const at::IValue& profilerConfigIValue);
};

enum class C10_API_ENUM EventKind : uint16_t {
  Mark,
  PushRange,
  PopRange,
  MemoryAlloc,
};

// To be deprecated, once we switch to Kineto profiling
struct TORCH_API LegacyEvent {
  LegacyEvent(
      EventKind kind,
      at::StringView name,
      uint16_t thread_id,
      bool record_cuda,
      at::RecordFunctionHandle handle = 0,
      std::vector<std::vector<int64_t>>&& shapes = {},
      int node_id = -1)
      : name_(std::move(name)),
        kind_(kind),
        thread_id_(thread_id),
        handle_(handle),
        shapes_(shapes),
        node_id_(node_id) {
    record(record_cuda);
  }

  // Constructor to be used in conjunction with LegacyEvent::fromIValue.
  LegacyEvent(
      EventKind kind,
      at::StringView name,
      uint16_t thread_id,
      at::RecordFunctionHandle handle,
      std::vector<std::vector<int64_t>>&& shapes,
      int node_id,
      bool is_remote,
      int64_t cpu_memory_usage,
      int64_t cpu_ns,
      bool cuda_recorded,
      int64_t cuda_memory_usage = 0,
      int device = -1,
      double cuda_us = -1)
      : cpu_ns_(cpu_ns),
        name_(std::move(name)),
        kind_(kind),
        thread_id_(thread_id),
        handle_(handle),
        shapes_(shapes),
        cpu_memory_usage_(cpu_memory_usage),
        cuda_memory_usage_(cuda_memory_usage),
        device_(device),
        node_id_(node_id),
        is_remote_(is_remote),
        cuda_us_(cuda_us) {
    // Sanity check values that were deserialized
    TORCH_INTERNAL_ASSERT(cpu_ns_ > 0);
    if (cuda_recorded) {
      TORCH_INTERNAL_ASSERT(device_ >= 0);
      TORCH_INTERNAL_ASSERT(cuda_us_ >= 0);
    }
  }

  // Returns IValues corresponding to event structure, to be used for
  // serialization.
  at::IValue toIValue() const;

  // Reconstructs an event from IValues given by toIValue.
  static LegacyEvent fromIValue(const at::IValue& eventIValue);

  void record(bool record_cuda);

  std::string kindStr() const {
    switch (kind_) {
      case EventKind::Mark: return "mark";
      case EventKind::PushRange: return "push";
      case EventKind::PopRange: return "pop";
      case EventKind::MemoryAlloc: return "memory_alloc";
    }
    throw std::runtime_error("unknown event kind");
  }

  const char* name() const {
    return name_.str();
  }

  uint64_t threadId() const {
    return thread_id_;
  }

  std::vector<std::vector<int64_t>> shapes() const {
    return shapes_;
  }

  double cpuElapsedUs(const LegacyEvent& e) const {
    return (e.cpu_ns_ - cpu_ns_)/(1000.0);
  }

  double cpuUs() const {
    return cpu_ns_ / (1000.0);
  }

  double cudaElapsedUs(const LegacyEvent& e) const;

  bool hasCuda() const {
    return cuda_event != nullptr || (isRemote() && device_ != -1);
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

  int64_t cpuMemoryUsage() const {
    return cpu_memory_usage_;
  }

  int64_t cudaMemoryUsage() const {
    return cuda_memory_usage_;
  }

  at::RecordFunctionHandle handle() const {
    return handle_;
  }

  // Node ID corresponding to this event.
  int nodeId( ) const {
    return node_id_;
  }

  // Set Node ID on this event.
  void setNodeId(int node_id) {
    node_id_ = node_id;
  }

  void setName(at::StringView newName_) {
    name_ = std::move(newName_);
  }

  bool isRemote() const {
    return is_remote_;
  }

  void setCudaUs(int64_t cuda_us) {
    cuda_us_ = cuda_us;
  }

  void setSequenceNr(int64_t sequence_nr) {
    sequence_nr_ = sequence_nr;
  }

  int64_t sequenceNr() const {
    return sequence_nr_;
  }

  void setCorrelationId(uint64_t correlation_id) {
    correlation_id_ = correlation_id;
  }

  uint64_t correlationId() const {
    return correlation_id_;
  }

  const std::vector<std::string>& stack() const {
    return stack_;
  }

  void setStack(const std::vector<std::string>& stack) {
    stack_ = stack;
  }

  uint64_t fwdThreadId() const {
    return fwd_thread_id_;
  }

  void setFwdThreadId(uint64_t fwd_thread_id) {
    fwd_thread_id_ = fwd_thread_id;
  }

  uint8_t scope() const {
    return scope_;
  }

  void setScope(uint8_t scope) {
    scope_ = scope;
  }

 private:
  // signed to allow for negative intervals, initialized for safety.
  int64_t cpu_ns_ = 0;
  at::StringView name_;
  EventKind kind_;
  uint64_t thread_id_;
  uint64_t fwd_thread_id_;
  at::RecordFunctionHandle handle_ {0};
  std::vector<std::vector<int64_t>> shapes_;
  int64_t cpu_memory_usage_ = 0;
  int64_t cuda_memory_usage_ = 0;
  int device_ = -1;
  CUDAEventStub cuda_event = nullptr;
  int node_id_ = 0;
  bool is_remote_ = false;
  int64_t cuda_us_ = -1;
  int64_t sequence_nr_ = -1;

  std::vector<std::string> stack_;
  uint8_t scope_;
  uint64_t correlation_id_;
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

  std::vector<LegacyEvent> consolidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<LegacyEvent> result;
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
  std::vector<LegacyEvent> events_;

  static const size_t kReservedCapacity = 1024;
};

// NOTE: profiler mode is thread local, with automatic propagation
// across thread boundary (e.g. at::launch tasks)
TORCH_API void enableProfilerLegacy(const ProfilerConfig&);
using thread_event_lists = std::vector<std::vector<LegacyEvent>>;
TORCH_API thread_event_lists disableProfilerLegacy(c10::optional<ProfilerDisableOptions> profilerDisableOptions = c10::nullopt);

// adds profiledEvents to the current thread local recorded events. Each event
// will be marked with node ID given by fromNodeId.
TORCH_API void addEventList(std::vector<LegacyEvent>&& profiledEvents);
// Returns if the profiler is currently enabled in the current thread.
TORCH_API bool profilerEnabled();
// Retrieve the thread_local ProfilerConfig.
TORCH_API ProfilerConfig getProfilerConfig();
// Writes profiled events to a stream.
TORCH_API void writeProfilerEventsToStream(std::ostream& out, const std::vector<LegacyEvent*>& events);

#ifdef USE_KINETO
struct TORCH_API KinetoEvent {
  int64_t threadId() const {
    return thread_id_;
  }

  c10::DeviceType deviceType() const {
    return device_type_;
  }

  int64_t fwdThreadId() const {
    return fwd_thread_id_;
  }

  const std::vector<std::vector<int64_t>>& shapes() const {
    return shapes_;
  }

  int64_t sequenceNr() const {
    return sequence_nr_;
  }

  const std::vector<std::string>& stack() const {
    return stack_;
  }

  uint8_t scope() const {
    return scope_;
  }

  KinetoEvent& threadId(int64_t thread_id) {
    thread_id_ = thread_id;
    return *this;
  }

  KinetoEvent& deviceType(c10::DeviceType device_type) {
    device_type_ = device_type;
    return *this;
  }

  KinetoEvent& fwdThreadId(int64_t fwd_thread_id) {
    fwd_thread_id_ = fwd_thread_id;
    return *this;
  }

  KinetoEvent& shapes(const std::vector<std::vector<int64_t>>& shapes) {
    shapes_ = shapes;
    return *this;
  }

  KinetoEvent& sequenceNr(int64_t sequence_nr) {
    sequence_nr_ = sequence_nr_;
    return *this;
  }

  KinetoEvent& stack(const std::vector<std::string>& st) {
    stack_ = st;
    return *this;
  }

  KinetoEvent& scope(uint8_t scope_id) {
    scope_id_ = scope_id;
    return *this;
  }

  // Kineto fields

  KinetoEvent& activity(const libkineto::TraceActivity& activity) {
    name_ = activity.name();
    deviceIndex_ = activity.deviceId();
    startUs_ = activity.timestamp();
    durationUs_ = activity.duration();
    correlationId_ = activity.correlationId();
    return *this;
  }

  std::string name() const {
    return name_;
  }

  uint64_t deviceIndex() const {
    return deviceIndex_;
  }

  uint64_t startUs() const {
    return startUs_;
  }

  uint64_t durationUs() const {
    return durationUs_;
  }

  uint64_t correlationId() const {
    return correlationId_;
  }

  KinetoEvent& correlationId(uint64_t correlationId)  {
    correlationId_ = correlationId;
    return *this;
  }

 private:
  int64_t thread_id_ = 0;
  c10::DeviceType device_type_ = c10::DeviceType::CPU,
  int64_t fwd_thread_id_ = 0;
  std::vector<std::vector<int64_t>> shapes_;
  int64_t sequence_nr_ = 0;
  std::vector<std::string> stack_;
  uint8_t scope_ = 0;

  std::string name_;
  uint64_t deviceIndex_ = 0;
  uint64_t startUs_ = 0;
  uint64_t durationUs_ = 0;
  uint64_t correlationId_ = 0;
};

struct TORCH_API ProfilerResult {
  ProfilerResult(
      const std::vector<std::vector<KinetoEvent>>& events,
      const thread_event_lists& legacy_events)
    : events_(events), legacy_events_(legacy_events) {}

  const std::vector<std::vector<KinetoEvent>> events() const {
    return events_;
  }

  const thread_event_lists& legacy_events() const {
    return legacy_events_;
  }

 private:
  std::vector<std::vector<KinetoEvent>> events_;
  thread_event_lists legacy_events_; // tensor mem alloc, start/stop
};
TORCH_API void enableProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities);
TORCH_API ProfilerResult disableProfiler();

TORCH_API void prepareProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities);
#endif // USE_KINETO

TORCH_API bool kinetoAvailable();

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
  void processEvents(const std::vector<LegacyEvent*>& events);
};

// A guard that enables the profiler, taking in an optional callback to process
// the results
// Usage:
// {
//   TLSProfilerGuard g([](thread_event_lists profilerResults) {
//     // process profilerResults
//   });
//   Code to profile
// }
struct TORCH_API TLSProfilerGuard {
  explicit TLSProfilerGuard(
      const ProfilerConfig& cfg,
      c10::optional<std::function<void(const thread_event_lists&)>>
          resultCallback = c10::nullopt,
      c10::optional<ProfilerDisableOptions> profilerDisableOptions =
          c10::nullopt)
      : cb_(std::move(resultCallback)),
        profilerDisableOptions_(std::move(profilerDisableOptions)) {
    enableProfiler(cfg);
  }
  ~TLSProfilerGuard() {
    thread_event_lists event_lists = disableProfiler(profilerDisableOptions_);
    if (cb_) {
      try {
        (*cb_)(event_lists);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Got error processing profiler events: " << e.what();
      }
    }
  }

 private:
  c10::optional<std::function<void(const thread_event_lists&)>> cb_;
  const c10::optional<ProfilerDisableOptions> profilerDisableOptions_;
};

} // namespace profiler
}} // namespace torch::autograd

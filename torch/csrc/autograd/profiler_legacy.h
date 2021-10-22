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
#include <torch/csrc/autograd/profiler_utils.h>
#ifndef _WIN32
#include <ctime>
#endif
#if defined(C10_IOS) && defined(C10_MOBILE)
#include <sys/time.h> // for gettimeofday()
#endif

#include <ATen/record_function.h>

#include <torch/csrc/jit/frontend/source_range.h>

struct CUevent_st;
typedef std::shared_ptr<CUevent_st> CUDAEventStub;

namespace torch { namespace autograd {

struct Node;

namespace profiler {

struct TORCH_API CUDAStubs {
  virtual void record(int* device, CUDAEventStub* event, int64_t* cpu_ns) const {
    fail();
  }
  virtual float elapsed(const CUDAEventStub* event, const CUDAEventStub* event2) const {
    fail();
    return 0.f;
  }
  virtual void nvtxMarkA(const char* name) const {
    fail();
  }
  virtual void nvtxRangePushA(const char* name) const {
    fail();
  }
  virtual void nvtxRangePop() const {
    fail();
  }
  virtual bool enabled() const {
    return false;
  }
  virtual void onEachDevice(std::function<void(int)> op) const {
    fail();
  }
  virtual void synchronize() const {
    fail();
  }
  virtual ~CUDAStubs();

private:
  void fail() const {
    AT_ERROR("CUDA used in profiler but not enabled.");
  }
};

TORCH_API void registerCUDAMethods(CUDAStubs* stubs);
TORCH_API const CUDAStubs* cudaStubs();

constexpr inline size_t ceilToMultiple(size_t a, size_t b) {
  return ((a + b - 1) / b) * b;
}

inline int64_t getTime(bool allow_monotonic = false) {
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
  auto mode = CLOCK_REALTIME;
  if (allow_monotonic) {
    mode = CLOCK_MONOTONIC;
  }
  clock_gettime(mode, &t);
  return static_cast<int64_t>(t.tv_sec) * 1000000000 + static_cast<int64_t>(t.tv_nsec);
#endif
}

enum class C10_API_ENUM EventKind : uint16_t {
  Mark,
  PushRange,
  PopRange,
  MemoryAlloc,
};

// To be deprecated, once we switch to Kineto profiling
struct TORCH_API LegacyEvent {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  LegacyEvent(
      EventKind kind,
      at::StringView name,
      uint16_t thread_id,
      bool record_cuda,
      at::RecordFunctionHandle handle = 0,
      std::vector<std::vector<int64_t>>&& shapes = {},
      int node_id = -1,
      bool is_async = false)
      : name_(std::move(name)),
        kind_(kind),
        thread_id_(thread_id),
        handle_(handle),
        shapes_(shapes),
        node_id_(node_id),
        is_async_(is_async) {
    record(record_cuda);
  }

  // Constructor to be used in conjunction with LegacyEvent::fromIValue.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
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

  EventKind kind() const {
    return kind_;
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
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
    return (e.cpu_ns_ - cpu_ns_)/(1000.0);
  }

  void setCpuUs(int64_t cpu_us) {
    cpu_ns_ = cpu_us * 1000.0;
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
    if (device.is_cuda() ||
        device.type() == c10::DeviceType::HIP) {
      cuda_memory_usage_ = alloc_size;
    } else if (device.is_cpu() ||
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

  const std::unordered_map<std::string, c10::IValue>& extraArgs() const {
    return extra_args_;
  }

  void setExtraArgs(std::unordered_map<std::string, c10::IValue>&& save_args) {
    extra_args_ = std::move(save_args);
  }

  uint64_t flops() {
    return flops_;
  }

  bool isAsync() {
    return is_async_;
  }

  void setFlops(uint64_t flops) {
    flops_ = flops;
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
  bool is_async_ = false;

  std::vector<std::string> stack_;
  uint8_t scope_;
  uint64_t correlation_id_;
  // Extra arguments for computing op flops
  std::unordered_map<std::string, c10::IValue> extra_args_;
  uint64_t flops_ = 0;
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

std::string getNvtxStr(
    const at::StringView& name,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes);

enum class C10_API_ENUM ProfilerState {
  Disabled = 0,
  CPU, // CPU-only profiling
  CUDA, // CPU + CUDA events
  NVTX,  // only emit NVTX markers
  KINETO, // use libkineto
  KINETO_GPU_FALLBACK, // use CUDA events when CUPTI is not available
  NUM_PROFILER_STATES, // must be the last one
};

struct TORCH_API ProfilerConfig {
  explicit ProfilerConfig(
      ProfilerState state,
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false)
      : state(state),
        report_input_shapes(report_input_shapes),
        profile_memory(profile_memory),
        with_stack(with_stack),
        with_flops(with_flops),
        with_modules(with_modules) {}
  ~ProfilerConfig() = default;
  ProfilerState state;
  bool report_input_shapes;
  bool profile_memory;
  bool with_stack;
  bool with_flops;
  bool with_modules;

  // Returns IValues corresponding to ProfilerConfig struct, to be used for
  // serialization.
  at::IValue toIValue() const;

  // Reconstructs a ProfilerConfig from IValues given by toIValue.
  static ProfilerConfig fromIValue(const at::IValue& profilerConfigIValue);
};

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

// A guard that enables the legacy profiler, taking in an optional callback to process
// the results
// Usage:
// {
//   TLSLegacyProfilerGuard g([](thread_event_lists profilerResults) {
//     // process profilerResults
//   });
//   Code to profile
// }
struct TORCH_API TLSLegacyProfilerGuard {
  explicit TLSLegacyProfilerGuard(
      const ProfilerConfig& cfg,
      c10::optional<std::function<void(const thread_event_lists&)>>
          resultCallback = c10::nullopt,
      c10::optional<ProfilerDisableOptions> profilerDisableOptions =
          c10::nullopt)
      : cb_(std::move(resultCallback)),
        // NOLINTNEXTLINE(performance-move-const-arg)
        profilerDisableOptions_(std::move(profilerDisableOptions)) {
    enableProfilerLegacy(cfg);
  }
  ~TLSLegacyProfilerGuard() {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    thread_event_lists event_lists = disableProfilerLegacy(profilerDisableOptions_);
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

struct TORCH_API FileLineFunc {
  std::string filename;
  size_t line;
  std::string funcname;
};
TORCH_API std::vector<FileLineFunc> prepareCallstack(const std::vector<jit::StackEntry>& cs);
TORCH_API std::vector<std::string> callstackStr(const std::vector<FileLineFunc>& cs);
TORCH_API std::vector<std::vector<int64_t>> inputSizes(const at::RecordFunction& fn);

struct TORCH_API ProfilerThreadLocalState : public c10::MemoryReportingInfoBase {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit ProfilerThreadLocalState(const ProfilerConfig& config)
      : config_(config), remoteProfiledEvents_{c10::nullopt} {}
  ~ProfilerThreadLocalState() override = default;

  const ProfilerConfig& config() const;

  thread_event_lists consolidate();

  void mark(std::string name, bool include_cuda = true);

  void setOrAddRemoteProfiledEvents(
      std::vector<LegacyEvent>&& remoteProfiledEvents);

  void pushRange(
      const at::RecordFunction& fn,
      const bool record_cuda,
      std::vector<std::vector<int64_t>>&& shapes = {});

  void popRange(const at::RecordFunction& fn, const bool record_cuda);

  void setCallbackHandle(at::CallbackHandle handle) {
    handle_ = handle;
  }

  at::CallbackHandle callbackHandle() const {
    return handle_;
  }

  bool hasCallbackHandle() {
    return handle_ > 0;
  }

  void reportMemoryUsage(
      void* /* unused */,
      int64_t alloc_size,
      int64_t /* total_allocated, unused for legacy */,
      int64_t /* total_reserved, unused for legacy */,
      c10::Device device) override;

  bool memoryProfilingEnabled() const override;

 protected:
  RangeEventList& getEventList(int64_t thread_id = -1);

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex state_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RangeEventList>>
      // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
      event_lists_map_;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  at::CallbackHandle handle_ = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  c10::optional<std::vector<std::vector<LegacyEvent>>> remoteProfiledEvents_;
};


} // namespace profiler
}} // namespace torch::autograd

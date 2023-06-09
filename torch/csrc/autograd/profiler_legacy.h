#pragma once

#include <cstdint>
#include <forward_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace autograd {

struct Node;

namespace profiler {

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
      case EventKind::Mark:
        return "mark";
      case EventKind::PushRange:
        return "push";
      case EventKind::PopRange:
        return "pop";
      case EventKind::MemoryAlloc:
        return "memory_alloc";
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
    return static_cast<double>(e.cpu_ns_ - cpu_ns_) / (1000.0);
  }

  void setCpuUs(int64_t cpu_us) {
    cpu_ns_ = static_cast<double>(cpu_us) * 1000.0;
  }

  double cpuUs() const {
    return static_cast<double>(cpu_ns_) / (1000.0);
  }

  double cudaElapsedUs(const LegacyEvent& e) const;

  bool hasCuda() const {
    return cuda_event != nullptr || (isRemote() && device_ != -1);
  }

  int device() const {
    return device_;
  }

  void updateMemoryStats(int64_t alloc_size, c10::Device device) {
    if (device.is_cuda() || device.type() == c10::DeviceType::HIP) {
      cuda_memory_usage_ = alloc_size;
    } else if (
        device.is_cpu() || device.type() == c10::DeviceType::MKLDNN ||
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
  int nodeId() const {
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
  at::RecordFunctionHandle handle_{0};
  std::vector<std::vector<int64_t>> shapes_;
  int64_t cpu_memory_usage_ = 0;
  int64_t cuda_memory_usage_ = 0;
  int device_ = -1;
  torch::profiler::impl::ProfilerVoidEventStub cuda_event = nullptr;
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
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,modernize-use-equals-default)
  RangeEventList() {
    events_.reserve(kReservedCapacity);
  }

  template <typename... Args>
  void record(Args&&... args) {
    std::lock_guard<std::mutex> guard(mutex_);
    events_.emplace_back(std::forward<Args>(args)...);
  }

  std::vector<LegacyEvent> consolidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
TORCH_API void enableProfilerLegacy(
    const torch::profiler::impl::ProfilerConfig&);
using thread_event_lists = std::vector<std::vector<LegacyEvent>>;
TORCH_API thread_event_lists disableProfilerLegacy(
    c10::optional<ProfilerDisableOptions> profilerDisableOptions =
        c10::nullopt);

// adds profiledEvents to the current thread local recorded events. Each event
// will be marked with node ID given by fromNodeId.
TORCH_API void addEventList(std::vector<LegacyEvent>&& profiledEvents);
// Writes profiled events to a stream.
TORCH_API void writeProfilerEventsToStream(
    std::ostream& out,
    const std::vector<LegacyEvent*>& events);

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

// A guard that enables the legacy profiler, taking in an optional callback to
// process the results Usage:
// {
//   TLSLegacyProfilerGuard g([](thread_event_lists profilerResults) {
//     // process profilerResults
//   });
//   Code to profile
// }
struct TORCH_API TLSLegacyProfilerGuard {
  explicit TLSLegacyProfilerGuard(
      const torch::profiler::impl::ProfilerConfig& cfg,
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
    thread_event_lists event_lists =
        disableProfilerLegacy(profilerDisableOptions_);
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
} // namespace autograd
} // namespace torch

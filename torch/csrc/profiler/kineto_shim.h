#pragma once

#include <memory>
#include <string>

// Skip Kineto dependency on mobile unless explicitly asked for.
// When is it explicitly asked for?
//   KinetoEdgeCPUProfiler uses KinetoProfiler for cpu
//   event profiling. This has a dependency on cpu only libkineto
#if defined(USE_KINETO) && defined(C10_MOBILE) && \
    !defined(EDGE_PROFILER_USE_KINETO)
#undef USE_KINETO
#endif

#include <ActivityType.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/api.h>

#ifdef USE_KINETO
// Forward declarations so we don't have to include `libkineto.h` in a header.
namespace libkineto {
class GenericTraceActivity;
struct CpuTraceBuffer;
class ActivityTraceInterface;
} // namespace libkineto
#endif

namespace torch {
namespace profiler {

#ifdef USE_KINETO
constexpr bool kKinetoAvailable{true};
#else
constexpr bool kKinetoAvailable{false};
#endif

namespace impl {
namespace kineto {

// ----------------------------------------------------------------------------
// -- Interface (Does not require Kineto) -------------------------------------
// ----------------------------------------------------------------------------
struct DeviceAndResource {
  int32_t device;
  int32_t resource;
};
const DeviceAndResource kineto_ids();

#ifdef USE_KINETO
using trace_t = libkineto::CpuTraceBuffer;
using interface_trace_t = libkineto::ActivityTraceInterface;
using activity_t = libkineto::GenericTraceActivity;
#else
struct DummyTraceBuffer {};
struct DummyTraceInterface {};

using trace_t = DummyTraceBuffer;
using interface_trace_t = DummyTraceBuffer;
struct activity_t;
#endif // USE_KINETO

void addMetadata(
    activity_t* activity,
    const std::string& key,
    const std::string& value);

// Wraps: libkineto::CpuTraceBuffer
struct TraceWrapper {
  TraceWrapper(const int64_t start_time, const std::string& name);
  TraceWrapper(TraceWrapper&&) = default;
  TraceWrapper(const TraceWrapper&) = delete;
  ~TraceWrapper();

  // The caller is expected to hold a mutex when calling `addCPUActivity`.
  activity_t* addCPUActivity(
      const std::string& name,
      const libkineto::ActivityType type,
      const DeviceAndResource device_and_resource,
      const uint64_t correlation_id,
      const int64_t start_time,
      const int64_t end_time);

  void transferCpuTrace(int64_t end_time);

  explicit operator bool() const;

  std::unique_ptr<trace_t>& get() {
    return cpu_trace_;
  }

 private:
  std::unique_ptr<trace_t> cpu_trace_;
};

// Wraps libkineto::ActivityTraceInterface
struct ActivityTraceWrapper {
  explicit ActivityTraceWrapper(std::unique_ptr<interface_trace_t>&& trace);
  ActivityTraceWrapper() = default;
  ActivityTraceWrapper(ActivityTraceWrapper&&) = default;
  ActivityTraceWrapper(const ActivityTraceWrapper&) = delete;
  explicit operator bool() const;
  void save(const std::string& path);

  const std::unique_ptr<interface_trace_t>& get() {
    return trace_;
  }

 private:
  std::unique_ptr<interface_trace_t> trace_;
#ifdef USE_KINETO
  bool saved_ = false; // Kineto's save is destructive
#endif
};

using ActivitySet = std::set<torch::autograd::profiler::ActivityType>;
void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config);
void startTrace();
ActivityTraceWrapper stopTrace();
void pushCorrelationId(uint64_t correlation_id);
void pushUserCorrelationId(uint64_t correlation_id);
void popCorrelationId();
void popUserCorrelationId();
void recordThreadInfo();
bool collectivesProfilerExists();

void logInvariantViolation(
    const std::string& assertion,
    const std::string& error,
    const std::string& profile_id,
    const std::string& group_profile_id);

} // namespace kineto
} // namespace impl
} // namespace profiler

namespace autograd {
namespace profiler {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type);

TORCH_API void addMetadataJson(
    const std::string& key,
    const std::string& value);

TORCH_API void profilerStep();

} // namespace profiler
} // namespace autograd
} // namespace torch
